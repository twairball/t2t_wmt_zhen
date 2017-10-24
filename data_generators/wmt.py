"""
WMT17 English-Chinese translation
All rights reserved @twairball, @bartgras, @raynardj

"""
import os
import tensorflow as tf

# Dependency imports
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.wmt import EOS
from tensor2tensor.data_generators.wmt import TranslateProblem

import random
import io

from . import utils

# 227k lines
_ZHEN_TRAIN_DATASETS = [[
        "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
        ["training/news-commentary-v12.zh-en.en",
            "training/news-commentary-v12.zh-en.zh"]]]

# 15,886,041 lines
_UN_TRAIN_DATASETS = [[
        "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/UNv1.0.en-zh.tar.gz",
        ["en-zh/UNv1.0.en-zh.en",
            "en-zh/UNv1.0.en-zh.zh"]]]

# casia2015: 1,050,000 lines
# casict2015: 2,036,833 lines
# datum2015:  1,000,003 lines
# datum2017: 1,999,968 lines
# NEU2017:  2,000,000 lines 
_CWMT_TRAIN_DATASETS = [
    [
        "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        [
            "cwmt/casia2015/casia2015_ch.txt",
            "cwmt/casia2015/casia2015_en.txt",
            "cwmt/casict2015/casict2015_ch.txt",
            "cwmt/casict2015/casict2015_en.txt",
            "cwmt/neu2017/NEU_cn.txt",
            "cwmt/neu2017/NEU_en.txt",
            "cwmt/datum2015/datum_ch.txt",
            "cwmt/datum2015/datum_en.txt",
            "cwmt/datum2017/Book1_cn.txt",
            "cwmt/datum2017/Book1_en.txt",
            "cwmt/datum2017/Book2_cn.txt",
            "cwmt/datum2017/Book2_en.txt",
            "cwmt/datum2017/Book3_cn.txt",
            "cwmt/datum2017/Book3_en.txt",
            "cwmt/datum2017/Book4_cn.txt",
            "cwmt/datum2017/Book4_en.txt",
            "cwmt/datum2017/Book5_cn.txt",
            "cwmt/datum2017/Book5_en.txt",
            "cwmt/datum2017/Book6_cn.txt",
            "cwmt/datum2017/Book6_en.txt",
            "cwmt/datum2017/Book7_cn.txt",
            "cwmt/datum2017/Book7_en.txt",
            "cwmt/datum2017/Book8_cn.txt",
            "cwmt/datum2017/Book8_en.txt",
            "cwmt/datum2017/Book9_cn.txt",
            "cwmt/datum2017/Book9_en.txt",
            "cwmt/datum2017/Book10_cn.txt",
            "cwmt/datum2017/Book10_en.txt",
            "cwmt/datum2017/Book11_cn.txt",
            "cwmt/datum2017/Book11_en.txt",
            "cwmt/datum2017/Book12_cn.txt",
            "cwmt/datum2017/Book12_en.txt",
            "cwmt/datum2017/Book13_cn.txt",
            "cwmt/datum2017/Book13_en.txt",
            "cwmt/datum2017/Book14_cn.txt",
            "cwmt/datum2017/Book14_en.txt",
            "cwmt/datum2017/Book15_cn.txt",
            "cwmt/datum2017/Book15_en.txt",
            "cwmt/datum2017/Book16_cn.txt",
            "cwmt/datum2017/Book16_en.txt",
            "cwmt/datum2017/Book17_cn.txt",
            "cwmt/datum2017/Book17_en.txt",
            "cwmt/datum2017/Book18_cn.txt",
            "cwmt/datum2017/Book18_en.txt",
            "cwmt/datum2017/Book19_cn.txt",
            "cwmt/datum2017/Book19_en.txt",
            "cwmt/datum2017/Book20_cn.txt",
            "cwmt/datum2017/Book20_en.txt"
        ]
    ]
]

# 2000 lines
_ZHEN_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-zhen-ref.en.sgm", "dev/newsdev2017-zhen-src.zh.sgm")
]]   

# Generate data
# t2t-datagen --data_dir=data/wmt --problem=translate_enzh_wmt --t2t_usr_dir=./data_generators

@registry.register_problem
class TranslateEnzhWmt(TranslateProblem):
    """WMT17 Zh-En translation"""

    @property
    def vocab_size(self):
        return 2**15  # 32k

    @property
    def source_vocab_filename(self):
        return "vocab.en.%d" % self.vocab_size

    @property
    def target_vocab_filename(self):
        return "vocab.zh.%d" % self.vocab_size

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ZH_TOK

    ## 
    ## Vocab generator
    ## 
    def get_source_vocab(self, data_dir):
        return utils.get_or_generate_vocab(data_dir, self.source_vocab_filename,
            self.vocab_size, "train.tok.en", num_iterations=4)

    def get_target_vocab(self, data_dir):
        return utils.get_or_generate_vocab(data_dir, self.target_vocab_filename,
            self.vocab_size, "train.tok.zh", num_iterations=4)
    
    ##
    ## generators and stuff
    ##
    def generator(self, data_dir, tmp_dir, train):
        """Generator for graph input. 
        Prepares dataset, build vocab, and feed dataset into generator. 
        """
        # custom pipeline for preparing WMT dataset
        prepare_wmt_data(data_dir, tmp_dir)
        
        # build vocab on training dataset
        source_vocab = self.get_source_vocab(data_dir)
        target_vocab = self.get_target_vocab(data_dir)
        tf.logging.info("[generator] vocab sizes, source: %d, target: %d" %
                        (source_vocab.vocab_size, target_vocab.vocab_size))
        
        data_filename = "train.tok" if train else "dev.tok"   
        source_filepath = os.path.join(data_dir, data_filename + ".en")
        target_filepath = os.path.join(data_dir, data_filename + ".zh")
        tf.logging.info("[generator] filepaths: %s, %s" % (source_filepath, target_filepath))

        return utils.bi_vocabs_token_generator(source_filepath, target_filepath, 
            source_vocab, target_vocab, EOS)

    def feature_encoders(self, data_dir):
        source_token = text_encoder.SubwordTextEncoder(
            os.path.join(data_dir, self.source_vocab_filename))
        target_token = text_encoder.SubwordTextEncoder(
            os.path.join(data_dir, self.target_vocab_filename))
        return {
            "inputs": source_token,
            "targets": target_token,
        }


##
## Preprocessing
##

def prepare_wmt_data(data_dir, tmp_dir):
    """ Prepare datasets for WMT17 ZH-EN.
    Download and preprocesses datasets if not cached on disk. 

    Start with News Commentary as base and hold out a sample dataset,
    for use as dev dataset. 
    
    News Commentary also requires additional cleaning and preprocessing.
    
    Then append additional parallel corpus from CWMT to training. 
    """
    # prepare training dataset if it isn't already available
    train_corpus_paths = [os.path.join(data_dir, "train.tok.%s" % lang) for lang in ["zh", "en"]]
    if not utils.do_files_exist(train_corpus_paths):
        
        # news commentary
        tf.logging.info("[prepare_wmt_data] preparing News Commentary dataset")
        # utils.prepare_data(data_dir, tmp_dir, _ZHEN_TRAIN_DATASETS, 
            # "nc.train.tok")
        utils.prepare_data(data_dir, tmp_dir, _ZHEN_TRAIN_DATASETS, 
            "train.tok")

        # additional preprocessing is required for nc dataset
        # in addition, we hold out 2000 sentences from this dataset
        # we use this as our dev set
        # tf.logging.info("[prepare_wmt_data] running addtl preprocessing")
        # preprocess_nc(data_dir, "nc.train.tok", "train.tok", "dev.tok", 2000)

        # append additional training data using cwmt corpuses
        tf.logging.info("[prepare_wmt_data] appending CWMT datasets")
        utils.prepare_data(data_dir, tmp_dir, _CWMT_TRAIN_DATASETS,
            "train.tok")

        # append additional training data using UN parallel corpuses
        tf.logging.info("[prepare_wmt_data] appending UN parallel datasets")
        utils.prepare_data(data_dir, tmp_dir, _UN_TRAIN_DATASETS,
            "train.tok")

                
                
# combined training dataset
_ZHEN_TRAIN_FULL_DATASETS = _ZHEN_TRAIN_DATASETS + \
    _CWMT_TRAIN_DATASETS + \
    _UN_TRAIN_DATASETS

@registry.register_problem
class TranslateEnzhWmtBase(TranslateProblem):
    """WMT17 Zh-En translation. No additional preprocessing done."""

    @property
    def vocab_size(self):
        return 2**15  # 32k

    @property
    def source_vocab_filename(self):
        return "vocab.en.%d" % self.vocab_size

    @property
    def target_vocab_filename(self):
        return "vocab.zh.%d" % self.vocab_size

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ZH_TOK

    def generator(self, data_dir, tmp_dir, train):
        datasets = _ZHEN_TRAIN_DATASETS if train else _ZHEN_TEST_DATASETS
        source_datasets = [[item[0], [item[1][0]]] for item in _ZHEN_TRAIN_DATASETS]
        target_datasets = [[item[0], [item[1][1]]] for item in _ZHEN_TRAIN_DATASETS]
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.source_vocab_name, self.targeted_vocab_size,
            source_datasets)
        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
            target_datasets)
        tag = "train" if train else "dev"
        data_path = _compile_data(tmp_dir, datasets, "wmt_zhen_tok_%s" % tag)
        return bi_vocabs_token_generator(data_path + ".lang1", data_path + ".lang2",
                                            source_vocab, target_vocab, EOS)

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

