"""
WMT17 English-Chinese translation

Baseline experiments using full available training data
"""

import os
import tensorflow as tf

# Dependency imports
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import generator_utils

from tensor2tensor.data_generators.translate import TranslateProblem
from tensor2tensor.data_generators.translate import compile_data
from tensor2tensor.data_generators.translate import bi_vocabs_token_generator

import random
import io

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# 227k lines
_NC_TRAIN_DATASETS = [[
        "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
        ["training/news-commentary-v12.zh-en.en",
            "training/news-commentary-v12.zh-en.zh"]]]

# 2000 lines
_NC_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-enzh-src.en.sgm", "dev/newsdev2017-enzh-ref.zh.sgm")
]]   

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
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/casia2015/casia2015_en.txt",
            "cwmt/casia2015/casia2015_ch.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/casict2015/casict2015_en.txt",
             "cwmt/casict2015/casict2015_ch.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/neu2017/NEU_en.txt",
            "cwmt/neu2017/NEU_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2015/datum_en.txt",
            "cwmt/datum2015/datum_ch.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book1_en.txt",
            "cwmt/datum2017/Book1_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book2_en.txt",
            "cwmt/datum2017/Book2_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book3_en.txt",
            "cwmt/datum2017/Book3_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book4_en.txt",
            "cwmt/datum2017/Book4_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book5_en.txt",
            "cwmt/datum2017/Book5_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book6_en.txt",
            "cwmt/datum2017/Book6_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book7_en.txt",
            "cwmt/datum2017/Book7_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book8_en.txt",
            "cwmt/datum2017/Book8_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book9_en.txt",
            "cwmt/datum2017/Book9_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book10_en.txt",
            "cwmt/datum2017/Book10_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book11_en.txt",
            "cwmt/datum2017/Book11_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book12_en.txt",
            "cwmt/datum2017/Book12_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book13_en.txt",
            "cwmt/datum2017/Book13_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book14_en.txt",
            "cwmt/datum2017/Book14_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book15_en.txt",
            "cwmt/datum2017/Book15_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book16_en.txt",
            "cwmt/datum2017/Book16_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book17_en.txt",
            "cwmt/datum2017/Book17_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book18_en.txt",
            "cwmt/datum2017/Book18_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book19_en.txt",
            "cwmt/datum2017/Book19_cn.txt"]],
    ["https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
        ["cwmt/datum2017/Book20_en.txt",
            "cwmt/datum2017/Book20_cn.txt"]]
]

# combined training dataset
_FULL_TRAIN_DATASETS = _NC_TRAIN_DATASETS + \
    _CWMT_TRAIN_DATASETS + \
    _UN_TRAIN_DATASETS

@registry.register_problem
class TranslateEnzhWmtBase(TranslateProblem):
    """WMT17 Zh-En translation. No additional preprocessing done."""

    @property
    def targeted_vocab_size(self):
        return 2**15  # 32k

    @property
    def source_vocab_name(self):
        return "vocab.zhen-en.%d" % self.targeted_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.zhen-zh.%d" % self.targeted_vocab_size

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ZH_TOK

    def get_datasets(self, train):
        return _FULL_TRAIN_DATASETS if train else _NC_TEST_DATASETS

    def generator(self, data_dir, tmp_dir, train):
        datasets = self.get_datasets(train)

        # build vocab from training datasets
        source_datasets = [[item[0], [item[1][0]]] for item in self.get_datasets(train=True)]
        target_datasets = [[item[0], [item[1][1]]] for item in self.get_datasets(train=True)]
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.source_vocab_name, self.targeted_vocab_size,
            source_datasets)
        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
            target_datasets)

        tag = "train" if train else "dev"
        data_path = compile_data(tmp_dir, datasets, "wmt_zhen_tok_%s" % tag)
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

