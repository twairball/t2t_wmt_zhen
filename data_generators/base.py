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

from tensor2tensor.data_generators.wmt import EOS
from tensor2tensor.data_generators.wmt import TranslateProblem
from tensor2tensor.data_generators.wmt import _compile_data
from tensor2tensor.data_generators.wmt import bi_vocabs_token_generator

import random
import io

from . import utils

from .wmt import _ZHEN_TEST_DATASETS, _ZHEN_TRAIN_DATASETS, \
    _CWMT_TRAIN_DATASETS, _UN_TRAIN_DATASETS

# combined training dataset
_ZHEN_TRAIN_FULL_DATASETS = _ZHEN_TRAIN_DATASETS + \
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
        return "vocab.zhen-zh.%d" % self.targeted_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.zhen-en.%d" % self.targeted_vocab_size

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

