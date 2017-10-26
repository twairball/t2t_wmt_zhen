#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils

import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', 
    required=True, help='input directory')
parser.add_argument('-v', '--vocab_filename', 
    required=False, default="vocab.zh", 
    help='vocab filename. e.g. train.tok')
parser.add_argument('-s', '--vocab_size', 
    required=True, type=int, default=32,
    help='vocab size, in k. Default 32k.')
parser.add_argument('-i', '--dataset_filename',
    required=False, default="train.tok.zh", 
    help="data filename. Default=train.tok.zh")
parser.add_argument('-b', '--file_byte_budget',
    required=False, default=1e6,
    help="file_byte_budget")

def main():
    opt = parser.parse_args()
    vocab_size = float(opt.vocab_size) * 1e6
    vocab_filename = "%s.%f" % (opt.vocab_filename, vocab_size)

    data_dir = opt.data_dir
    dataset_filename = opt.dataset_filename
    file_byte_budget = opt.file_byte_budget

    utils.get_or_generate_vocab(data_dir, vocab_filename, vocab_size, 
        dataset_filename, file_byte_budget)


if __name__ == '__main__':
    main()