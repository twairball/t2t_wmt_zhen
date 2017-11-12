#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptation of tensor2tensor/data_generators/generator_utils.py
"""
import tensorflow as tf
import os
import tarfile
import re
from collections import defaultdict

from tensor2tensor.data_generators.generator_utils import maybe_download
from tensor2tensor.data_generators.generator_utils import gunzip_file
from tensor2tensor.data_generators.generator_utils import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators.wmt import _preprocess_sgm

import jieba
jieba.initialize()

def _get_dataset_filename(dataset):
    return dataset[0][1][0]

def _preprocess(line, is_zh=False):
    # remove xml tags
    line = re.sub(r'<.+?>', '', line.strip())
        
    # tokenize with jieba
    if is_zh:
        line = " ".join(jieba.cut(line))

    return line

def hybridSplit(str):
    """ Hybrid split for English/Chinese. 
    E.g. 
        hybridSplit("女人喜欢Louis Vuitton, Hermes")
        =>  ['女人喜欢', 'Louis', 'Vuitton', 'Hermes']
    
    TODO: fix bug that removes punctuation
    """
    regex = r"[\u4e00-\ufaff]+|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(regex, str, re.UNICODE)
    return matches

def _clean_parallel(src_path, ref_path):
    """clean parallel corpus
    - merge or remove blank lines
    - remove lines with high source/ref length ratio
    - remove senteces <min, >max lengths
    """
    # TODO
    pass

def do_files_exist(filepaths):
    return not(False in [tf.gfile.Exists(f) for f in filepaths])

def get_lang(filename):
    """ get language from filename """
    filename = filename.replace("_cn", '.zh').replace("_ch", '.zh').replace("_", ".")
    if filename.endswith(".sgm") or filename.endswith(".txt"):
        return filename.split('.')[-2]
    else:
        return filename.split('.')[-1]

def prepare_data(data_dir, tmp_dir, sources, out_filename="train.tok", use_jieba=True):
    """Preprocess dataset. Download, unarchive and preprocess. 
    Skips processing if file exists. 
    Writes to e.g. /data/t2t_datagen/train.tok.en
    """

    for source in sources:
        url = source[0]
        filename = os.path.basename(url)
        compressed_file = maybe_download(tmp_dir, filename, url)

        for lang_file in source[1]:
            # pre-processed dataset path, e.g. train.tok.en
            lang = get_lang(lang_file)
            _pp = "%s.%s" % (out_filename, lang)
            tf.logging.info("Reading file: %s, preprocessing to target file: %s" % (lang_file, _pp))
            pp_filepath = os.path.join(data_dir, _pp)

            # unzip
            filepath = os.path.join(tmp_dir, lang_file)
            if not tf.gfile.Exists(filepath):
                read_type = "r:gz" if filename.endswith("tgz") else "r"
                with tarfile.open(compressed_file, read_type) as corpus_tar:
                    corpus_tar.extractall(tmp_dir)

            # For some datasets a second extraction is necessary.
            if lang_file.endswith(".gz"):
                new_filepath = os.path.join(tmp_dir, lang_file[:-3])
                if tf.gfile.Exists(new_filepath):
                    tf.logging.info(
                        "Subdirectory %s already exists, skipping unpacking" % filepath)
                else:
                    tf.logging.info("Unpacking subdirectory %s" % filepath)
                    gunzip_file(filepath, new_filepath)
                filepath = new_filepath

            # read and clean each line, and write to target
            with tf.gfile.GFile(filepath, mode="r") as source_file:
                with tf.gfile.GFile(pp_filepath, mode="a") as out_file:
                    is_zh = lang == "zh"
                    is_zh = is_zh and use_jieba
                    for line in source_file:
                        line = _preprocess(line.strip(), is_zh)
                        out_file.write(line + "\n")


def get_or_generate_vocab(data_dir,
                          vocab_filename,
                          vocab_size,
                          dataset_filename,
                          _file_byte_budget=5e9,
                          num_iterations=4):
    """Generate a vocabulary from dataset_filename.

    *
        This generator differs from generator_utils.get_or_generate_vocab in that reads from 
        a single preprocessed dataset. 

        Uses file_byte_budget 5e9 default, which gets ~ 32k vocab
    *
    Args: 
        data_dir: The base directory where data and vocab files are stored. 
        vocab_filename: relative filename where vocab file is stored
        vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
        dataset_filename: filename where dataset file is stored

    """
    def generate():
        tf.logging.info("Generating vocab from: %s" % dataset_filename)
        filepath = os.path.join(data_dir, dataset_filename)
        # Use Tokenizer to count the word occurrences.
        with tf.gfile.GFile(filepath, mode="r") as source_file:
            file_byte_budget = _file_byte_budget
            counter = 0
            countermax = int(source_file.size() / file_byte_budget / 2)
            for line in source_file:
                if counter < countermax:
                    counter += 1
                else:
                    if file_byte_budget <= 0:
                        break
                    line = line.strip()
                    file_byte_budget -= len(line)
                    counter = 0
                    yield line


    return get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                     generate(), num_iterations)

def get_or_generate_vocab_inner(data_dir, vocab_filename, vocab_size,
                                generator, num_iterations=1e3):
    """Inner implementation for vocab generators.

    *
        has minimum token count set to 50. 
    *
    Args:
        data_dir: The base directory where data and vocab files are stored. If None,
            then do not save the vocab even if it doesn't exist.
        vocab_filename: relative filename where vocab file is stored
        vocab_size: target size of the vocabulary constructed by SubwordTextEncoder
        generator: a generator that produces tokens from the vocabulary

    Returns:
        A SubwordTextEncoder vocabulary object.
    """
    if data_dir is None:
        vocab_filepath = None
    else:
        vocab_filepath = os.path.join(data_dir, vocab_filename)

    if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
        tf.logging.info("Found vocab file: %s", vocab_filepath)
        vocab = text_encoder.SubwordTextEncoder(vocab_filepath)
        return vocab

    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    token_counts = defaultdict(int)
    for item in generator:
        for tok in tokenizer.encode(text_encoder.native_to_unicode(item)):
            token_counts[tok] += 1

    vocab = text_encoder.SubwordTextEncoder.build_to_target_size(
        vocab_size, token_counts, 50, 1e3, int(num_iterations))

    if vocab_filepath is not None:
        vocab.store_to_file(vocab_filepath)
    return vocab

def bi_vocabs_token_generator(source_path,
                              target_path,
                              source_token_vocab,
                              target_token_vocab,
                              eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.

    This generator assumes the files at source_path and target_path have
    the same number of lines and yields dictionaries of "inputs" and "targets"
    where inputs are token ids from the " "-split source (and target, resp.) lines
    converted to integers using the token_map.

    * 
      This generator differs from tensor2tensor.translate.bi_vocabs_token_generator
      It adds additional logging and saves from tokenizer exceptions 
    *

    Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    source_token_vocab: text_encoder.TextEncoder object.
    target_token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

    Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
    """
    eos_list = [] if eos is None else [eos]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            line_num = 0
            while source and target:                
                try:
                    source_ints = source_token_vocab.encode(source.strip()) + eos_list
                    target_ints = target_token_vocab.encode(target.strip()) + eos_list
                    yield {"inputs": source_ints, "targets": target_ints}
                    source, target = source_file.readline(), target_file.readline()
                    line_num += 1
                    continue
                except:
                    tf.logging.info("[line %d] source: %s\n%s target: %s" % (line_num, source, 
                        " " * 10, target))
                    source, target = source_file.readline(), target_file.readline()
                    line_num += 1
