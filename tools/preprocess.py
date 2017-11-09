#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess datasets
"""
from __future__ import print_function
import argparse
import re
import jieba
import io

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest="input", help='input file')

def _preprocess(line, is_zh=False):
    # remove xml tags
    line = re.sub(r'<.+?>', '', line.strip())
        
    # tokenize with jieba
    if is_zh:
        line = " ".join(jieba.cut(line))

    return line

def main():
    args = parser.parse_args()
    filename = args.input
    is_zh = filename.endswith(".zh.sgm") or filename.endswith(".zh")
    
    with io.open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = _preprocess(line, is_zh).strip()

            # skip line if blank or xml
            if len(line) > 0:            
                print(line.encode('utf-8'))
    
if __name__ == "__main__":
    main()
