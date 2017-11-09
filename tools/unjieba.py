#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Removes whitespaces from jieba tokenization
"""
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest="input", help='input file')

def unjieba(line):
    # Remove the spaces
    # But 3 spaces between 2 english words will be reduced to one
    return line.rstrip().replace(" " * 3, "<spc>").replace(" ", "").replace("<spc>", " ")

def main():
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        for line in f:
            print(unjieba(line))
    
if __name__ == "__main__":
    main()
