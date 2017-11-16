# WMT17 English-Chinese 

This repo is a collection of experiments on WMT17 English-Chinese translation task. 

## Setup

`pip install -r requirements.txt`

## Run

````
# orig wmt zh-en translation task in t2t repo
./main/gen_orig.sh
./main/train_orig.sh

# baseline wmt
./main/gen_base.sh
./main/train_base.sh

# wmt with preprocessing
./main/gen_wmt.sh
./main/train_wmt.sh

````

## Results

1. Original 

This is original WMT17 Zh-En translation problem from tensor2tensor repo. 
It trains only on News Commentary (227k lines) and builds vocab size 8k.

2. Base

Trains using full training dataset (~24MM lines) and uses jieba segmenter for Chinese corpus tokenization. Builds vocab size 32k. 

3. Preprocessed

Trains using cleaned dataset (~18MM lines) after preprocessing: 
- Remove duplicate sentences
- Remove sentences with source/target word ratio > 9
- Remove sentences with source/target word ratio < 0.1111
- Remove short sentences with source length < 5

Uses jieba segmenter for Chinese corpus. Builds vocab size 32k. 