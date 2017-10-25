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