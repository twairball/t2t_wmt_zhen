#!/bin/bash

# train wmt17 zh-en
mkdir -p ./data/wmt
mkdir -p ./train/wmt
t2t-trainer \
  --t2t_usr_dir=./data_generators \
  --data_dir=./data/wmt \
  --problems=translate_enzh_wmt \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=./train/wmt \
