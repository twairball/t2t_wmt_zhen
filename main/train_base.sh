#!/bin/bash

# train wmt17 zh-en
mkdir -p ./data/wmt_base
mkdir -p ./train/wmt_base
t2t-trainer \
  --t2t_usr_dir=./data_generators \
  --data_dir=./data/wmt_base \
  --problems=translate_enzh_wmt_base \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=./train/wmt_base \
