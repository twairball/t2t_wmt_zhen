#!/bin/bash

# train wmt17 zh-en orig problem from t2t
mkdir -p ./data/orig
mkdir -p ./train/orig
t2t-trainer \
  --generate_data \
  --data_dir=./data/orig \
  --problems=translate_enzh_wmt8k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=./train/orig \
