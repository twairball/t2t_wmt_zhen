#!/bin/bash

mkdir -p ./data/wmt_base

t2t-datagen --t2t_usr_dir=./data_generators \
  --data_dir=./data/wmt_base \
  --problem=translate_enzh_wmt_base
