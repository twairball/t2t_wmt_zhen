#!/bin/bash

mkdir -p ./data/base

t2t-datagen --t2t_usr_dir=./data_generators \
  --data_dir=./data/base \
  --problem=translate_enzh_wmt_base
