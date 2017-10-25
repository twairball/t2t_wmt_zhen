#!/bin/bash

mkdir -p ./data/wmt

t2t-datagen --t2t_usr_dir=./data_generators \
  --data_dir=./data/wmt \
  --problem=translate_enzh_wmt
