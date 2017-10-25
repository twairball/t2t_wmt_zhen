#!/bin/bash

mkdir -p ./data/base

t2t-datagen \
  --data_dir=data/base \
  --problem=translate_enzh_wmt_base