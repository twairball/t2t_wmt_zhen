#!/bin/bash

mkdir -p ./data/wmt

t2t-datagen \
  --data_dir=data/wmt \
  --problem=translate_enzh_wmt