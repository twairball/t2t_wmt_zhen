#!/bin/bash

mkdir -p ./data/orig

t2t-datagen \
  --data_dir=data/orig \
  --problem=translate_enzh_wmt8k