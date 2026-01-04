#!/bin/bash

torchrun --nproc_per_node=1 train_sot.py \
  --config-name train_sot \
  exp_dir=experiments/sot_test \
  tokenizer=bert-base-chinese \
  data.train_data_config=configs/data_configs/train_data_config.yaml \
  data.valid_data_config=configs/data_configs/valid_data_config.yaml \
  sot_training=true