#!/bin/bash
python decode.py \
  exp_dir=experiments/diarization_AliMeeting \
  data.test_data_config=configs/AliMeeting/data_configs/test_data_config.yaml \
  checkpoint.iter=15000 \
  checkpoint.avg=5 
