#!/bin/bash

# Set available GPUs
export CUDA_VISIBLE_DEVICES=0


python train.py --train_data_path /home/zhiyi/HKU/data/interpolation_data --device cuda --cudnn_benchmark --pin_memory --train_log_path train_log_exp5 --epoch 20 --batch_size 48 --warmup_steps 1500 --warmup_start_factor 0.1 --lr_schedule cosine_wr 