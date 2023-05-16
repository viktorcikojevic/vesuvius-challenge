#!/bin/bash

# Run the train.py script 
python ../../train.py  \
                --in_channels 16 \
                --out_channels 1 \
                --init_features 16 \
                --class_one_weight 10 \
                --seed 42 \
                --learning_rate 1e-4 \
                --batch_size 16 \
                --num_steps 20000000 \
                --eval_steps 64 \
                --log_freq 128 \
                --dataset_dir   '/home/viktor/Documents/kaggle/vesuvius-challenge/datasets/dataset-1' \
                --device 'cuda' \
                --cache_refresh_interval 128 \
                --cache_n_images 256 \
                --device gpu
