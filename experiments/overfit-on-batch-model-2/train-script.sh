#!/bin/bash

# Run the train.py script 
python ../../train.py  \
                --in_channels 16 \
                --out_channels 1 \
                --init_features 8 \
                --class_one_weight 5 \
                --seed 42 \
                --learning_rate 1e-4 \
                --batch_size 2 \
                --num_steps 100000 \
                --eval_steps 1 \
                --log_freq 20 \
                --dataset_dir   '../../../../datasets/dataset-1' \
                --device 'cuda' \
                --cache_refresh_interval 100000000 \
                --cache_n_images 4
