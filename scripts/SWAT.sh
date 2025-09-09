#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

echo "Running training for SWaT"
python main.py --mode train \
               --win_size 80 \
               --num_epochs 1 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 51 \
               --output_c 51 \
               --dataset SWaT \
               --data_path dataset/SWaT \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt

echo "Running testing for SWaT"
python main.py --mode test \
               --win_size 80 \
               --num_epochs 1 \
               --anormly_ratio 0.3 \
               --threshold 0.0125 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 51 \
               --output_c 51 \
               --dataset SWaT \
               --data_path dataset/SWaT \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt
