#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Running training for PSM"
python main.py --mode train \
               --win_size 90 \
               --num_epochs 8 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 25 \
               --output_c 25 \
               --dataset PSM \
               --data_path dataset/PSM \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt

echo "Running testing for PSM"
python main.py --mode test \
               --win_size 90 \
               --num_epochs 8 \
               --anormly_ratio 1.35 \
               --threshold 0.01111208 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 25 \
               --output_c 25 \
               --dataset PSM \
               --data_path dataset/PSM \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt
