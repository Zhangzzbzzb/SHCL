#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Running training for SMAP"
python main.py --mode train \
               --win_size 110 \
               --num_epochs 1 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 25 \
               --output_c 25 \
               --dataset SMAP \
               --data_path dataset/SMAP \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt

echo "Running testing for SMAP"
python main.py --mode test \
               --win_size 110 \
               --num_epochs 1 \
               --anormly_ratio 2 \
               --threshold 0.02170261 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 25 \
               --output_c 25 \
               --dataset SMAP \
               --data_path dataset/SMAP \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt
