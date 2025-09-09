#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Running training for MSL"
python main.py --mode train \
               --win_size 80 \
               --num_epochs 10 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 55 \
               --output_c 55 \
               --dataset MSL \
               --data_path dataset/MSL \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt

echo "Running testing for MSL"
python main.py --mode test \
               --win_size 80 \
               --num_epochs 10 \
               --anormly_ratio 1 \
               --threshold 0.64737129 \
               --lr 0.0001 \
               --gpu 0 \
               --batch_size 64 \
               --seed 1 \
               --input_c 55 \
               --output_c 55 \
               --dataset MSL \
               --data_path dataset/MSL \
               --d_model 128 \
               --e_layers 3 \
               --fr 0.4 \
               --tr 0.4 \
               --seq_size 5 \
               --model_save_path cpt
