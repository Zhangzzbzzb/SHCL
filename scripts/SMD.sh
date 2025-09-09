#!/bin/bash
export CUDA_VISIBLE_DEVICES=5


train_mode="train"
test_mode="test"


lr=0.0001
gpu=0
batch_size=64
seed=1
input_c=38
output_c=38
d_model=128
e_layers=3
fr=0.4
tr=0.4
seq_size=5
model_save_path=cpt

# Define the list of machines to iterate over
machines=('machine-3-11')

for win_size in $(seq 50 10 130); do
    for epoch in {1..5}; do
        for machine in "${machines[@]}"; do
            echo "Running training with win_size=$win_size, epoch=$epoch on $machine"
            # 训练阶段
            python main.py --mode $train_mode \
                           --win_size $win_size \
                           --num_epochs $epoch \
                           --lr $lr \
                           --gpu $gpu \
                           --batch_size $batch_size \
                           --seed $seed \
                           --input_c $input_c \
                           --output_c $output_c \
                           --dataset $machine \
                           --data_path dataset/Service/$machine/$machine \
                           --d_model $d_model \
                           --e_layers $e_layers \
                           --fr $fr \
                           --tr $tr \
                           --seq_size $seq_size \
                           --model_save_path $model_save_path

            
            for anormly_ratio in $(seq 0.05 0.05 0.4); do
                echo "Running testing with anormly_ratio=$anormly_ratio on $machine"
                # 测试阶段
                python main.py --mode $test_mode \
                               --win_size $win_size \
                               --num_epochs $epoch \
                               --anormly_ratio $anormly_ratio \
                               --lr $lr \
                               --gpu $gpu \
                               --batch_size $batch_size \
                               --seed $seed \
                               --input_c $input_c \
                               --output_c $output_c \
                               --dataset $machine \
                               --data_path dataset/Service/$machine/$machine \
                               --d_model $d_model \
                               --e_layers $e_layers \
                               --fr $fr \
                               --tr $tr \
                               --seq_size $seq_size \
                               --model_save_path $model_save_path
            done
        done
    done
done
