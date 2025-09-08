#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 设置训练阶段的模式
train_mode="train"
test_mode="test"

# 训练的参数范围
win_size_start=40
win_size_end=80
win_size_step=10

epoch_start=1
epoch_end=10
epoch_step=1

# 测试阶段的异常检测阈值范围
anormly_ratio_start=0.2
anormly_ratio_end=1.0
anormly_ratio_step=0.05

# 训练阶段的参数遍历
for win_size in $(seq $win_size_start $win_size_step $win_size_end); do
    for epoch in $(seq $epoch_start $epoch_step $epoch_end); do
        # 设置训练的参数
        echo "Running training with win_size=$win_size and epoch=$epoch"
        
        # 运行训练
        python main.py --mode $train_mode \
                       --win_size $win_size \
                       --num_epochs $epoch \
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

        # 测试阶段的参数遍历
        for anormly_ratio in $(seq $anormly_ratio_start $anormly_ratio_step $anormly_ratio_end); do
            # 设置测试模式
            echo "Running testing with anormly_ratio=$anormly_ratio"

            # 修改 mode 为 test 并运行测试
            python main.py --mode $test_mode \
                           --win_size $win_size \
                           --num_epochs $epoch \
                           --anormly_ratio $anormly_ratio \
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
        done
    done
done
