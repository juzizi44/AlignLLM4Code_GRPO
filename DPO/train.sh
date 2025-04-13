#!/bin/bash

# 运行训练脚本
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 direct_preference_optimization.py \
    --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --data_file "/data/AlignLLM4Code_GRPO/DPO/data/train/comment_train.jsonl" \
    --test_size 0.2 \
    --seed 42 \
    --output_dir "./result/0412_Qwen2.5-Coder-7B-Instruct-DPO_comment_1000_epoch12" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 12 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --logging_steps 10 \
    --eval_steps 10 \
    --weight_decay 0.001 \
    --gradient_checkpointing false \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" 