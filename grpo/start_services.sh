#!/bin/bash

# 启动reward model服务
python /data/AlignLLM4Code_GRPO/grpo/reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/output/cosine/comment_04111604_1000_epoch24_2cuda" \
    --device "cuda:7" \
    --port 8004 &

