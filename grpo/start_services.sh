#!/bin/bash

# 启动reward model服务
python /data/AlignLLM4Code_GRPO/grpo/reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/output/constant_with_warmup/comment_04092054_1000_epoch24" \
    --device "cuda:0" \
    --port 8004 &

