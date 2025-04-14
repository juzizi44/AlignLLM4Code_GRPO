#!/bin/bash

# 启动reward model服务
python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/comment_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:6" \
    --port 8003 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/efficiency_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:6" \
    --port 8004 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/functionality_04121112_1000_epoch24_2cuda_1e-5" \
    --device "cuda:6" \
    --port 8005 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/modularity_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:6" \
    --port 8006 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/robustness_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:7" \
    --port 8007 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/simplicity_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:7" \
    --port 8008 &

python ./reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/final_reward_model/standardization_04122211_1000_epoch24_2cuda_1e-5" \
    --device "cuda:7" \
    --port 8009 &