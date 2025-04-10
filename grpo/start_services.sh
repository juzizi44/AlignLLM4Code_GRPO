#!/bin/bash

# 启动reward model服务
python /data/AlignLLM4Code_GRPO/grpo/reward_service.py \
    --model_path "/data/AlignLLM4Code_GRPO/reward_model/output/comment_04092054_1000_epoch36" \
    --device "cuda:1" \
    --port 8004 &

# 等待reward model服务启动
# sleep 10

# # 启动GRPO训练
# deepspeed --master_port=28502 --include localhost:0,1,2 train.py \
#     --lora_enable True \
#     --freeze_llm False \
#     --lora_r 32 \
#     --lora_alpha 128 \
#     --lora_namespan_exclude "['score', 'rm_head', 'embed_tokens']" \
#     --bf16 True \
#     --torch_dtype "bfloat16" \
#     --num_lora_modules -1 \
#     --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --meta_data "/data/AlignLLM4Code_GRPO/grpo/human-eval/data/HumanEval.jsonl" \
#     --output_dir grpo_output/20250409 \
#     --eval_dim "comment" \
#     --output_dim 1 \
#     --use_special_tokens False \
#     --reward_token "special" \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 24 \
#     --learning_rate 1e-6 \
#     --special_token_lr 1e-6 \
#     --report_to tensorboard \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "constant_with_warmup" \
#     --eval_strategy "steps" \
#     --logging_steps 10 \
#     --eval_epochs 0.1 \
#     --save_epochs 0.25 \
#     --max_length 6144 \
#     --gradient_checkpointing False \
#     --deepspeed ds_config/zero0.json \
#     --save_only_model True \
#     --save_full_model False \
#     --dataloader_num_workers 8 \
#     --max_prompt_length 6000 \
#     --max_completion_length 6000 