export TOKENIZERS_PARALLELISM=false 
deepspeed --master_port=28507 --include localhost:0,1,2,3 train.py \
    --lora_enable True \
    --freeze_llm False \
    --lora_r 32 \
    --lora_alpha 128 \
    --lora_namespan_exclude "['score', 'rm_head', 'embed_tokens']" \
    --bf16 True \
    --torch_dtype "bfloat16" \
    --num_lora_modules -1 \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --meta_data "./data/final_data/grpo_train_data_100_100.jsonl" \
    --output_dir output_model/20250414 \
    --eval_dim "comment" \
    --output_dim 1 \
    --use_special_tokens False \
    --reward_token "special" \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 12 \
    --num_generations 4 \
    --num_iterations 3 \
    --learning_rate 1e-5 \
    --special_token_lr 1e-5  \
    --report_to tensorboard \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "steps" \
    --logging_steps 10 \
    --eval_epochs 0.5 \
    --save_epochs 1 \
    --max_length 6144 \
    --gradient_checkpointing True \
    --deepspeed ds_config/zero0.json \
    --save_only_model True \
    --save_full_model False \
    --dataloader_num_workers 8 \
    --max_prompt_length 6000 \
    --max_completion_length 6000 \

    # --beta 0.005 \
    # --optim "adamw_8bit" \
    # --adam_beta1 0.9 \
    # --adam_beta2 0.99 \
    # --weight_decay 0.1 \
    # --max_grad_norm 0.1 \
    # --log_on_each_node False \
    # --use_vllm False \
