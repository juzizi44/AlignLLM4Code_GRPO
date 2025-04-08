# CUDA_VISIBLE_DEVICES=6
deepspeed --master_port=28500 --include localhost:0,1 train.py \
    --lora_enable True \
    --freeze_llm False \
    --lora_r 32 \
    --lora_alpha 128 \
    --lora_namespan_exclude "['score', 'rm_head', 'embed_tokens']" \
    --bf16 True \
    --torch_dtype "bfloat16" \
    --num_lora_modules -1 \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --meta_data "/data/AlignLLM4Code_GRPO/reward_model/raw_data/75k/lcs_split_data/comment_lcs_split_results.jsonl" \
    --output_dir comment_04072025_epoch24_839_qwen3b \
    --eval_dim "comment" \
    --output_dim 1 \
    --use_special_tokens True \
    --reward_token "special" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 24 \
    --learning_rate 1e-6 \
    --special_token_lr 1e-6 \
    --report_to tensorboard \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "constant_with_warmup" \
    --eval_strategy "steps" \
    --logging_steps 10 \
    --eval_epochs 0.1 \
    --save_epochs 0.25 \
    --max_length 6144 \
    --gradient_checkpointing False \
    --deepspeed ds_config/zero0.json \
    --save_only_model True \
    --save_full_model False \
    --dataloader_num_workers 8

    # --logging_epochs 0.01 \
    # --meta_data_test "/data/zhainx/water/westlake/hq_anno_0326/TextVideoConsistency/action/action_aaab.csv" \
    # --data_dir "/data/zhainx/water/westlake/data/video" \
    # --merger_lr 2e-6 \
    # --tune_merger True \
    # --vision_lr 2e-6 \
    # --sample_type "uniform" \
    # --fps 2 \
    # --max_frame_pixels 200704 \
    # --prompt_template_type "detailed_special" \
    # --vision_lora False \
    # --freeze_vision_tower False \