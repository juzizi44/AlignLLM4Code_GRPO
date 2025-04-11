import os
import logging
import argparse
import random

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)

# 配置logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of workers for preprocessing")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model name or path")
parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument("--data_file", type=str, default="/data/AlignLLM4Code_GRPO/DPO/data/train/comment_train.jsonl", help="数据文件路径")
args = parser.parse_args()

# 应用chat template的函数
def apply_chat_template(example, tokenizer):
    prompt_msg = [
        {'content': "", 'role': 'system'},
        {'content': example['instruction'], 'role': 'user'},
    ]
    chosen_msg = [
        {'content': example['chosen'], 'role': 'assistant'}
    ]
    rejected_msg = [
        {'content': example['rejected'], 'role': 'assistant'}
    ]
    example['text_chosen'] = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
    example['text_rejected'] = tokenizer.apply_chat_template(rejected_msg, tokenize=False)
    example['text_prompt'] = tokenizer.apply_chat_template(prompt_msg, tokenize=False)

    return example

if __name__=="__main__":
    # 设置分布式训练环境
    torch.cuda.set_device(0)  # 设置主GPU
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 加载数据集
    logger.info(f"从自定义JSONL文件加载数据集: {args.data_file}")
    dataset = load_dataset("json", data_files=args.data_file)
    raw_datasets = dataset
    
    # 分割数据集为训练集和测试集
    if "test" not in raw_datasets:
        logger.info(f"将数据集分割为训练集和测试集，测试集比例为 {args.test_size}")
        raw_datasets = raw_datasets["train"].train_test_split(
            test_size=args.test_size, 
            seed=args.seed
        )
    
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    
    # 应用chat template
    raw_datasets = raw_datasets.map(
        lambda x: apply_chat_template(x, tokenizer),
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template"
    )
    
    # 重命名列以符合TRL要求
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    
    # 记录一些随机样本
    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    #     logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")
    
    # 限制数据集大小为训练集10条，测试集2条
    dpo_train = raw_datasets["train"]
    dpo_test = raw_datasets["test"]
    # dpo_train = raw_datasets["train"].select(range(10))
    # dpo_test = raw_datasets["test"].select(range(2))
    
    # 加载模型
    instruct_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16
        # device_map="auto",  # 自动处理模型在多GPU上的分配
    )

    # 使用lora微调
    lora_config=LoraConfig(
        peft_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj','v_proj']
    )  # lora参数

    dpo_lora_model=get_peft_model(instruct_model,lora_config) # lora模型

    # 训练参数
    training_args = DPOConfig(
        output_dir="./result/20250410_Qwen2.5-Coder-7B-Instruct-DPO_comment_1000_epoch12",
        per_device_train_batch_size=1, # 训练时每个GPU加载的batch大小
        per_device_eval_batch_size=2, # 评价时每个GPU加载的batch大小
        gradient_accumulation_steps=4,  # 梯度更新的间隔步数
        eval_strategy='steps', # 评估的策略
        logging_steps=10, # 记录日志的间隔
        eval_steps=10, # 评估的间隔
        weight_decay=0.001, # 权重衰减
        # ddp_find_unused_parameters=False,  # 分布式训练参数
        ddp_backend="nccl",  # 使用NCCL后端，支持NVLink
        gradient_checkpointing=False,  # 启用梯度检查点以节省显存
        num_train_epochs=3,
    )

    # 加载训练器
    trainer = DPOTrainer(
        model=dpo_lora_model,
        args=training_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_test,
        processing_class=tokenizer
    )

    # 训练
    trainer.train()


