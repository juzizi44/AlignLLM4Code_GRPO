import random
import json
from dataclasses import dataclass, field, asdict
import torch
import os
import wandb
import regex as re
import time

from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config


from data import DataConfig, create_dataset

from utils import TrainingConfig, ModelConfig, PEFTLoraConfig, load_model_from_checkpoint
from trainer import CustomGRPOTrainer
from transformers import AutoModelForCausalLM
import ast
from trl import GRPOConfig

def save_configs_to_json(data_config, training_args, model_config, peft_lora_config):
    """
    将所有配置保存到JSON文件中
    """
    config_dict = {
        "data_config": asdict(data_config), # 数据配置
        "training_args": asdict(training_args), # 训练参数
        "model_config": asdict(model_config), # 模型配置
        "peft_lora_config": asdict(peft_lora_config), # LoRA配置
    }
    # 删除本地设备相关信息
    del config_dict["training_args"]["local_rank"]
    del config_dict["training_args"]["_n_gpu"]

    save_path = os.path.join(training_args.output_dir, "model_config.json")

    os.makedirs(training_args.output_dir, exist_ok=True)
    print(training_args.output_dir)

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):

    
    # 定义我们关注的模块类型：线性层 和 嵌入层
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding

    # 存储符合条件的模块名称
    lora_module_names = []

    # 遍历模型中的所有模块（包括子模块）
    for name, module in model.named_modules():
        # 如果模块名包含排除列表中的任意关键字，就跳过
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue

        # 如果模块是线性层或嵌入层，则加入候选列表
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    # 如果设置了只选择部分模块（如最后几个），则切片选择
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]

    # 如果启用了 verbose 输出，则打印匹配结果
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")

    # 返回最终筛选出的模块名列表
    return lora_module_names



def set_requires_grad(parameters, requires_grad):
    """
    批量设置某些模型参数是否参与训练（是否需要计算梯度）
    参数:
        parameters: 需要设置的参数列表
        requires_grad: 是否需要计算梯度
    """
    for p in parameters:
        p.requires_grad = requires_grad


def create_model_and_tokenizer(
        model_config, peft_lora_config, training_args,
        cache_dir=None,
    ):


    # 获取模型使用的数据类型（如 float32, float16, bfloat16）
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    ) # 如果是 "float16" → 变成 torch.float16

    # 获取量化配置（如果启用 8bit/4bit 加载模型）
    quantization_config = get_quantization_config(model_config)

    # 构造模型加载参数（是否量化、是否使用 cache、设备映射方式等）
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    ) # gradient_checkpointing是反向传播的时候重算省空间，use_cache是缓存了kv省时间

    # 加载 tokenizer，并设置 pad 方向为右侧（适用于 causal LM）
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path,
                                              padding_side="right",
                                              cache_dir=cache_dir)

    # 如果启用了特殊 token，则添加它并获取其 ID
    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ['<|reward|>']  # 这个 token 用于打分时定位位置信号
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
        local_files_only=False,
        **model_kwargs
    )

    # 如果添加了特殊 token，需要调整模型 embedding 层大小
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(tokenizer)) 

    # 设置模型的数据精度为 bf16 或 fp16（根据训练配置）
    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    # 如果启用了 LoRA 微调
    if peft_lora_config.lora_enable:
        # 自动查找可插入 LoRA 的模块名
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude
        )

        # 构造 LoRA 配置
        peft_config = LoraConfig(
            target_modules=target_modules,  # 要插入 LoRA 的模块名列表
            r=peft_lora_config.lora_r,  # LoRA 的秩（低秩分解参数）
            lora_alpha=peft_lora_config.lora_alpha,  # 缩放系数
            lora_dropout=peft_lora_config.lora_dropout,  # Dropout 防止过拟合
            task_type=peft_lora_config.lora_task_type,  # 任务类型：如 CAUSAL_LM
            use_rslora=peft_lora_config.use_rslora,  # 是否使用 rank-scaling LoRA
            bias="none",  # 不对 bias 做 LoRA 微调
            modules_to_save=peft_lora_config.lora_modules_to_save,  # 只保存这些模块（可选）
        )

        # 将 LoRA 插入模型
        model = get_peft_model(model, peft_config)
    else:
        # 如果未启用 LoRA，则配置设为 None
        peft_config = None

    # 将 tokenizer 的 padding 方式写入 model.config（有助于后续使用）
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.pad_token_id = tokenizer.pad_token_id

    # 返回模型、tokenizer 和 LoRA 配置
    return model, tokenizer, peft_config


def train():

    ## ====> 1: 解析参数
    parser = HfArgumentParser((DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig))
    data_config, training_args, model_config, peft_lora_config = parser.parse_args_into_dataclasses()

    # 检查LoRA配置的有效性
    assert not (peft_lora_config.lora_enable and model_config.freeze_llm), \
        'When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA.'

    if peft_lora_config.lora_enable: # 确保 lora_namespan_exclude 是个列表对象（list），即使它是从字符串读取来的配置值。
        if peft_lora_config.lora_namespan_exclude is not None:
            peft_lora_config.lora_namespan_exclude = ast.literal_eval(peft_lora_config.lora_namespan_exclude)
        else:
            peft_lora_config.lora_namespan_exclude = []

    ## ===> Step 2: 加载和配置模型
    model, tokenizer, peft_config = create_model_and_tokenizer(
        model_config=model_config,
        peft_lora_config=peft_lora_config,
        training_args=training_args,
    )

    ## 加载预训练模型
    if training_args.load_from_pretrained is not None:
        model, checkpoint_step = load_model_from_checkpoint(model, training_args.load_from_pretrained, training_args.load_from_pretrained_step)
    model.train()

    # 配置模型参数梯度
    if peft_lora_config.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        # 设置LLM参数的梯度
        set_requires_grad(model_to_configure.model.parameters(), not model_config.freeze_llm)



    # 设置回归头的梯度

    # set_requires_grad(model_to_configure.score.parameters(), True)

    ## ===> Step 3: 加载和配置数据集
    train_dataset = create_dataset(data_config)
    train_dataset = train_dataset.shuffle(seed=42)

    # 准备验证集
    if training_args.conduct_eval:
        if data_config.meta_data_test is not None:
            random.seed(42)
            valid_dataset = create_dataset(data_config, meta_file=data_config.meta_data_test)
        else:
            dataset = train_dataset.train_test_split(test_size=0.20)
            train_dataset = dataset['train']
            valid_dataset = dataset['test']
    else:
        valid_dataset = None

    print(f"===> Selected {len(train_dataset)} samples for training.")
    print(f"===> Selected {len(valid_dataset)} samples for testing.")

    # 配置数据收集器和训练参数
    num_gpu = int(os.environ.get("WORLD_SIZE", 1))
    

    # data_collator = CodeGenDataCollator(tokenizer)

    # 计算训练步数
    actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_gpu
    total_steps = training_args.num_train_epochs * len(train_dataset) // actual_batch_size
    if training_args.save_epochs is not None:
        training_args.save_steps = round(training_args.save_epochs * len(train_dataset) / actual_batch_size)
    if training_args.eval_epochs is not None:
        training_args.eval_steps = round(training_args.eval_epochs * len(train_dataset) / actual_batch_size)
    if training_args.logging_epochs is not None:
        training_args.logging_steps = round(training_args.logging_epochs * len(train_dataset) / actual_batch_size)

    # 打印训练配置信息
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        print(f"===> Using {num_gpu} GPUs.")
        print(f"===> Total Batch Size: {actual_batch_size}")
        print(f"===> Training Epochs: {training_args.num_train_epochs}")
        print(f"===> Total Steps: {total_steps}")
        print(f"===> Save Steps: {training_args.save_steps}")
        print(f"===> Eval Steps: {training_args.eval_steps}")
        print(f"===> Logging Steps: {training_args.logging_steps}")

    ## ===> Step 4: 保存配置
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config, peft_lora_config)

    print(train_dataset)
    
    ## ===> Step 5: 开始训练
    # 配置回调函数
    # special_token_ids = model.special_token_ids
    # callbacks = []
    # if special_token_ids is not None:
    #     callbacks.append(PartialEmbeddingUpdateCallback(special_token_ids))
    
    
    # 创建训练器并开始训练
    trainer = CustomGRPOTrainer(
        model=model,
        base_reward_url="http://localhost",  # 修改为基础URL，各个reward服务会自动加上对应的端口号
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if training_args.conduct_eval else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_from_checkpoint=os.path.join(training_args.load_from_pretrained, "checkpoint-60")
    # trainer.train()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    
    # 保存最终模型
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, os.path.join(training_args.output_dir, 'final_model.pth'))
        model.config.save_pretrained(training_args.output_dir)

    # 清理分布式训练资源
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 确保所有进程都完成了工作
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()