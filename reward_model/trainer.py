import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Sampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from peft import PeftModel
import safetensors
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    Qwen2ForSequenceClassification,
    Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available)

from transformers.trainer import (
    is_sagemaker_mp_enabled,
    is_peft_available,
    is_datasets_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    is_torch_xla_available,
)

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.trainer_pt_utils import nested_detach, find_batch_size
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import RewardTrainer

from geomloss import SamplesLoss
from .utils import get_peft_state_non_lora_maybe_zero_3


# if is_torch_xla_available():
#     import torch_xla.core.xla_model as xm
# else:
#     IS_XLA_FSDPV2_POST_2_2 = False


class PartialEmbeddingUpdateCallback(TrainerCallback):
    """
    这是一个特殊的训练回调类，用于控制模型训练过程中哪些token的嵌入会被更新。
    
    主要用途：
    - 在训练过程中，只允许更新特定的token嵌入（如特殊标记）
    - 其他所有token的嵌入保持不变，回到它们的初始状态
    
    这种机制在以下场景特别有用：
    - 当你想保留预训练模型的大部分知识
    - 只希望模型学习新的特殊token的含义
    - 防止训练过程中破坏已有的token表示
    """
    def __init__(self, special_token_ids):
        """
        初始化这个回调类
        
        参数:
            special_token_ids: 一个列表，包含所有需要被更新的特殊token的ID
                             这些token的嵌入会在训练过程中被更新
                             而其他token的嵌入将保持不变
        """
        super().__init__()
        self.special_token_ids = special_token_ids  # 存储需要更新的特殊token的ID列表
        self.orig_embeds_params = None  # 用于存储所有token的原始嵌入值

    def on_train_begin(self, args, state, control, **kwargs):
        """
        在训练开始时执行
        
        功能:
        - 获取模型的所有token嵌入
        - 保存这些嵌入的副本，作为后续恢复非特殊token的参考
        """
        model = kwargs.get("model")
        self.orig_embeds_params = model.get_input_embeddings().weight.clone().detach()

    def on_step_end(self, args, state, control, **kwargs):
        """
        在每个训练步骤结束时执行
        
        功能:
        - 创建一个掩码，标记哪些token需要保持原样（非特殊token）
        - 使用这个掩码，将非特殊token的嵌入恢复到它们的原始状态
        - 这样确保只有特殊token的嵌入会被更新，其他token保持不变
        """
        model = kwargs.get("model")
        processing_class = kwargs.get("processing_class")

        # 创建一个全1的掩码，表示所有token默认都需要保持原样
        index_no_updates = torch.ones((len(processing_class),), dtype=torch.bool)
        # 将特殊token的位置设为False，表示这些token允许被更新
        index_no_updates[self.special_token_ids] = False
        
        with torch.no_grad():
            # 使用掩码，将非特殊token的嵌入恢复到它们的原始状态
            model.get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[index_no_updates]


class CodeGenRewardModel(Qwen2ForSequenceClassification):
    def __init__(
        self,
        config,
        output_dim:int=1,
        reward_token='last',  # 提取奖励分数的方式：'last'表示最后一个token，'mean'表示平均值，'special'表示特殊token
        special_token_ids=None  # 特殊token的ID列表，用于基于这些token输出计算奖励分数
    ):
        config.num_labels = output_dim
        assert config.num_labels == 1 # 确保我们的打分模型只给一个维度打分
        super().__init__(config)

        self.reward_token = reward_token  # 设置奖励分数提取方式
        self.special_token_ids = special_token_ids  # 设置特殊token ID
        if self.special_token_ids is not None:
            self.reward_token = "special"  # 如果提供了特殊token，则使用特殊token模式

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0] # Update：同样适用于左右padding，但我们代码的设置是 right padding
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        # 根据不同的策略提取奖励分数
        if self.reward_token == "last":
            # 使用最后一个token的输出
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        elif self.reward_token == "mean":
            # 使用所有有效token的平均值
            valid_lengths = torch.clamp(last_non_pad_token, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            # 使用特殊token的输出
            # 创建特殊token的掩码
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            # 获取特殊token的输出
            pooled_logits = logits[special_token_mask, ...]
            
            # 重塑输出维度
            pooled_logits = pooled_logits.view(batch_size, 1, -1)   # [B, 1, N] assert 1 attributes
            # if self.output_dim == 3:
            #     # 如果是3维输出，使用对角线元素
            #     pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError("Invalid reward_token")

        # print("pooled_logits", pooled_logits)
        
        return {"logits": pooled_logits}

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        # if not return_dict:
        #     output = (pooled_logits,) + transformer_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )


class CodeGenTrainer(Trainer):
    """
    自定义训练器类
    主要功能：
    1. 实现自定义的优化器创建，支持不同模块使用不同的学习率
    2. 实现自定义的损失计算，结合MSE损失和Sinkhorn距离
    3. 实现自定义的模型保存策略
    4. 支持特殊token的训练和保存
    """
    def __init__(self, *args, **kwargs):
        """
        初始化训练器
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        super().__init__(*args, **kwargs)

    # def create_optimizer(self):
    #     """
    #     创建优化器
    #     主要功能：
    #     1. 为不同模块(视觉、合并等)设置不同的学习率
    #     2. 实现参数分组和权重衰减
    #     3. 支持特殊token的学习率设置
    #     4. 处理不同参数组的权重衰减策略
    #     """
    #     # 检查是否启用了SageMaker多进程训练
    #     if is_sagemaker_mp_enabled():
    #         return super().create_optimizer()

    #     opt_model = self.model

    #     if self.optimizer is None:
    #         # 获取需要权重衰减的参数名称
    #         decay_parameters = self.get_decay_parameter_names(opt_model)
    #         decay_parameters = [name for name in decay_parameters if "bias" not in name]

    #         # 如果没有特殊学习率，使用默认参数组
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
    #                 "weight_decay": 0.0,
    #             },
    #         ]

    #         # 处理特殊token的参数组
    #         if self.model.special_token_ids:
    #             special_token_embeddings = opt_model.get_input_embeddings().weight
    #             special_token_embeddings.requires_grad = True

    #             optimizer_grouped_parameters.extend([
    #                 {
    #                     "params": [special_token_embeddings],
    #                     "lr": self.args.special_token_lr,
    #                     "weight_decay": 0.0,
    #                 },
    #             ])

    #         # 获取优化器类和参数
    #         optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
    #         # 创建优化器
    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    #     return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        计算损失函数
        主要功能：
        1. 获取模型输出的logits
        2. 计算MSE损失和Sinkhorn距离
        3. 组合两种损失得到最终损失
        Args:
            model: 模型实例
            inputs: 输入数据
            return_outputs: 是否返回输出
            num_items_in_batch: 批次中的项目数量
        Returns:
            计算得到的损失值
        """
        # 获取模型输出
        logits = model(
            return_dict=True,
            **inputs['data']
        )['logits']
    
        loss = None
        if inputs['label'] is not None:
            # 将标签移动到正确的设备
            labels = inputs['label'].to(logits.device)
            # 使用MSE损失
            base_loss = MSELoss()
        else:
            raise NotImplementedError

        # # 计算Sinkhorn距离损失
        # w_loss = SamplesLoss(loss='sinkhorn', p=2, blur=.05)
        # 组合MSE损失和Sinkhorn距离损失
        loss = base_loss(logits.squeeze(), labels.squeeze())

        print(logits, labels, '*'*50)

        # loss = (base_loss(logits.squeeze(), labels.squeeze()) +
        #     0.2 * w_loss(logits, labels.unsqueeze(1))
        # )
        if return_outputs:
            return loss, {'logits': logits}

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        logits = torch.stack(logits).permute(1, 0, 2)   # [B, N]

        labels = inputs["label"]   # [B, N]

        return loss, logits, labels


    def _save_checkpoint(self, model, trial, metrics=None):
        """
        保存检查点
        主要功能：
        1. 保存模型权重
        2. 保存优化器和调度器状态
        3. 保存随机数生成器状态
        4. 支持PEFT模型的特殊保存策略
        Args:
            model: 模型实例
            trial: 超参数搜索试验
            metrics: 评估指标
        """
        if isinstance(self.model, PeftModel):
            # 创建检查点文件夹
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            # 获取输出目录
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存模型
            self.save_model(output_dir, _internal_call=True)

            # 保存非LoRA权重
            if not self.args.save_full_model:
                non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=True)
                torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.pth"))

            # 保存优化器和调度器状态
            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_rng_state(output_dir)

        else:
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        保存模型
        主要功能：
        1. 保存模型权重和配置
        2. 保存tokenizer
        3. 保存训练参数
        4. 支持不同的保存格式（safetensors/pytorch）
        Args:
            output_dir: 输出目录
            state_dict: 模型状态字典
        """
        # 设置输出目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 确定支持的模型类型
        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        
        # 保存模型
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # 处理特殊保存策略
            if not self.args.save_full_model:
                state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, 'model.pth'))

        # 保存tokenizer
        if self.processing_class is not None:
            os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)
            self.processing_class.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
