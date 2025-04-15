import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Tuple, List, Dict

import torch
import torch.utils.data
from peft import PeftModel
import safetensors
from transformers import (
    PreTrainedModel,
)
from transformers.trainer import (
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
)


from transformers.utils import is_peft_available

from utils import get_peft_state_non_lora_maybe_zero_3


from trl import GRPOTrainer


import aiohttp
import asyncio


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        base_reward_url: str,  # 修改为基础URL
        **kwargs
    ):
        self.reward_urls = {
            'comment': f"{base_reward_url}:8003/reward",
            'efficiency': f"{base_reward_url}:8004/reward",
            'functionality': f"{base_reward_url}:8005/reward",
            'modularity': f"{base_reward_url}:8006/reward",
            'robustness': f"{base_reward_url}:8007/reward",
            'simplicity': f"{base_reward_url}:8008/reward",
            'standardization': f"{base_reward_url}:8009/reward"
        }
        kwargs['reward_funcs'] = [self.reward_func]
        super().__init__(model=model, **kwargs)

    async def _async_fetch_reward(self, url: str, prompt: str, completions: List[str]) -> List[float]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"prompts": prompt, "completions": completions}
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Reward API error: {resp.status} - {text}")
                result = await resp.json()
                return result["rewards"]

    async def _async_fetch_all_rewards(self, prompt: str, completions: List[str]) -> List[float]:
        tasks = []
        for url in self.reward_urls.values():
            tasks.append(self._async_fetch_reward(url, prompt, completions))
        
        all_rewards = await asyncio.gather(*tasks)
        # 计算每个completion的平均reward
        num_completions = len(completions)
        avg_rewards = [0.0] * num_completions
        
        for rewards in all_rewards:
            for i in range(num_completions):
                avg_rewards[i] += rewards[i]
        
        # 除以维度数得到平均值
        num_dimensions = len(self.reward_urls)
        avg_rewards = [r / num_dimensions for r in avg_rewards]
        
        return avg_rewards

    def reward_func(self, completions, **kwargs):
        if not completions:
            return []

        prompt = kwargs.get("prompts", "")
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        rewards = asyncio.run(
            self._async_fetch_all_rewards(prompt, completions)
        )
        return rewards
   

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
                non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
                torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.pth"))

            # 保存优化器和调度器状态


            # if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            # self._save_rng_state(output_dir)
           

            # 保存trainer状态
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

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
