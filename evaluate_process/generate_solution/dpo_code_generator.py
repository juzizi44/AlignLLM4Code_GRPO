# File: dpo_code_generator.py
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import Optional

class DpoCodeGenerator:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda:0",
        temperature: float = 0.8,
        padding_side: str = "left"
    ):
        """
        初始化 DPO 微调后的代码生成器
        Args:
            base_model_path: 基础模型路径（Hugging Face 格式）
            adapter_path: LoRA Adapter 存放路径
            device: 运行设备（如 "cuda:0" 或 "cpu"）
            temperature: 生成温度，控制输出随机性
            padding_side: 填充方向（"left" 或 "right"）
        """
        self.device = torch.device(device)
        self.temperature = temperature

        # 加载分词器并设置填充方向
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = padding_side

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16
        ).to(self.device)

        # 加载微调 Adapter
        self.adapter = PeftModel.from_pretrained(
            self.model,
            adapter_path
        ).to(self.device)
        self.adapter.eval()

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 2048,
        num_samples: Optional[int] = None
    ):
        """
        从 JSONL 文件加载 prompt 并生成，边生成边保存
        Args:
            input_file: 输入 JSONL 文件路径，每行包含 "index" 和 "prompt"
            output_file: 输出 JSONL 文件路径；若路径不存在则自动创建
            max_new_tokens: 最大生成 token 数
            num_samples: 最多处理条数（None 表示全部）
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file) or '.'
        os.makedirs(output_dir, exist_ok=True)

        # 计算 progress 总数
        if num_samples is not None:
            total = num_samples
        else:
            with open(input_file, 'r', encoding='utf-8') as f_tmp:
                total = sum(1 for _ in f_tmp)

        # 打开文件和进度条
        progress = tqdm(total=total, desc="生成进度")
        processed = 0

        # 逐行读取并生成，同时保存结果
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                if num_samples is not None and processed >= num_samples:
                    break
                data = json.loads(line)
                idx = data.get("index")
                prompt = data.get("prompt", "")

                # 构造聊天模板输入
                chat_input = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': ''},
                        {'role': 'user', 'content': prompt}
                    ],
                    tokenize=False
                )
                inputs = self.tokenizer(
                    chat_input,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.adapter.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        top_p=0.85,
                        top_k=40,
                        repetition_penalty=1.1
                    )

                # 解码并写入
                # 获取输入的token长度
                input_length = inputs.input_ids.shape[1]
                # 只解码新生成的部分（去除输入部分）
                response = self.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
                record = {"index": idx, "prompt": prompt, "response": response}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                processed += 1
                progress.update(1)

        progress.close()

# 示例用法
if __name__ == "__main__":
    generator = DpoCodeGenerator(
        base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path="../../DPO/result/20250412_Qwen2.5-Coder-7B-Instruct-DPO_comment_200_epoch12/checkpoint-480",
        device="cuda:7",
        temperature=0.8,
        padding_side="left"
    )
    generator.generate_from_file(
        input_file="../../grpo/data/75k/correct_data/grpo_test_data.jsonl",
        output_file="./output/dpo/dpo_generated.jsonl",
        max_new_tokens=2048,
        num_samples=10
    )
