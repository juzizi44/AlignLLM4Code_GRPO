# File: base_code_generator.py
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Optional

class BaseCodeGenerator:
    def __init__(
        self,
        base_model_path: str,
        device: str = "cuda:0",
        temperature: float = 0.8,
        padding_side: str = "left"
    ):
        """
        初始化基础模型生成器
        Args:
            base_model_path: 基础模型路径（Hugging Face 格式）
            device: 运行设备，例如 "cuda:0" 或 "cpu"
            temperature: 生成温度，用于控制输出的随机性
            padding_side: 填充方向，可选 "left" 或 "right"
        """
        self.base_model_path = base_model_path
        self.device = torch.device(device)
        self.temperature = temperature

        # 加载分词器并设置填充方向
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = padding_side

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16
        ).to(self.device)

    def generate_from_file(
        self,
        input_file: str,
        output_file: str,
        max_new_tokens: int = 2048,
        num_samples: Optional[int] = None
    ):
        """
        从输入文件批量生成代码并保存到输出文件
        Args:
            input_file: 输入 JSONL 文件路径，每行包含至少 "index" 和 "prompt" 字段
            output_file: 输出 JSONL 文件路径，每行包含生成结果以及原始索引；若路径目录不存在，会自动创建
            max_new_tokens: 每次生成的最大 token 数
            num_samples: 最多处理的样本数量，None 表示处理所有样本
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)

        # 确定进度条总数
        if num_samples is not None:
            total = num_samples
        else:
            # 统计总行数以便完整进度展示
            with open(input_file, 'r', encoding='utf-8') as f_tmp:
                total = sum(1 for _ in f_tmp)

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

                # 构造聊天格式输入
                chat_input = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'system', 'content': ''},
                        {'role': 'user', 'content': prompt}
                    ],
                    tokenize=False
                )

                # 分词并生成
                inputs = self.tokenizer(
                    chat_input,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        top_p=0.85,
                        top_k=40,
                    )

                # 解码响应
                response = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # 写入结果并立即刷新
                result = {"index": idx, "prompt": prompt, "response": response}
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                processed += 1
                progress.update(1)

        progress.close()

# 示例用法
if __name__ == "__main__":
    generator = BaseCodeGenerator(
        base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        device="cuda:6",
        temperature=0.8,
        padding_side="left"
    )
    # num_samples=50 表示最多生成 50 条；若设为 None，则处理全部
    generator.generate_from_file(
        input_file="../../grpo/data/75k/correct_data/grpo_test_data.jsonl",
        output_file="./output/base_model/Qwen2.5-Coder-7B-Instruct_solution.jsonl",
        max_new_tokens=2048,
        num_samples=10
    )
