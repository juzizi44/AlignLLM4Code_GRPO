import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

class CodeGenerator:
    def __init__(self, base_model_path, device="cuda:6"):
        """
        初始化代码生成器
        Args:
            base_model_path: 基础模型路径
            device: 运行设备
        """
        self.base_model_path = base_model_path
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        self.adapter_model = None
        
    def load_adapter(self, adapter_path):
        """
        加载adapter
        Args:
            adapter_path: adapter路径
        """
        self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path).to(self.device)
        self.adapter_model.eval()
        
    def generate_code(self, instruction, use_adapter=True, max_new_tokens=6000):
        """
        生成代码答案
        Args:
            instruction: 输入指令
            use_adapter: 是否使用adapter模型
            max_new_tokens: 最大生成token数
        Returns:
            生成的代码答案
        """
        messages = [
            {'content': "", 'role': 'system'},
            {'content': instruction, 'role': 'user'}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        model = self.adapter_model if use_adapter and self.adapter_model is not None else self.base_model
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def batch_generate(self, input_file, output_file, use_adapter=True):
        """
        批量生成代码答案
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            use_adapter: 是否使用adapter模型
        """
        results = []
        with open(input_file, "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                instruction = data["prompt"]
                response = self.generate_code(instruction, use_adapter=use_adapter)
                results.append({
                    "index": data["index"],
                    "prompt": instruction,
                    "response": response
                })
                
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
                
        print(f"结果已保存到: {output_file}")

def main():
    # 模型配置
    base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path = "/data/AlignLLM4Code_GRPO/grpo/output_model/20250414/checkpoint-117"
    input_file = "/data/AlignLLM4Code_GRPO/grpo/data/testdata.jsonl"
    output_file = "generated_solution/model_comparison_results.jsonl"
    
    # 初始化生成器
    print("正在初始化模型...")
    generator = CodeGenerator(base_model_path, device="cuda:4")
    
    results = []
    # 读取输入文件
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="处理问题"):
            data = json.loads(line)
            prompt = data["prompt"]
            
            # 使用原始模型生成答案
            print(f"\n处理问题 {data['index']}")
            answer1 = generator.generate_code(prompt, use_adapter=False)
            
            # 如果是第一个问题，加载adapter
            if len(results) == 0:
                print("\n加载adapter模型...")
                generator.load_adapter(adapter_path)
            
            # 使用训练后的模型生成答案
            answer2 = generator.generate_code(prompt, use_adapter=True)
            
            # 保存结果
            result = {
                "index": data["index"],
                "prompt": prompt,
                "answer1": answer1,
                "answer2": answer2
            }
            results.append(result)
            
            # 实时写入结果
            with open(output_file, "a") as f_out:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
    print(f"\n所有结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
