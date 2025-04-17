import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import multiprocessing as mp

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
        self.tokenizer.padding_side = "left"
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
        
    def batch_generate_code(self, prompts, use_adapter=True, max_new_tokens=2048, batch_size=4):
        """
        批量生成代码答案
        Args:
            prompts: 输入提示列表
            use_adapter: 是否使用adapter模型
            max_new_tokens: 最大生成token数
            batch_size: 批处理大小
        Returns:
            生成的代码答案列表
        """
        results = []
        model = self.adapter_model if use_adapter and self.adapter_model is not None else self.base_model
        model.eval()  # 确保模型在评估模式
        
        model_type = "adapter" if use_adapter else "base"
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"生成答案 ({model_type} 模型)"):
            batch_prompts = prompts[i:i+batch_size]
            messages_list = [
                [
                    {'content': "", 'role': 'system'},
                    {'content': prompt, 'role': 'user'}
                ] for prompt in batch_prompts
            ]
            
            prompts_text = [self.tokenizer.apply_chat_template(m, tokenize=False) for m in messages_list]
            inputs = self.tokenizer(prompts_text, return_tensors="pt", padding=True).to(self.device)
            
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
            
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(responses)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
        return results

def generate_worker(prompts, base_model_path, adapter_path, device, use_adapter, return_dict, key):
    """
    并行处理的worker函数
    """
    generator = CodeGenerator(base_model_path, device=device)
    if use_adapter:
        generator.load_adapter(adapter_path)
    results = generator.batch_generate_code(prompts, use_adapter=use_adapter)
    return_dict[key] = results

def main():
    # 模型配置
    base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path = "../DPO/result/20250412_Qwen2.5-Coder-7B-Instruct-DPO_comment_200_epoch12/checkpoint-480"
    file1 = "../grpo/data/75k/correct_data/grpo_test_data.jsonl"
    file2 = "../grpo/data/110k/correct_data/grpo_test_data.jsonl"
    output_file = "generated_solution/model_comparison_results_dpo.jsonl"
    
    # 加载数据
    data_list = []
    for file_path in [file1, file2]:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 50:  # 每个文件只取50条数据
                    break
                data_list.append(json.loads(line))
    
    prompts = [item["prompt"] for item in data_list]
    
    print("\n使用 multiprocessing 启动 base 和 adapter 模型生成...")
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # 创建两个进程分别运行base模型和adapter模型
    p1 = mp.Process(target=generate_worker, 
                   args=(prompts, base_model_path, adapter_path, "cuda:4", False, return_dict, "base"))
    p2 = mp.Process(target=generate_worker, 
                   args=(prompts, base_model_path, adapter_path, "cuda:5", True, return_dict, "adapter"))
    
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    base_outputs = return_dict["base"]
    adapter_outputs = return_dict["adapter"]
    
    # 保存结果
    with open(output_file, "w", encoding='utf-8') as f:
        for i, data in enumerate(tqdm(data_list, desc="保存结果")):
            result = {
                "index": data["index"],
                "prompt": data["prompt"],
                "answer1": base_outputs[i],
                "answer2": adapter_outputs[i]
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\n所有结果已保存到: {output_file}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()




# import os
# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# from tqdm import tqdm

# class CodeGenerator:
#     def __init__(self, base_model_path, device="cuda:6"):
#         """
#         初始化代码生成器
#         Args:
#             base_model_path: 基础模型路径
#             device: 运行设备
#         """
#         self.base_model_path = base_model_path
#         self.device = torch.device(device)
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_path,
#             torch_dtype=torch.float16,
#             device_map=None
#         ).to(self.device)
#         self.adapter_model = None
        
#     def load_adapter(self, adapter_path):
#         """
#         加载adapter
#         Args:
#             adapter_path: adapter路径
#         """
#         self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path).to(self.device)
#         self.adapter_model.eval()
        
#     def generate_code(self, instruction, use_adapter=True, max_new_tokens=6000):
#         """
#         生成代码答案
#         Args:
#             instruction: 输入指令
#             use_adapter: 是否使用adapter模型
#             max_new_tokens: 最大生成token数
#         Returns:
#             生成的代码答案
#         """
#         messages = [
#             {'content': "", 'role': 'system'},
#             {'content': instruction, 'role': 'user'}
#         ]
        
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         model = self.adapter_model if use_adapter and self.adapter_model is not None else self.base_model
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=0.3,
#                 top_p=0.85,
#                 top_k=40,
#                 num_beams=1,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 repetition_penalty=1.1
#             )
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         return response

#     def batch_generate(self, input_file, output_file, use_adapter=True):
#         """
#         批量生成代码答案
#         Args:
#             input_file: 输入文件路径
#             output_file: 输出文件路径
#             use_adapter: 是否使用adapter模型
#         """
#         results = []
#         with open(input_file, "r") as f:
#             for line in tqdm(f):
#                 data = json.loads(line)
#                 instruction = data["prompt"]
#                 response = self.generate_code(instruction, use_adapter=use_adapter)
#                 results.append({
#                     "index": data["index"],
#                     "prompt": instruction,
#                     "response": response
#                 })
                
#         with open(output_file, "w") as f:
#             for result in results:
#                 f.write(json.dumps(result) + "\n")
                
#         print(f"结果已保存到: {output_file}")

# def main():
#     # 模型配置
#     base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
#     # adapter_path = "/data/AlignLLM4Code_GRPO/DPO/result/20250410_Qwen2.5-Coder-7B-Instruct-DPO_comment_1000_epoch12/checkpoint-600"
#     adapter_path = "../DPO/result/20250412_Qwen2.5-Coder-7B-Instruct-DPO_comment_200_epoch12/checkpoint-480"

    
#     input_file = "../grpo/data/75k/correct_data/grpo_test_data.jsonl"
#     output_file = "model_comparison_results_dpo.jsonl"
    
#     # 初始化生成器
#     print("正在初始化模型...")
#     generator = CodeGenerator(base_model_path, device="cuda:6")
    
#     results = []
#     # 读取输入文件
#     with open(input_file, "r") as f:
#         for line in tqdm(f, desc="处理问题"):
#             data = json.loads(line)
#             prompt = data["prompt"]
            
#             # 使用原始模型生成答案
#             print(f"\n处理问题 {data['index']}")
#             answer1 = generator.generate_code(prompt, use_adapter=False)
            
#             # 如果是第一个问题，加载adapter
#             if len(results) == 0:
#                 print("\n加载adapter模型...")
#                 generator.load_adapter(adapter_path)
            
#             # 使用训练后的模型生成答案
#             answer2 = generator.generate_code(prompt, use_adapter=True)
            
#             # 保存结果
#             result = {
#                 "index": data["index"],
#                 "prompt": prompt,
#                 "answer1": answer1,
#                 "answer2": answer2
#             }
#             results.append(result)
            
#             # 实时写入结果
#             with open(output_file, "a") as f_out:
#                 f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
#     print(f"\n所有结果已保存到: {output_file}")

# if __name__ == "__main__":
#     main()