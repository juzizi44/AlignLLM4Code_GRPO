import pandas as pd
from openai import OpenAI
import os
import json
import time
import argparse
import re
import prompt_template.system_prompt_for_generate_test_cases as system_prompt_for_generate_test_cases
import prompt_template.user_prompt_for_generate_test_cases as user_prompt_for_generate_test_cases
from openai_client import OpenAIClient
from dotenv import load_dotenv
import random
import concurrent.futures
# 加载环境变量
load_dotenv()


# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate ai test cases')
    parser.add_argument('--preference', required=True, choices=['efficiency', 'robustness', 'functionality'])
    parser.add_argument('--base_dataset', required=True, choices=['75k', '110k'])
    parser.add_argument('--start_index', type=int, default=0, help='起始索引值')
    parser.add_argument('--end_index', type=int, default=5, help='结束索引值')
    parser.add_argument('--api_key', required=True, help='API key for OpenAI')
    
    return parser.parse_args()


def generate_parallel_solutions(user_prompt, clients):
    results = {}
    def generate_with_client(name, client):
        try:

            return name, client.get_answer(user_prompt)
        except Exception as e:
            return name, f"Error: {str(e)}"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(generate_with_client, name, client): name for name, client in clients.items()}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result  
    return results


def process_results(user_prompt, index1, clients, preference,instruction):

    result = generate_parallel_solutions(user_prompt, clients)
    model_name = list(clients.keys())[0]
    result_json = {
        "index": index1,
        "preference": preference,
        "prompt": instruction,  # 用户的原始提示/指令
        "test_cases": result[model_name],  # AI生成的测试用例结果
    }
    
    return result_json

# 获取 `output_file` 中的最后一个索引
def get_last_processed_index(output_file,start_index):
    if not os.path.exists(output_file):
        return -1+start_index  # 如果文件不存在，从头开始

    last_index = -1
    with open(output_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):  # `line_number` 是行号
            last_index = line_number  # 每一行的行号更新为最大值
            

    return last_index + start_index - 1

def format_time(seconds):
    """Converts seconds into hours, minutes, and seconds format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def main():
    
    args = parse_arguments()
    clients = {
        "gpt-4o-mini-2024-07-18": OpenAIClient(
            api_key=args.api_key,  # 从参数获取 API Key
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini-2024-07-18",
            system_prompt=None,
            temperature=0.8
        ),
    }
    
    
    # 给系统提示赋值
    for name, client in clients.items():
        client.system_prompt = system_prompt_for_generate_test_cases.SYSTEM_PROMPTS.get_agent(f"{args.preference}")

    input_file = f"/data/AlignLLM4Code_GRPO/evaluate/eval_data/raw_data/{args.base_dataset}/grpo_test_data.jsonl"

  
    output_file = f"/data/AlignLLM4Code_GRPO/evaluate/eval_data/result/{args.base_dataset}/ai_test_cases/test_cases_{args.base_dataset}_{args.preference}_{args.start_index}_{args.end_index}.jsonl" 
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 获取上次处理的最后索引
    last_index = get_last_processed_index(output_file,args.start_index) 
    print(f"Last processed index: {last_index}")

    # 读取输入数据
    df_evol = pd.read_json(input_file, lines=True)

    # 只处理 **索引大于 `last_index`** 的数据
    df_unprocessed = df_evol[df_evol.index > last_index].head(args.end_index - args.start_index + 1)

    start_time = time.time()
    
    # 追加写入文件，开启行缓冲 `buffering=1`
    with open(output_file, "a", encoding="utf-8", buffering=1) as f:
        for index, row in df_unprocessed.iterrows():
            index1 = row["index"]
            instruction = row["prompt"]
            print("=" * 50)
            print(f"Processing line:{index},index: {index1}")
            prompt = user_prompt_for_generate_test_cases.USER_PROMPTS.get_prompt(f"{args.preference}")
            
        
            # 生成 prompt
            user_prompt = prompt.format(
                code_problem = instruction,

            )

            max_retries = 2
            attempt = 0
            while attempt < max_retries:
                try:
                    attempt += 1
                    result_json = process_results(user_prompt, index1, clients,args.preference,instruction)
                    if result_json:
                        
                        json.dump(result_json, f, ensure_ascii=False)
                        f.write("\n")
                        break

                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        print("Retrying...")
                        time.sleep(1)
                    else:
                        print("Max retries reached, skipping index.")
                        result_json = {
                            "index": index1,
                            "preference":args.preference,
                            "prompt": instruction,
                            "test_cases": {},
                        }
              
                        json.dump(result_json, f, ensure_ascii=False)
                        f.write("\n")
                    

            elapsed_time = time.time() - start_time
            formatted_time = format_time(elapsed_time)
            print(f"Time elapsed after processing index {index1}: {formatted_time}")

            if index >= args.end_index:
                print(f"Reached {args.end_index} results.")
                break
                

    end_time = time.time()  # Record the end time
    total_elapsed_time = end_time - start_time  # Calculate the total elapsed time
    formatted_total_time = format_time(total_elapsed_time)
    print(f"Program completed successfully.")
    print(f"Total time taken: {formatted_total_time}")
if __name__ == "__main__":
    main()
