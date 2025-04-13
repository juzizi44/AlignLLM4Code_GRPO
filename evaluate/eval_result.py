import pandas as pd
from openai import OpenAI
import os
import json
import time
import argparse
import re
import prompt_template.system_prompt_for_eval as system_prompt_for_eval
import prompt_template.user_prompt_for_eval as user_prompt_for_eval
from openai_client import OpenAIClient
from dotenv import load_dotenv
import random
import concurrent.futures
# 加载环境变量
load_dotenv()


# 解析命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate code solutions using OpenAI API')
    parser.add_argument('--preference', required=True, choices=['comment', 'efficiency', 'modularity', 'simplicity', 'robustness', 'functionality', 'standardization'])
    parser.add_argument('--base_dataset', required=True, choices=['75k', '110k'])
    parser.add_argument('--max_index', type=int, default=5)
    parser.add_argument('--api_key', required=True, help='API key for OpenAI')
    parser.add_argument('--use_ai_test', action='store_true', help='是否使用AI生成的测试用例')
    parser.add_argument('--start_index', type=int, default=0, help='起始索引值')
    parser.add_argument('--end_index', type=int, default=5, help='结束索引值')

    return parser.parse_args()

def fix_redundant_braces(json_str):
   
    # 去掉所有空白字符以便计算
    clean_str = re.sub(r'\s', '', json_str)
    
    # 计算左右花括号数量
    left_braces = clean_str.count('{')
    right_braces = clean_str.count('}')
    
    
    # 如果右花括号过多
    if right_braces > left_braces:
        # 找到最后一个有效的右花括号位置
        depth = 0
        last_valid_pos = 0
        
        for i, char in enumerate(clean_str):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    last_valid_pos = i
                    
        # 使用原始JSON字符串直到最后一个有效的右花括号
        # 先找到原始字符串中对应位置
        original_depth = 0
        original_last_pos = 0
        
        for i, char in enumerate(json_str):
            if char == '{':
                original_depth += 1
            elif char == '}':
                original_depth -= 1
                if original_depth == 0:
                    original_last_pos = i
                    break
        
        return json_str[:original_last_pos+1]
    
    return json_str


def fix_result(result):

    answer_content = result
    # 用正则表达式提取 JSON
    # 用正则表达式提取 JSON，允许结尾有两行或更多的反引号
    match = re.search(r'(\{.*\})(\n?`{3,})?', answer_content, re.DOTALL)

    # 提取出 JSON 字符串
    json_string = match.group(1).strip() if match else answer_content

    # # 删除结尾的反引号
    json_string = re.sub(r'`{3,}\s*$', '', json_string)
    json_string = fix_redundant_braces(json_string)
    
    pattern1 = r'"\n\s*\},\n\s*"solution_final_score"'
    replacement1 = r'"\n}},"solution_final_score"'
    json_string = re.sub(pattern1, replacement1, json_string)
    pattern2 = r'"\n\s*\},\s*\},\n\s*"solution_final_score"'
    replacement2 = r'"\n}},"solution_final_score"'
    json_string = re.sub(pattern2, replacement2, json_string)
    json_string = fix_redundant_braces(json_string)
    # print()
    print(json_string)
    
   

    return json_string

# 并行调用多个模型
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


def process_results(user_prompt, index1, lang, clients, solution_to_model,preference):


    result = generate_parallel_solutions(user_prompt, clients)  # 生成不同模型的答案
    print("//////result"*20,result)

   
    # 处理各个模型的结果
    scores = {}
    for model_name in clients.keys():
        scores[model_name] = json.loads(fix_result(result[model_name]))
    
    # 解析评分，将solution1~solution6替换为实际模型名称
    restored_scores = {}
    for model_name, model_scores in scores.items():
        print(model_name,"//////model_scores"*20,model_scores)
        restored_scores[model_name] = {}
        for solution_key, actual_model in solution_to_model.items():
            # 使用实际模型名称作为键
            print("solution_key",solution_key)
            print("actual_model",actual_model)
            print("model_scores",model_scores)
            print("model_scores.get(solution_key, {})",model_scores.get(solution_key, {}))
            solution_score = model_scores.get(solution_key, {}).get("solution_final_score", None)
            print("solution_score",solution_score)
            restored_scores[model_name][actual_model] = solution_score if solution_score is not None else "No score available"
    
    # 构造最终 JSON 结果
    result_json = {
        "index": index1,
        "preference":preference,
        "programming_language": lang,
        "evaluation": restored_scores,
        "responses": {model_name: result[model_name] for model_name in clients.keys()},
    }
    
    return result_json


# 获取 `output_file` 中的最后一个索引
# 获取 `output_file` 中的最后一个索引
def get_last_processed_index(output_file):
    if not os.path.exists(output_file):
        return -1  # 如果文件不存在，从头开始

    last_index = -1
    with open(output_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):  # `line_number` 是行号
            last_index = line_number  # 每一行的行号更新为最大值
            

    return last_index - 1

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
        client.system_prompt = system_prompt_for_eval.SYSTEM_PROMPTS.get_agent(f"{args.preference}")

    input_file = f"/home/fsq/AlignLLM4Code/evaluate/eval_data/raw_data/{args.base_dataset}/data.jsonl"
    if args.use_ai_test and (args.preference == "robustness" or args.preference == "functionality"):
        ai_test_file = f"/home/fsq/AlignLLM4Code/evaluate/eval_data/result/{args.base_dataset}/ai_test_cases/test_cases_{args.base_dataset}_{args.preference}_{args.start_index}_{args.end_index}.jsonl"
        # 添加文件检查和错误处理
        try:
            if not os.path.exists(ai_test_file):
                raise FileNotFoundError(f"AI test file not found: {ai_test_file}")
            
            # 先检查文件是否为空
            if os.path.getsize(ai_test_file) == 0:
                raise ValueError(f"AI test file is empty: {ai_test_file}")
            
            # 使用更安全的方式读取JSONL文件
            df_ai_test = pd.read_json(ai_test_file, lines=True, orient='records')
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading AI test file: {e}")
            exit(1)
        except Exception as e:
            print(f"Unexpected error while reading AI test file: {e}")
            exit(1)
        
    output_file = f"/home/fsq/AlignLLM4Code/evaluate/eval_data/result/{args.base_dataset}/{args.preference}_score_result.jsonl" 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 获取上次处理的最后索引
    last_index = get_last_processed_index(output_file)
    print(f"Last processed index: {last_index}")

    # 读取输入数据
    df_evol = pd.read_json(input_file, lines=True)

    # 只处理 **索引大于 `last_index`** 的数据
    df_unprocessed = df_evol[df_evol.index > last_index].head(args.max_index)
    print(df_unprocessed)

    start_time = time.time()
    
    # 追加写入文件，开启行缓冲 `buffering=1`
    with open(output_file, "a", encoding="utf-8", buffering=1) as f:
        for index, row in df_unprocessed.iterrows():
            index1 = row["index"]
            instruction = row["code_instruction"]
            lang = row['programming_language']
            answers = row['results']
            print("=" * 50)
            print(f"Processing index: {index1}, Language: {lang}")
            if args.use_ai_test and (args.preference == "robustness" or args.preference == "functionality"):
                ai_test = df_ai_test.loc[index, "test_cases"]
                prompt = user_prompt_for_eval.USER_PROMPTS.get_prompt(f"{args.preference}_ai_test") # todo
                
            else:
                prompt = user_prompt_for_eval.USER_PROMPTS.get_prompt(f"{args.preference}")
            
            
            # 创建带索引的映射
            solution_keys = [
                "untrained_model_output",
                "fine_tuned_output"
            ]

            # 复制原始映射
            solution_mapping = {key: answers[key] for key in solution_keys}

            # 生成随机顺序
            random.shuffle(solution_keys)

            # 重新排列 solutions
            solutions = [solution_mapping[key] for key in solution_keys]

            # 生成 solution1 ~ solution6 到实际模型名的映射
            solution_to_model = {
                f"solution{i+1}": solution_keys[i] for i in range(2)
            }
            if args.use_ai_test and (args.preference == "robustness" or args.preference == "functionality"):
                user_prompt = prompt.format(
                    code_problem = instruction,
                    ai_test = ai_test,
                    solution1=solutions[0],
                    solution2=solutions[1],
            
                )
            else:
                # 生成 prompt
                user_prompt = prompt.format(
                    code_problem = instruction,
                    solution1=solutions[0],
                    solution2=solutions[1],
            
                )

            max_retries = 2
            attempt = 0
            while attempt < max_retries:
                try:
                    attempt += 1
                    # 将solution到模型的映射传递给process_results函数
                    result_json = process_results(user_prompt, index1, lang, clients, solution_to_model,args.preference)
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
                            "programming_language": lang,
                            "evaluation": {model_name: "failed" for model_name in clients.keys()},
                            "responses": {},
                        }
                        json.dump(result_json, f, ensure_ascii=False)
                        f.write("\n")
                    

            elapsed_time = time.time() - start_time
            formatted_time = format_time(elapsed_time)
            print(f"Time elapsed after processing index {index1}: {formatted_time}")

            if index >= args.max_index:
                print(f"Reached {args.max_index} results.")
                break
                

    end_time = time.time()  # Record the end time
    total_elapsed_time = end_time - start_time  # Calculate the total elapsed time
    formatted_total_time = format_time(total_elapsed_time)
    print(f"Program completed successfully.")
    print(f"Total time taken: {formatted_total_time}")
if __name__ == "__main__":
    main()
