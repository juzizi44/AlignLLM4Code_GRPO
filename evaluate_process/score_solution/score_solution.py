import pandas as pd
from openai import OpenAI
import os
import json
import time
import re
import prompt_template.system_prompt_for_eval as system_prompt_for_eval
import prompt_template.user_prompt_for_eval_2 as user_prompt_for_eval_2
import prompt_template.user_prompt_for_eval_3 as user_prompt_for_eval_3
import prompt_template.user_prompt_for_eval_4 as user_prompt_for_eval_4
import prompt_template.user_prompt_for_eval_5 as user_prompt_for_eval_5
import prompt_template.user_prompt_for_eval_6 as user_prompt_for_eval_6
from openai_client import OpenAIClient
# from dotenv import load_dotenv
import random
import concurrent.futures
from typing import Dict, List, Any

# 加载环境变量
# load_dotenv()

class SolutionScorer:
    def __init__(self, api_key: str, preference: str, max_index: int = 5,
                 use_ai_test: bool = False, start_index: int = 0, end_index: int = 5):
        self.api_key = api_key
        self.preference = preference
        self.max_index = max_index
        self.use_ai_test = use_ai_test
        self.start_index = start_index
        self.end_index = end_index
        self.clients = self._initialize_clients()

    def _initialize_clients(self) -> Dict[str, OpenAIClient]:
        """初始化OpenAI客户端"""
        clients = {
            "gpt-4o-mini-2024-07-18": OpenAIClient(
                api_key=self.api_key,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini-2024-07-18",
                system_prompt=None,
                temperature=0.8
            ),
        }
        for name, client in clients.items():
            client.system_prompt = system_prompt_for_eval.SYSTEM_PROMPTS.get_agent(self.preference)
        return clients

    @staticmethod
    def fix_redundant_braces(json_str: str) -> str:
        """修复JSON字符串中的冗余花括号"""
        clean_str = re.sub(r'\s', '', json_str)
        left_braces = clean_str.count('{')
        right_braces = clean_str.count('}')
        if right_braces > left_braces:
            original_depth = 0
            for i, char in enumerate(json_str):
                if char == '{': original_depth += 1
                elif char == '}':
                    original_depth -= 1
                    if original_depth == 0:
                        return json_str[:i+1]
        return json_str

    def fix_result(self, result: str) -> str:
        """修复和格式化评分结果"""
        match = re.search(r'(\{.*\})(\n?`{3,})?', result, re.DOTALL)
        json_string = match.group(1).strip() if match else result
        json_string = re.sub(r'`{3,}\s*$', '', json_string)
        json_string = self.fix_redundant_braces(json_string)
        json_string = re.sub(r'"\n\s*\},\n\s*"solution_final_score"', r'"\n}}","solution_final_score"', json_string)
        json_string = re.sub(r'"\n\s*\},\s*\},\n\s*"solution_final_score"', r'"\n}}","solution_final_score"', json_string)
        return self.fix_redundant_braces(json_string)

    def generate_parallel_solutions(self, user_prompt: str) -> Dict[str, str]:
        """并行调用多个模型生成答案"""
        results = {}
        def call_client(name: str, client: OpenAIClient):
            try:
                return name, client.get_answer(user_prompt)
            except Exception as e:
                return name, f"Error: {e}"
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(call_client, name, client): name for name, client in self.clients.items()}
            for future in concurrent.futures.as_completed(futures):
                name, output = future.result()
                results[name] = output
        return results

    def process_results(self, user_prompt: str, index: int, solution_to_model: Dict[str, str]) -> Dict[str, Any]:
        raw = self.generate_parallel_solutions(user_prompt)
        print("\n=== Debug Info ===")
        print("Raw responses:")
        for m, r in raw.items():
            print(f"\nModel {m}:")
            print(r)
            
        parsed = {m: json.loads(self.fix_result(raw[m])) for m in self.clients}
        print("\nParsed responses:")
        for m, p in parsed.items():
            print(f"\nModel {m}:")
            print(json.dumps(p, indent=2))
            
        print("\nSolution to model mapping:")
        print(json.dumps(solution_to_model, indent=2))
        
        evals = {}
        for m, scores in parsed.items():
            print(f"\nProcessing scores for model {m}:")
            print("Scores:", json.dumps(scores, indent=2))
            sol_to_solution = {f"sol{i}": f"solution{i}" for i in range(1, len(solution_to_model) + 1)}
            evals[m] = {solution_to_model[k]: scores.get(sol_to_solution[k], {}).get("solution_final_score")
                        for k in solution_to_model}
            print("Processed evals:", json.dumps(evals[m], indent=2))
            
        print("\n=== End Debug Info ===")
        
        return {"index": index, "preference": self.preference,
                "evaluation": evals, "responses": raw}

    @staticmethod
    def get_last_processed_index(output_file: str) -> int:
        if not os.path.exists(output_file): return -1
        with open(output_file, encoding='utf-8') as f:
            return sum(1 for _ in f) - 1

    @staticmethod
    def format_time(sec: float) -> str:
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"

    def score_solutions(self, input_files: List[str], output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # 读取所有文件
        dfs = [pd.read_json(fp, lines=True) for fp in input_files]
        # 取交集并对齐
        indices = sorted(set.intersection(*(set(df['index']) for df in dfs)))
        dfs = [df[df['index'].isin(indices)] for df in dfs]
        to_process = dfs[0].head(self.max_index)
        print(f"\n需要处理的index列表：{list(to_process['index'])}\n")

        start = time.time()
        with open(output_file, 'a', encoding='utf-8', buffering=1) as out:
            for _, row in to_process.iterrows():
                actual_index = row['index']  # 使用实际的index值
                sols = [df[df['index'] == actual_index]['response'].iloc[0] for df in dfs]
                keys = [f"ans{i+1}" for i in range(len(sols))]
                mapping = dict(zip(keys, sols))
                random.shuffle(keys)
                shuffled = [mapping[k] for k in keys]
                sol_map = {f"sol{i+1}": keys[i] for i in range(len(keys))}

                # 选择正确的 prompt 模板
                if self.use_ai_test and self.preference in ('robustness', 'functionality'):
                    prompt = user_prompt_for_eval_2.USER_PROMPTS.get_prompt(f"{self.preference}_ai_test")
                else:
                    mod = __import__(f"prompt_template.user_prompt_for_eval_{len(input_files)}", fromlist=['USER_PROMPTS'])
                    prompt = mod.USER_PROMPTS.get_prompt(self.preference)
                user_prompt = prompt.format(code_problem=row['prompt'], **{f"solution{i+1}": shuffled[i] for i in range(len(shuffled))})

                # 重试并写入
                for attempt in range(2):
                    try:
                        res = self.process_results(user_prompt, actual_index, sol_map)  # 使用实际的index值
                        out.write(json.dumps(res, ensure_ascii=False) + '\n')
                        break
                    except Exception as e:
                        if attempt == 1:
                            fail = {"index": actual_index, "preference": self.preference,  # 使用实际的index值
                                    "evaluation": {m: 'failed' for m in self.clients}, "responses": {}}
                            out.write(json.dumps(fail, ensure_ascii=False) + '\n')
                        time.sleep(1)
                print(f"Processed index {actual_index}, time: {self.format_time(time.time()-start)}")
        print(f"Done in {self.format_time(time.time()-start)}")


if __name__ == "__main__":
    # 直接用字典配置参数，移除 argparse
    args = {
        'preference': 'efficiency',
        'api_key': os.getenv('OPENAI_API_KEY', ''),
        'input_files': [
            '../generate_solution/output/base_model/base_code_generated.jsonl',
            '../generate_solution/output/dpo/dpo_generated.jsonl',
            '../generate_solution/output/grpo/grpo_generated.jsonl',
            '../generate_solution/output/openai_model/gpt_generated.jsonl'
        ],
        'output_file': './scored_solution/efficiency_scores.jsonl',
        'max_index': 10,
        'use_ai_test': False
    }
    scorer = SolutionScorer(
        api_key=args['api_key'],
        preference=args['preference'],
        max_index=args['max_index'],
        use_ai_test=args['use_ai_test']
    )
    scorer.score_solutions(args['input_files'], args['output_file'])
