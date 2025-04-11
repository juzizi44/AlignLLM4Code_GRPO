import json
import os
from itertools import combinations

# 定义所有维度
dimensions = [
    'comment',
    'efficiency',
    'functionality',
    'modularity',
    'robustness',
    'simplicity',
    'standardization'
]

# 定义数据集大小
datasets = ['75k', '110k']

for dataset in datasets:
    # 创建输出目录
    output_dir = f"/data/AlignLLM4Code_GRPO/Qwen2.5-SFT-DPO/longest_common_subsequence/{dataset}/dpo_train_data"
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个维度
    for dimension in dimensions:
        # 输入文件路径
        input_file = f'/data/AlignLLM4Code_GRPO/Qwen2.5-SFT-DPO/longest_common_subsequence/{dataset}/lcs_final_data/{dimension}_lcs_merged_results.jsonl'
        # 输出文件路径
        output_file = os.path.join(output_dir, f'{dimension}_dpo_data.jsonl')

        # 处理数据
        processed_data = []

        try:
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    # 判断max_lcs_length是否大于等于2
                    if data.get('max_lcs_length', 0) < 2:
                        continue
                        
                    # 获取lcs_score和max_lcs_sequence
                    lcs_scores = data.get('lcs_score', [])
                    lcs_sequences = data.get('max_lcs_sequence', {})
                    
                    # 确保lcs_scores和lcs_sequences的长度一致
                    if len(lcs_scores) != len(lcs_sequences):
                        continue
                        
                    # 创建模型名称和分数的映射
                    model_scores = {}
                    for i, (model_name, sequence) in enumerate(lcs_sequences.items()):
                        model_scores[model_name] = {
                            'score': lcs_scores[i],
                            'sequence': sequence
                        }
                    
                    # 生成所有可能的模型对组合
                    model_pairs = list(combinations(model_scores.keys(), 2))
                    
                    # 为每对模型创建数据
                    for model1, model2 in model_pairs:
                        score1 = model_scores[model1]['score']
                        score2 = model_scores[model2]['score']
                        sequence1 = model_scores[model1]['sequence']
                        sequence2 = model_scores[model2]['sequence']
                        
                        # 确定chosen和reject
                        if score1 == score2:
                            # 如果分数相等，跳过该数据对
                            continue
                        elif score1 > score2:
                            chosen = sequence1
                            reject = sequence2
                        else:
                            chosen = sequence2
                            reject = sequence1
                        
                        # 创建新的数据项
                        new_data = {
                            "index": data.get('index', 0),
                            "preference": "chosen",
                            "instruction": data.get('code-instruction', ''),
                            "chosen": chosen,
                            "rejected": reject,
                            "dimension": dimension  # 添加维度信息
                        }
                        
                        processed_data.append(new_data)

            # 保存处理后的数据
            with open(output_file, 'w') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')

            print(f"处理完成 {dataset} - {dimension}，共生成 {len(processed_data)} 条数据，保存到 {output_file}")
            
        except FileNotFoundError:
            print(f"警告：文件 {input_file} 不存在，已跳过")
        except Exception as e:
            print(f"处理 {dataset} - {dimension} 时发生错误：{str(e)}")

print("所有数据处理完成！")

import json
import os
from pathlib import Path

# 定义所有维度
dimensions = [
    'comment',
    'efficiency',
    'functionality',
    'modularity',
    'robustness',
    'simplicity',
    'standardization'
]

# 定义数据集大小
datasets = ['75k', '110k']

# 创建输出目录
base_output_dir = "/data/AlignLLM4Code_GRPO/Qwen2.5-SFT-DPO/data"
train_dir = os.path.join(base_output_dir, "train")
test_dir = os.path.join(base_output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 用于存储所有训练数据的列表
all_train_data = []

# 处理每个维度
for dimension in dimensions:
    train_data = []
    test_data = []
    
    # 处理每个数据集
    for dataset in datasets:
        input_file = f'/data/AlignLLM4Code_GRPO/Qwen2.5-SFT-DPO/longest_common_subsequence/{dataset}/dpo_train_data/{dimension}_dpo_data.jsonl'
        
        try:
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            
            # 取前500条作为训练数据
            train_data.extend(data[:500])
            # 剩余的作为测试数据
            test_data.extend(data[500:])
            
        except FileNotFoundError:
            print(f"警告：文件 {input_file} 不存在，已跳过")
            continue
        except Exception as e:
            print(f"处理 {dataset} - {dimension} 时发生错误：{str(e)}")
            continue
    
    # 保存训练数据
    train_file = os.path.join(train_dir, f"{dimension}_train.jsonl")
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # 保存测试数据
    test_file = os.path.join(test_dir, f"{dimension}_test.jsonl")
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"处理完成 {dimension}：")
    print(f"- 训练数据：{len(train_data)} 条")
    print(f"- 测试数据：{len(test_data)} 条")
    
    # 将训练数据添加到总的训练数据集中
    all_train_data.extend(train_data)

# 保存所有维度的训练数据合集
all_train_file = os.path.join(train_dir, "all_dimensions_train.jsonl")
with open(all_train_file, 'w') as f:
    for item in all_train_data:
        f.write(json.dumps(item) + '\n')

print(f"\n所有处理完成！")
print(f"总训练数据集大小：{len(all_train_data)} 条") 