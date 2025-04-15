import json
import itertools
import os
from collections import defaultdict

def process_test_file(input_file, output_file, max_pairs=None):
    """
    处理测试文件，将相同index的数据组合成偏好数据对。
    
    参数:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        max_pairs (int, optional): 最多保存的偏好数据对数量，None表示保存所有
    """
    # 读取输入文件
    print(f"正在读取文件: {input_file}")
    data_by_index = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 按index分组
                data_by_index[item['index']].append(item)
    
    print(f"发现了 {len(data_by_index)} 个不同的index")
    
    # 生成偏好数据对
    preference_pairs = []
    pair_count = 0
    
    for index, items in data_by_index.items():
        # 确保同一index至少有2个数据
        if len(items) < 2:
            continue
            
        # 生成所有可能的组合对
        for item_a, item_b in itertools.combinations(items, 2):
            score_a = item_a.get('final_score', 0)
            score_b = item_b.get('final_score', 0)
            
            # 创建偏好标签: 1表示A优于B，-1表示B优于A，0表示平局
            if score_a > score_b:
                preference = 1
            elif score_a < score_b:
                preference = -1
            else:
                preference = 0
                
            # 创建偏好数据对
            pair = {
                "index": index,
                "answerA": item_a['answer'],
                "answerB": item_b['answer'],
                "modelA": item_a['generation_model'],
                "modelB": item_b['generation_model'],
                "scoreA": score_a,
                "scoreB": score_b,
                "preference": preference
            }
            
            preference_pairs.append(pair)
            pair_count += 1
    
    print(f"生成了 {pair_count} 个偏好数据对")
    
    # 如果指定了最大数量，则只保留前max_pairs条数据
    if max_pairs is not None and max_pairs > 0:
        preference_pairs = preference_pairs[:max_pairs]
        print(f"根据限制，只保留前 {max_pairs} 条偏好数据对")
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in preference_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"已将结果保存到: {output_file}")

def process_all_dimensions(input_dir, output_dir, max_pairs=200):
    """
    处理所有维度的测试文件
    
    参数:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        max_pairs (int): 每个维度最多保存的偏好数据对数量
    """
    # 所有维度的测试文件名
    dimensions = [
        "comment_test.jsonl",
        "efficiency_test.jsonl",
        "functionality_test.jsonl",
        "modularity_test.jsonl",
        "robustness_test.jsonl",
        "simplicity_test.jsonl",
        "standardization_test.jsonl"
    ]
    
    # 处理每个维度的文件
    for dim_file in dimensions:
        input_file = os.path.join(input_dir, dim_file)
        
        # 从文件名提取维度名称
        dimension_name = dim_file.replace("_test.jsonl", "")
        output_file = os.path.join(output_dir, f"{dimension_name}_preference_pairs.jsonl")
        
        # 处理该维度的文件
        process_test_file(input_file, output_file, max_pairs)
        print(f"完成处理维度: {dimension_name}")
        print("-" * 50)

if __name__ == "__main__":
    input_dir = "reward_model/raw_data/final_data/test"
    output_dir = "/home/ytan089/AlignLLM4Code_GRPO/evaluate/eval_reward_model/solution_and_label"
    
    # 处理所有维度
    process_all_dimensions(input_dir, output_dir, max_pairs=200) 