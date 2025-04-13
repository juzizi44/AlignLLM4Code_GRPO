import json
import os
from pathlib import Path

def process_dimension(dimension_name, source_dirs, output_dir, train_size=500):
    """
    处理单个维度的数据分割
    
    Args:
        dimension_name: 维度名称（如'comment', 'efficiency'等）
        source_dirs: 源数据目录列表
        output_dir: 输出目录
        train_size: 每个源文件用于训练的数据量
    """
    train_data = []
    test_data = []
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for source_dir in source_dirs:
        file_path = os.path.join(source_dir, f'{dimension_name}_lcs_split_results.jsonl')
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 添加训练数据
        train_data.extend(lines[:train_size])
        # 添加测试数据
        test_data.extend(lines[train_size:])
    
    # 保存训练数据
    train_output_path = os.path.join(train_dir, f'{dimension_name}_train.jsonl')
    with open(train_output_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    # 保存测试数据
    test_output_path = os.path.join(test_dir, f'{dimension_name}_test.jsonl')
    with open(test_output_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    
    print(f"Processed {dimension_name}:")
    print(f"  Train data: {len(train_data)} lines")
    print(f"  Test data: {len(test_data)} lines")

def main():
    # 设置路径
    base_dir = Path(__file__).parent
    source_dirs = [
        os.path.join(base_dir, '75k', 'lcs_split_data'),
        os.path.join(base_dir, '110k', 'lcs_split_data')
    ]
    output_dir = os.path.join(base_dir, 'final_data')
    
    # 所有需要处理的维度
    dimensions = [
        'comment',
        'efficiency',
        'functionality',
        'modularity',
        'robustness',
        'simplicity',
        'standardization'
    ]
    
    # 处理每个维度
    for dimension in dimensions:
        process_dimension(dimension, source_dirs, output_dir)

if __name__ == '__main__':
    main()