import json
import numpy as np
from collections import defaultdict
import os

def calculate_dimension_scores(file_path):
    scores_answer1 = []
    scores_answer2 = []
    better_count = 0  # answer2比answer1好的数量
    worse_count = 0   # answer2比answer1差的数量
    equal_count = 0   # answer2和answer1相等的数量
    
    # 读取JSONL文件
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # 从evaluation字段中获取scores
            if 'evaluation' in data:
                for model_scores in data['evaluation'].values():
                    if 'answer1' in model_scores and 'answer2' in model_scores:
                        score1 = float(model_scores['answer1'])
                        score2 = float(model_scores['answer2'])
                        scores_answer1.append(score1)
                        scores_answer2.append(score2)
                        
                        # 比较分数
                        if score2 > score1:
                            better_count += 1
                        elif score2 < score1:
                            worse_count += 1
                        else:
                            equal_count += 1
    
    # 计算平均分
    avg_score1 = float(np.mean(scores_answer1)) if scores_answer1 else 0
    avg_score2 = float(np.mean(scores_answer2)) if scores_answer2 else 0
    
    return avg_score1, avg_score2, better_count, worse_count, equal_count

def calculate_improvement_percentage(score1, score2):
    """计算answer2相对于answer1的提升百分比"""
    if score1 == 0:
        return 0 if score2 == 0 else float('inf')
    return ((score2 - score1) / score1) * 100

def calculate_all_dimensions():
    base_path = "/home/ytan089/AlignLLM4Code_GRPO/evaluate/scored_solution"
    dimensions = [
        "comment",
        "efficiency",
        "functionality",
        "modularity",
        "robustness",
        "simplicity",
        "standardization"
    ]
    
    results = {}
    total_score1 = 0
    total_score2 = 0
    total_better = 0
    total_worse = 0
    total_equal = 0
    
    for dim in dimensions:
        file_path = os.path.join(base_path, f"{dim}_score_result.jsonl")
        if os.path.exists(file_path):
            avg_score1, avg_score2, better, worse, equal = calculate_dimension_scores(file_path)
            improvement = calculate_improvement_percentage(avg_score1, avg_score2)
            results[dim] = {
                "answer1_avg": avg_score1,
                "answer2_avg": avg_score2,
                "improvement": improvement,
                "better_count": better,
                "worse_count": worse,
                "equal_count": equal
            }
            total_score1 += avg_score1
            total_score2 += avg_score2
            total_better += better
            total_worse += worse
            total_equal += equal
    
    total_improvement = calculate_improvement_percentage(total_score1, total_score2)
    results["total"] = {
        "answer1_total": total_score1,
        "answer2_total": total_score2,
        "improvement": total_improvement,
        "total_better": total_better,
        "total_worse": total_worse,
        "total_equal": total_equal
    }
    
    return results

if __name__ == "__main__":
    results = calculate_all_dimensions()
    print("\n评分结果汇总:")
    print("-" * 50)
    
    # 先打印各维度的分数
    for dim, scores in results.items():
        if dim != "total":
            print(f"\n{dim}维度:")
            print(f"Answer 1 平均分: {scores['answer1_avg']:.4f}")
            print(f"Answer 2 平均分: {scores['answer2_avg']:.4f}")
            if scores['improvement'] == float('inf'):
                print("提升百分比: Answer 1 为0，无法计算提升百分比")
            else:
                print(f"Answer 2 相对 Answer 1 的提升百分比: {scores['improvement']:+.2f}%")
            print(f"Answer 2 优于 Answer 1 的样本数: {scores['better_count']}")
            print(f"Answer 2 差于 Answer 1 的样本数: {scores['worse_count']}")
            print(f"Answer 2 等于 Answer 1 的样本数: {scores['equal_count']}")
    
    # 最后打印总分
    print("\n" + "=" * 50)
    print("总分:")
    print(f"Answer 1 总分: {results['total']['answer1_total']:.4f}")
    print(f"Answer 2 总分: {results['total']['answer2_total']:.4f}")
    if results['total']['improvement'] == float('inf'):
        print("总分提升百分比: Answer 1 为0，无法计算提升百分比")
    else:
        print(f"Answer 2 相对 Answer 1 的总分提升百分比: {results['total']['improvement']:+.2f}%")
    print(f"\n所有维度统计:")
    print(f"Answer 2 优于 Answer 1 的总样本数: {results['total']['total_better']}")
    print(f"Answer 2 差于 Answer 1 的总样本数: {results['total']['total_worse']}")
    print(f"Answer 2 等于 Answer 1 的总样本数: {results['total']['total_equal']}")
