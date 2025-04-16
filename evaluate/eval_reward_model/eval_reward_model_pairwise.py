import json
import os
from typing import List
from math import isclose

# 你之前定义的 suff_stats, calc_acc, calc_accuracy_with_ties, calc_accuracy_without_ties 可直接复制进来
# def debug_suff_stats(h, m):
#     C, D, Th, Tm, Thm = suff_stats(h, m, -1)
#     print("🔍 Debug suff_stats:")
#     print(f"   ➤ C   (Consistent):  {C}")
#     print(f"   ➤ D   (Discordant):  {D}")
#     print(f"   ➤ Th  (Human tie):   {Th}")
#     print(f"   ➤ Tm  (Model tie):   {Tm}")
#     print(f"   ➤ Thm (Both tie):    {Thm}")
#     total = C + D + Th + Tm + Thm
#     print(f"   ➤ Total comparisons: {total}")
#     if C + D + Tm == 0:
#         print("⚠️  Warning: C + D + Tm == 0. Division by zero will occur in calc_accuracy_without_ties.")


def suff_stats(h, m, epsilon):
    C = D = Th = Tm = Thm = 0
    for hi, mi in zip(h, m):
        if hi == 0 and abs(mi) <= epsilon:
            Thm += 1
        elif hi == 0:
            Th += 1
        elif abs(mi) <= epsilon:
            Tm += 1
        elif hi * mi > 0:
            C += 1
        else:
            D += 1
    return C, D, Th, Tm, Thm

def calc_acc(C, D, Th, Tm, Thm):
    return (C + Thm) / (C + D + Th + Tm + Thm)

def calc_accuracy_with_ties(h, m):
    try:
        C, D, Th, Tm, Thm = suff_stats(h, m, -1)
        sorted_pairs = sorted(zip(h, m), key=lambda x: abs(x[1]))
        acc_star = float('-inf')
        epsilon_star = 0
        epsilon_curr = -1
        current_stat = {'C': C, 'D': D, 'Th': Th, 'Tm': Tm, 'Thm': Thm}
        for hi, mi in sorted_pairs:
            if hi == 0 and abs(mi) < epsilon_curr:
                current_stat['Thm'] -= 1
            elif hi == 0:
                current_stat['Th'] -= 1
            elif abs(mi) < epsilon_curr:
                current_stat['Tm'] -= 1
            elif hi * mi > 0:
                current_stat['C'] -= 1
            else:
                current_stat['D'] -= 1
            epsilon_curr = abs(mi)
            if hi == 0 and abs(mi) <= epsilon_curr:
                current_stat['Thm'] += 1
            elif hi == 0:
                current_stat['Th'] += 1
            elif abs(mi) <= epsilon_curr:
                current_stat['Tm'] += 1
            elif hi * mi > 0:
                current_stat['C'] += 1
            else:
                current_stat['D'] += 1
            acc_curr = calc_acc(**current_stat)
            if acc_curr > acc_star:
                acc_star = acc_curr
                epsilon_star = epsilon_curr
        return acc_star
    except Exception as e:
        print("Error in tie_calibration:", e)
        return 0

def calc_accuracy_without_ties(h, m):
    C, D, Th, Tm, Thm = suff_stats(h, m, -1)
    return C / (C + D + Tm)

def evaluate_dimension_score(dim_name: str, file_path: str):
    h: List[int] = []
    m: List[float] = []

    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if f"{dim_name}A_reward_score" not in item or f"{dim_name}B_reward_score" not in item:
                continue
            if "label" not in item:
                continue

            label = item["label"]
            score_a = item[f"{dim_name}A_reward_score"]
            score_b = item[f"{dim_name}B_reward_score"]

            h.append(label)
            m.append(score_a - score_b)

    print(f"\n==== {dim_name.upper()} ====")
    # debug_suff_stats(h, m)

    acc_with = calc_accuracy_with_ties(h, m)
    acc_without = calc_accuracy_without_ties(h, m)

    print(f"✔ Accuracy with ties:    {acc_with:.4f}")
    print(f"✔ Accuracy without ties: {acc_without:.4f}")


if __name__ == "__main__":
    base_path = "/data/AlignLLM4Code_GRPO/evaluate/eval_reward_model/solution_and_label"
    
    for dim in [
        "comment",
        "efficiency",
        "functionality",
        "modularity",
        "robustness",
        "simplicity",
        "standardization"
    ]:
        file_path = os.path.join(base_path, f"{dim}_scored_pairs.jsonl")
        if os.path.exists(file_path):
            evaluate_dimension_score(dim, file_path)
        else:
            print(f"⚠️ File not found for dimension: {dim}")
