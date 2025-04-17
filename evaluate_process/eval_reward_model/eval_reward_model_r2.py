# import json
# import os
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import numpy as np

# # 所有维度名称（与文件名前缀一致）
# REWARD_DIMENSIONS = [
#     "comment",
#     "efficiency",
#     "functionality",
#     "modularity",
#     "robustness",
#     "simplicity",
#     "standardization"
# ]

# INPUT_DIR = "./solution_and_label_r2"

# def compute_r2(file_path):
#     y_true = []
#     y_pred = []

#     with open(file_path, 'r') as f:
#         for line in f:
#             item = json.loads(line)
#             if "final_score" in item and "reward_model_score" in item:
#                 y_true.append(item["final_score"])
#                 y_pred.append(item["reward_model_score"])

#     if not y_true:
#         return None

#     # 拟合线性模型
#     X = np.array(y_pred).reshape(-1, 1)
#     y = np.array(y_true)
#     model = LinearRegression()
#     model.fit(X, y)
#     y_fit = model.predict(X)

#     return r2_score(y, y_fit)

# def main():
#     print("📊 R² scores across dimensions:\n")
#     for dim in REWARD_DIMENSIONS:
#         file_path = os.path.join(INPUT_DIR, f"{dim}_scored.jsonl")
#         if not os.path.exists(file_path):
#             print(f"❌ File not found: {file_path}")
#             continue

#         r2 = compute_r2(file_path)
#         if r2 is None:
#             print(f"⚠️  No valid data in {dim}")
#         else:
#             print(f"✅ {dim:<15}: R² = {r2:.4f}")

# if __name__ == "__main__":
#     main()


import json
import os
import matplotlib.pyplot as plt

# 文件名配置
REWARD_DIMENSIONS = [
    "comment",
    "efficiency",
    "functionality",
    "modularity",
    "robustness",
    "simplicity",
    "standardization"
]

INPUT_DIR = "./solution_and_label_r2"
OUTPUT_DIR = "./reward_scatter_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_scores(file_path):
    x, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if "final_score" in item and "reward_model_score" in item:
                y.append(item["final_score"])           # 纵轴：真实分数
                x.append(item["reward_model_score"])    # 横轴：reward model
    return x, y

def draw_scatter(x, y, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel("Reward Model Score")
    plt.ylabel("Final (True) Score")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    for dim in REWARD_DIMENSIONS:
        file_path = os.path.join(INPUT_DIR, f"{dim}_scored.jsonl")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        x, y = load_scores(file_path)
        if not x:
            print(f"⚠️ No valid data in {dim}")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{dim}_scatter.png")
        draw_scatter(x, y, f"{dim.capitalize()} — Reward vs Final Score", output_path)
        print(f"✅ Saved plot: {output_path}")

if __name__ == "__main__":
    main()

