import json
import aiohttp
import asyncio
from tqdm import tqdm
import os

# 配置：维度及其对应的 reward model API 端口
BASE_REWARD_URL = "http://localhost"
REWARD_DIMENSIONS = {
    "comment": 8003,
    "efficiency": 8004,
    "functionality": 8005,
    "modularity": 8006,
    "robustness": 8007,
    "simplicity": 8008,
    "standardization": 8009
}

# 输入目录路径
DATA_DIR = "/data/AlignLLM4Code_GRPO/evaluate/eval_reward_model/solution_and_label"


async def fetch_reward(session, url, prompt_list, completions):
    async with session.post(
        url,
        json={"prompts": prompt_list, "completions": completions}
    ) as resp:
        text = await resp.text()  # 原始返回
        if resp.status != 200:
            raise RuntimeError(f"Reward API error: {resp.status} - {text}")
        
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to decode JSON response: {text}")

        print(f"🔍 [DEBUG] reward response: {result}")
        return result["rewards"]



async def process_file(reward_name, port):
    input_path = os.path.join(DATA_DIR, f"{reward_name}_preference_pairs.jsonl")
    output_path = os.path.join(DATA_DIR, f"{reward_name}_scored_pairs.jsonl")

    print(f"Processing {reward_name} from {input_path} → {output_path}")

    url = f"{BASE_REWARD_URL}:{port}/reward"
    updated_lines = []

    async with aiohttp.ClientSession() as session:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Scoring {reward_name}"):
            item = json.loads(line)
            prompt = item.get("prompt", "")
            if not prompt:
                raise ValueError(f"Missing prompt in item: {item}")

            completions = [item["answerA"], item["answerB"]]
        

            # ✅ 修复：将 prompt 包装成列表
            scores = await fetch_reward(session, url, [prompt,prompt], completions)

            item[f"{reward_name}A_reward_score"] = scores[0]
            item[f"{reward_name}B_reward_score"] = scores[1]
            updated_lines.append(item)

    with open(output_path, 'w') as f:
        for item in updated_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Finished scoring {reward_name} ({len(updated_lines)} lines)\n")


async def main():
    tasks = []
    for name, port in REWARD_DIMENSIONS.items():
        tasks.append(process_file(name, port))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
