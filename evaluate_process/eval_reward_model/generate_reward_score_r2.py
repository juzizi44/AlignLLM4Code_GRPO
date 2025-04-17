import json
import aiohttp
import asyncio
from tqdm import tqdm
import os

# API 配置
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

# 文件配置
INPUT_DIR = "../../reward_model/raw_data/final_data/test"
OUTPUT_DIR = "./solution_and_label_r2"
MAX_SAMPLES = 200

async def fetch_reward(session, url, prompts, completions):
    async with session.post(
        url,
        json={"prompts": prompts, "completions": completions}
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"Reward API error: {resp.status} - {text}")
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to decode JSON response: {text}")
        return result["rewards"]

async def process_dimension(dimension_name, port):
    input_file = os.path.join(INPUT_DIR, f"{dimension_name}_test.jsonl")
    output_file = os.path.join(OUTPUT_DIR, f"{dimension_name}_scored.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 Processing dimension: {dimension_name} → {output_file}")
    updated_lines = []

    async with aiohttp.ClientSession() as session:
        with open(input_file, 'r') as f:
            lines = f.readlines()[:MAX_SAMPLES]

        for line in tqdm(lines, desc=f"Scoring {dimension_name}"):
            try:
                item = json.loads(line)
                prompt = item.get("code-instruction", "")
                completion = item.get("answer", "")
                raw_score = item.get("final_score", None)

                if not prompt or not completion or raw_score is None:
                    continue

                # 调用 reward model 接口
                rewards = await fetch_reward(session, f"{BASE_REWARD_URL}:{port}/reward", [prompt], [completion])
                item["reward_model_score"] = rewards[0]

                # 替换 final_score
                item["final_score"] = abs((raw_score - 20) / 2)

                updated_lines.append(item)

            except Exception as e:
                print(f"⚠️  Skipped a line due to error: {e}")
                continue

    # 写入新文件
    with open(output_file, 'w') as f:
        for item in updated_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Finished scoring {dimension_name}, total: {len(updated_lines)}")

async def main():
    tasks = []
    for name, port in REWARD_DIMENSIONS.items():
        tasks.append(process_dimension(name, port))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
