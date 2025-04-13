from fastapi import FastAPI, HTTPException
import uvicorn
import torch
import requests
from reward.inference import CodeGenRewardInference
from pydantic import BaseModel
from typing import List

# FastAPI应用实例
app = FastAPI()

# 请求数据模型
class RewardRequest(BaseModel):
    prompts: List[str]
    completions: List[str]

# Reward模型类 - 负责实际的模型推理
class RewardModel:
    def __init__(self, model_path, device):
        self.model = CodeGenRewardInference(
            load_from_pretrained=model_path,
            device=device,
            dtype=torch.bfloat16
        )

    
    def get_rewards(self, prompts, completions, batch_size=1):
        try:
            print(f"Model training mode: {self.model.model.training}")
            rewards = []
            for i in range(0, len(prompts), batch_size):
                sub_prompts = prompts[i:i + batch_size]
                sub_completions = completions[i:i + batch_size]
                result = self.model.reward(
                    code_instructions=sub_prompts,
                    answers=sub_completions
                )
                rewards.extend([r["reward"] for r in result])
            return rewards
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Reward服务类 - 单例模式，用于客户端调用
class RewardService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RewardService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, api_url="http://localhost:8000"):
        if not hasattr(self, 'initialized'):
            self.api_url = api_url
            self.initialized = True
            
    def get_rewards(self, prompts, completions):
        try:
            response = requests.post(
                f"{self.api_url}/reward",
                json={
                    "prompts": prompts,
                    "completions": completions
                }
            )
            response.raise_for_status()
            return response.json()["rewards"]
        except Exception as e:
            raise RuntimeError(f"Error getting rewards from API: {str(e)}")

# 全局reward model实例
reward_model = None

# API端点
@app.post("/reward")
async def get_reward(request: RewardRequest):
    global reward_model
    if reward_model is None:
        raise HTTPException(status_code=500, detail="Reward model not initialized")
    
    rewards = reward_model.get_rewards(request.prompts, request.completions)
    return {"rewards": rewards}

# 启动服务器
def start_server(model_path, device, host="0.0.0.0", port=8000):
    global reward_model
    reward_model = RewardModel(model_path, device)
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    start_server(args.model_path, args.device, args.host, args.port) 