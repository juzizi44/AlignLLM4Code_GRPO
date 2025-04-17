from openai import OpenAI
import time



class OpenAIClient:
    def __init__(self, api_key, base_url, model, system_prompt=None, temperature=0.8):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_answer(self, user_prompt, max_retries=2):
        attempt = 0
        while attempt < max_retries:
            try:
                attempt += 1
                # 打印调试信息
                print("=" * 50)
                print(f"🔹 Attempt: {attempt}")
                print(f"🔹 Model: {self.model}")
                print(f"🔹 Base URL: {self.base_url}")
                # print(f"🔹 API KEY: {self.api_key}")
                # print(f"🔹 System Prompt: {self.system_prompt}")
                # print(f"🔹 User Prompt: {user_prompt}")
                print(f"🔹 Temperature: {self.temperature}")
                print("=" * 50)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=False,
                    temperature=self.temperature,
                )
           
                return response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    print("Retrying...")
                    # print(f"🔹 API KEY: {self.api_key}")
                    time.sleep(1)
                else:
                    print("Max retries reached, returning 'failed'.")
                    return "failed"

class SystemPrompts:
    def __init__(self, **agents):
        self.agents = agents

    def get_agent(self, agent_name):
        return self.agents.get(agent_name, None)
    

class UserPrompts:
    def __init__(self, **prompts):
        self.prompts = prompts

    def get_prompt(self, prompt_name):
        return self.prompts.get(prompt_name, None)