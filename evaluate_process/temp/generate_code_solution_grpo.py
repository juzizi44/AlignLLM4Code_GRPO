import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import multiprocessing as mp
import math
import safetensors
from grpo.utils import get_peft_state_non_lora_maybe_zero_3, _adjust_state_dict_keys, _insert_adapter_name_into_state_dict

def init_model(base_model_path, adapter_path, device, use_adapter):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=None
    ).to(device)

    if use_adapter:
        model_state_dict = base_model.state_dict()
        lora_ckpt = os.path.join(adapter_path, "adapter_model.safetensors")
        non_lora_ckpt = os.path.join(adapter_path, "non_lora_state_dict.pth")

        if os.path.exists(lora_ckpt):
            lora_state_dict = safetensors.torch.load_file(lora_ckpt)
            lora_state_dict, has_adapter_name = _adjust_state_dict_keys(lora_state_dict, model_state_dict.keys())
            if not has_adapter_name:
                lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")
            model_state_dict.update(lora_state_dict)

        if os.path.exists(non_lora_ckpt):
            non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
            non_lora_state_dict, _ = _adjust_state_dict_keys(non_lora_state_dict, model_state_dict.keys())
            model_state_dict.update(non_lora_state_dict)

        base_model.load_state_dict(model_state_dict, strict=False)

    return tokenizer, base_model

def batch_generate_code(prompts, tokenizer, model, device, max_new_tokens=1024, batch_size=4):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        messages_list = [
            [
                {'content': "", 'role': 'system'},
                {'content': prompt, 'role': 'user'}
            ] for prompt in batch_prompts
        ]
        prompts_text = [tokenizer.apply_chat_template(m, tokenize=False) for m in messages_list]
        inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.85,
                top_k=40,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1
            )

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(responses)
    return results

def generate_worker(prompts, base_model_path, adapter_path, device, use_adapter, return_dict, key):
    tokenizer, model = init_model(base_model_path, adapter_path, device, use_adapter)
    results = batch_generate_code(prompts, tokenizer, model, device, batch_size=2)
    return_dict[key] = results

def main():
    base_model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path = "/data/AlignLLM4Code_GRPO/grpo/output_model/20250414/checkpoint-204"
    file1 = "/data/AlignLLM4Code_GRPO/grpo/data/75k/correct_data/grpo_test_data.jsonl"
    file2 = "/data/AlignLLM4Code_GRPO/grpo/data/110k/correct_data/grpo_test_data.jsonl"
    output_file = "generated_solution/model_comparison_results.jsonl"

    data_list = []
    for path in [file1, file2]:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                data_list.append(json.loads(line))

    prompts = [item["prompt"] for item in data_list]

    print("\n使用 multiprocessing 启动 base 和 adapter 模型生成...")
    manager = mp.Manager()
    return_dict = manager.dict()

    p1 = mp.Process(target=generate_worker, args=(prompts, base_model_path, adapter_path, "cuda:4", False, return_dict, "base"))
    p2 = mp.Process(target=generate_worker, args=(prompts, base_model_path, adapter_path, "cuda:5", True, return_dict, "adapter"))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    base_outputs = return_dict["base"]
    adapter_outputs = return_dict["adapter"]

    with open(output_file, "w", encoding='utf-8') as f:
        for i, data in enumerate(tqdm(data_list, desc="Writing results")):
            result = {
                "index": data["index"],
                "prompt": data["prompt"],
                "answer1": base_outputs[i],
                "answer2": adapter_outputs[i]
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n所有结果已保存到: {output_file}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
