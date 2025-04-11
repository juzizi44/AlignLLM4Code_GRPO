
目录结构：
.
|-- README.md
|-- __init__.py
|-- __pycache__
|-- grpo
|   |-- __init__.py
|   |-- __pycache__
|   |-- data.py
|   |-- ds_config
|   |   |-- zero0.json
|   |   |-- zero2.json
|   |   `-- zero3.json
|   |-- grpo_output `grpo输出的模型保存在这里`
|   |   `-- 20250409
|   |-- human-eval `GRPO的数据集`
|   |   `-- data
|   |-- reward `reward_service.py会调用这里面的文件，我们不需要运行这里面的文件`
|   |   |-- __init__.py
|   |   |-- __pycache__
|   |   |-- data.py
|   |   |-- inference.py
|   |   |-- prompt_template
|   |   |-- train.py
|   |   |-- trainer.py
|   |   `-- utils.py
|   |-- reward_service.py `使用rewardmodel推理，api提供服务的形式`
|   |-- start_services.sh `启动reward-model的脚本`
|   |-- train.py
|   |-- train.sh
|   |-- trainer.py
|   |-- utils.py
|   `-- wandb
|-- requirements.txt
|
|
|
|-- reward_model `训练reward model的文件夹`
|   |-- __init__.py
|   |-- __pycache__
|   |-- core
|   |-- data.py
|   |-- ds_config
|   |   |-- zero0.json
|   |   |-- zero2.json
|   |   `-- zero3.json
|   |-- environment.yaml
|   |-- inference.py
|   |-- output
|   |   |-- comment_04092054_1000_epoch36
|   |   |-- efficiency_04092054_1000_epoch36
|   |   |-- functionality_04092054_1000_epoch36
|   |   |-- modularity_04092054_1000_epoch36
|   |   |-- robustness_04090944_epoch36
|   |   |-- robustness_04092054_1000_epoch36
|   |   `-- simplicity_04090944_epoch36
|   |-- prompt_template
|   |   |-- __pycache__
|   |   |-- action.py
|   |   `-- code_quality.py
|   |-- raw_data
|   |   |-- 110k
|   |   |-- 75k
|   |   |-- final_data
|   |   `-- process_data.py
|   |-- requirements.txt
|   |-- test.ipynb
|   |-- train.py
|   |-- train.sh
|   |-- trainer.py
|   `-- utils.py
`-- venv `整个项目的环境`
    |-- bin



1. 先运行`start_services.sh`，把reward model加载起来。（注意选一张空的卡）
```
. /data/AlignLLM4Code_GRPO/grpo/start_services.sh
```
（我放在了screen:`1037977.reward_service` ）

测试命令
```
curl -X POST http://localhost:8004/reward \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["def add(a, b):","def add(c, d):"], "completions": ["return a + b","return c + d"]}'
```

2. 然后运行`. train.sh`,开始跑grpo。


目前报错：
```
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 886.00 MiB. GPU 1 has a total capacity of 47.41 GiB of which 319.44 MiB is free. Process 3747386 has 47.09 GiB memory in use. Of the allocated memory 45.56 GiB is allocated by PyTorch, and 972.90 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```