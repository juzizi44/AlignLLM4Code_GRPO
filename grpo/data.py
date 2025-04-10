from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path


from datasets import load_dataset

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# from qwen_vl_utils import process_vision_info

# from .prompt_template.code_quality import (
#     COMMENT_PROMPT,
#     EFFICIENCY_PROMPT,
#     MODULARITY_PROMPT,
#     SIMPLICITY_PROMPT,
#     ROBUSTNESS_PROMPT,
#     FUNCTIONALITY_PROMPT,
#     STANDARDIZATION_PROMPT
# )

@dataclass
class DataConfig:
    """
    数据配置类，用于存储数据处理相关的参数
    """
    meta_data: str = "/data/AlignLLM4Code_GRPO/grpo/human-eval/data/HumanEval.jsonl"  # 元数据CSV文件路径
    # data_dir: str = "./data/video"  # 视频数据目录
    meta_data_test: str = None  # 测试集元数据路径
    # max_frame_pixels: int = 240 * 320
    # num_frames: float = None  # 处理的帧数
    # fps: float = 2.0  # 视频帧率
    # p_shuffle_frames: float = 0.0  # 帧随机打乱的概率
    # p_color_jitter: float = 0.0  # 颜色抖动的概率
    eval_dim: str = "comment"  # 评估维度
    # prompt_template_type: str = "none"
    # add_noise: bool = False  # 是否添加噪声
    # sample_type: str = "uniform"  # 采样类型
    max_prompt_length: int = 6000  # 提示文本的最大长度
    max_completion_length: int = 6000  # 补全文本的最大长度


# def build_prompt(dimension, code_problem, solution):
#     if dimension == 'comment':
#         return COMMENT_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'efficiency':
#         return EFFICIENCY_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'modularity':
#         return MODULARITY_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'simplicity':
#         return SIMPLICITY_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'robustness':
#         return ROBUSTNESS_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'functionality':
#         return FUNCTIONALITY_PROMPT.format(code_problem=code_problem, solution=solution)
#     elif dimension == 'standardization':
#         return STANDARDIZATION_PROMPT.format(code_problem=code_problem, solution=solution)
#     else:
#         raise ValueError(f"Invalid template type {dimension}")


# def convert_anno_csv_to_reward_data(
#         example, eval_dims='comment', 
# ):
#     """
#     将JSON标注数据转换为奖励学习所需的数据格式
    
#     参数:
#         example: 包含GSB(Good/Same/Bad)数据的字典
#         data_dir: 视频文件目录路径
#         eval_dims: 评估维度（"action"/"color"等）
#         max_pixels: 视频最大像素数
#         num_frames: 处理的帧数
#         sample_type: 采样类型
#     """

#     data = [
#         {
#             "role": "user",
#             "content": build_prompt(eval_dims, example['code-instruction'])
#         }
#     ]

#     score = torch.tensor(example['final_score'], dtype=torch.float)
#     score = torch.clamp((score - 20)/2., 0., 5.)
    

#     return {'data': data, 'label': score}



def create_dataset(data_config: DataConfig, meta_file=None):
    # 从json中加载数据，然后进行数据转换
    if meta_file is None:
        meta_file = data_config.meta_data
    dataset = load_dataset('json', data_files=meta_file)

    # convert_func = lambda example : convert_anno_csv_to_reward_data(
    #     example, data_config.eval_dim
    # )

    # dataset = dataset.map(convert_func, remove_columns=dataset['train'].column_names, load_from_cache_file=False)
    dataset = dataset['train']

    return dataset


# class CodeGenDataCollator:
#     """
#     QWen视觉语言模型的数据整理器，用于批处理数据
#     """
#     def __init__(self, tokenizer):
#     # def __init__(self, processor, add_noise=False, p_shuffle_frames=0.0, p_color_jitter=0.0):

#         """
#         初始化数据整理器
        
#         参数:
#             processor: 数据处理器
#             add_noise: 是否添加噪声
#             p_shuffle_frames: 帧随机打乱的概率
#             p_color_jitter: 颜色抖动的概率
#         """
#         self.tokenizer = tokenizer

#     def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
#         """
#         对序列进行填充到指定长度
        
#         参数:
#             sequences: 输入序列
#             attention_mask: 注意力掩码
#             max_len: 目标长度
#             padding_side: 填充位置（'right'或'left'）
#         """
#         assert padding_side in ['right', 'left']
#         if sequences.shape[1] >= max_len:
#             return sequences, attention_mask
        
#         pad_len = max_len - sequences.shape[1]
#         padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

#         sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.tokenizer.pad_token_id)
#         attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

#         return sequences_padded, attention_mask_padded

#     def __call__(self, features, enable_noise=True):
#         """
#         预处理输入数据，转换为token序列并返回批次数据
        
#         参数:
#             features: 输入特征
#             enable_noise: 是否启用噪声
#         返回:
#             处理后的批次数据
#         """

#         update_features = []

#         for idx, feature in enumerate(features):
#             # update_features.append(self._clean_message(feature['data']))
#             # 直接使用原始数据
#             update_features.append(feature['data'])

#         # image_inputs, video_inputs = process_vision_info(update_features)
#         # video_inputs = [video_inputs[i].float() / 255.0 for i in range(len(video_inputs))]
#         # do_rescale = False

#         batch = self.tokenizer(
#             text=self.tokenizer.apply_chat_template(update_features, tokenize=False, add_generation_prompt=True),
#             padding=True,
#             return_tensors='pt',
#         )

#         max_len = batch['input_ids'].shape[1]
#         batch['input_ids'], batch['attention_mask'] = self._pad_sequence(
#             batch['input_ids'], batch['attention_mask'], max_len, 'right'
#         )

#         label = torch.stack([torch.tensor(feature['label']) for feature in features])

#         batch = {
#             'data': batch,
#             'label': label
#         }

#         return batch


if __name__ == '__main__':
    """
    主函数：用于测试数据加载和处理流程
    """

    data_args = DataConfig()

    dataset = create_dataset(data_args, meta_file=data_args.meta_data)

    # dataset = load_dataset(
    #     'csv', 
    #     data_files='/data/zhainx/water/westlake/hq_anno_fine_grained/TextVideoConsistency/action/action_aaaa.csv'
    # )
    # print('down!')
    # convert_func = lambda example : convert_anno_csv_to_reward_data(
    #     example, './data/video', 'action'
    # )
    # print(dataset['train'].shape)
    # dataset = dataset.map(convert_func, remove_columns=dataset['train'].column_names, load_from_cache_file=False)
    dataset = iter(dataset)
    for iter in dataset:
        print(iter)
        break