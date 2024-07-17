import datasets
import json
import os
import numpy as np
import random
import torch

from collections import OrderedDict
import transformers
from vllm import LLM, SamplingParams
import pickle as pkl
from transformers import AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


load_dict_path="llama2_7b_sft_halos_2_3/LATEST/policy.pt"
cache_path="/home/wxt/.cache/huggingface/hub"

bin_file_path_1 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_1 = torch.load(bin_file_path_1, map_location='cpu')

bin_file_path_2 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_2 = torch.load(bin_file_path_2, map_location='cpu')

state_dict_all = torch.load(os.path.join(cache_path, load_dict_path), map_location='cpu')
state_dict=state_dict_all['state']

# 更新 binary_data_1 和 binary_data_2
for key, value in state_dict.items():
    if key in binary_data_1:
        binary_data_1[key] = value
    if key in binary_data_2:
        binary_data_2[key] = value

# 保存更新后的 binary_data_1 和 binary_data_2 回 bin 文件
torch.save(binary_data_1, bin_file_path_1)
torch.save(binary_data_2, bin_file_path_2)
    
