import os
import sys
import json
import torch

from functools import reduce
from joblib import Parallel, delayed

# Directory containing the JSONL files
directory_path = '/mnt/inspurfs/share_data/llm_data/0811_raw_train_merged_V7/train/en'

path_src_lst = ['/'.join([directory_path, item]) for item in sorted(os.listdir(directory_path)) if item != 'bpe']
path_src_lst = [['/'.join([path, item]) for item in sorted(os.listdir(path)) if item.endswith('.jsonl')] for path in path_src_lst]
path_src_lst = reduce(lambda x, y: x + y, path_src_lst)

proc_num = 36
json_num = len(path_src_lst) // proc_num + 1

# print(len(path_src_lst))

def separate_jsonl(idx):

    # List to store all JSONL data
    documents_str = []
    length = []
    ids = []
    file_paths = []

    num = 0
    sample_num = 0

    for j in range(json_num):
        
        if idx * json_num + j >= len(path_src_lst):
            break
        file_path = path_src_lst[idx * json_num + j]

        with open(file_path, 'r', encoding='utf-8') as f:
            jsonl_data = [json.loads(line) for line in f]
            for i in range(len(jsonl_data)):
                input_ids = jsonl_data[i]['input_ids']
                if len(input_ids) > 4096:
                    length.append(len(input_ids))
                    ids.append(jsonl_data[i]['id'])
                    file_paths.append(file_path)
                    documents_str.append(" ".join([str(x) for x in input_ids]))
                    sample_num += 1
                elif num % 10 == 0:
                    length.append(len(input_ids))
                    ids.append(jsonl_data[i]['id'])
                    file_paths.append(file_path)
                    documents_str.append(" ".join([str(x) for x in input_ids]))
                    sample_num += 1
                num += 1
        if idx == 0: 
            print(f"total lines in {file_path}: {num}, sampled lines: {sample_num}")
            sys.stdout.flush()
        num = 0
        sample_num = 0
        
    if idx == 0: 
        print(f"total lines: {len(documents_str)}")
        sys.stdout.flush()

    # save to disk
    torch.save(documents_str, f'/mnt/inspurfs/share_data/llm_data/lvkai/0811_raw_train_merged_V7/cluster_data/documents_str_{idx}.pkl')
    torch.save(length, f'/mnt/inspurfs/share_data/llm_data/lvkai/0811_raw_train_merged_V7/cluster_data/length_{idx}.pkl')
    torch.save(ids, f'/mnt/inspurfs/share_data/llm_data/lvkai/0811_raw_train_merged_V7/cluster_data/ids_{idx}.pkl')
    torch.save(file_paths, f'/mnt/inspurfs/share_data/llm_data/lvkai/0811_raw_train_merged_V7/cluster_data/file_paths_{idx}.pkl')


# srun -p llm_o --cpus-per-task=36 python 231121_0811_downsampling.py

Parallel(n_jobs=proc_num)(delayed(separate_jsonl)(iter) for iter in range(proc_num))
