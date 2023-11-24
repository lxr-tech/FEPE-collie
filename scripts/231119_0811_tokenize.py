import os
import sys
import json

import torch
import sentencepiece as spm

from io import StringIO
from functools import reduce

from joblib import Parallel, delayed

from datetime import datetime

enc = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/tokenizer.model'

enc_sp = spm.SentencePieceProcessor()
enc_sp.load(enc)

root_src = '/mnt/petrelfs/share_data/llm_data/0811_raw_train_merged/train/en'
root_dst = '/mnt/inspurfs/share_data/llm_data/0811_raw_train_merged_V7/train/en'
meta_dst = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/0811_raw_train_merged_meta'

# step 1 : 按照root_src的目录格式，在root_dst下新建对应文件夹路径（不同路径不同domain）
# step 2 : 启不同进程tokenize负责不同路径的jsonl，保留全部其他信息，丢到对应路径下；同时记录不同data_url/domain的总共数量，长度超过4k/16k的数量
# step 3 : 合并不同进程统计的data_url/domain的全部/4k/16k的数据条数，并且统计不同数据条数对应data_url数量，确定other分布

# # step 1

# path_lst = ['/'.join([root_dst, item]) for item in sorted(os.listdir(root_src)) if item != 'bpe']
# for path_dst in path_lst:
#     os.makedirs(path_dst, exist_ok=True)

# # step 2

# path_src_lst = ['/'.join([root_src, item]) for item in sorted(os.listdir(root_src)) if item != 'bpe']
# path_src_lst = [['/'.join([path, item]) for item in sorted(os.listdir(path)) if item.endswith('.jsonl')] for path in path_src_lst]
# path_src_lst = reduce(lambda x, y: x + y, path_src_lst)
# path_dst_lst = [path.replace(root_src, root_dst) for path in path_src_lst]

# # for idx, path_dst in enumerate(path_dst_lst):
# #     print(idx, path_dst)

# # proc_num = 36
# # json_num = len(path_src_lst) // proc_num  # 19

# path_src_lst = path_src_lst[684: ]
# path_dst_lst = path_dst_lst[684: ]

# def separate_jsonl(i):

#     num, total = 0, 2600000  #  * json_num
#     url_dct = {}
#     index_4k, index_16k, index_32k = [], [], []
    
#     # for j in range(json_num):
        
#     # if i * json_num + j >= len(path_src_lst):
#     #     break

#     path_dst = path_dst_lst[i]  #  * json_num + j
#     path_src = path_src_lst[i]  #  * json_num + j
#     file_dst = open(path_dst, mode='w+')
#     file_src = open(path_src, mode='r')
#     json_str = []

#     while True:
#         try:
#             sample = json.loads(file_src.readline())
#         except:
#             file_dst.write(''.join(json_str))
#             json_str = []
#             break
        
#         if num % 10 == 0:
#             input_ids = enc_sp.encode(sample['content'])
#             sample['input_ids'] = input_ids
#             del sample['content']
#             data_url = sample['data_url']
#             data_url = data_url if not data_url.startswith('http') else data_url.split('/')[2]
#             if data_url not in url_dct:
#                 url_dct[data_url] = {'1': 1, '4k': 0, '16k': 0, '32k': 0}
#             else:
#                 url_dct[data_url]['1'] += 1
            
#             if len(input_ids) >= 4096:
#                 file_dst.write(''.join(json_str))
#                 offset = file_dst.tell()
#                 json_str = []
#                 index_4k.append({'file': path_dst, 'offset': offset, })
#                 url_dct[data_url]['4k'] += 1
#             if len(input_ids) >= 16384:
#                 index_16k.append({'file': path_dst, 'offset': offset, })
#                 url_dct[data_url]['16k'] += 1
#             if len(input_ids) >= 32768:
#                 index_32k.append({'file': path_dst, 'offset': offset, })
#                 url_dct[data_url]['32k'] += 1
#             json_str.append(json.dumps(sample) + '\n')
            
#         if i == 0 and num % 10000 == 0:
#             print('[{}]\t {} / {}\t num(url) = {}\t num(4k) = {}\t num(16k) = {}\t num(32k) = {}'.format(
#                 datetime.now().time(), num, total,
#                 len(url_dct), len(index_4k), len(index_16k), len(index_32k)))
#             sys.stdout.flush()
#         num += 1

#     torch.save(url_dct, '{}/url_dct-{}.pkl'.format(meta_dst, 36 + i))
#     torch.save(index_4k, '{}/index_4k-{}.pkl'.format(meta_dst, 36 + i))
#     torch.save(index_16k, '{}/index_16k-{}.pkl'.format(meta_dst, 36 + i))
#     torch.save(index_32k, '{}/index_32k-{}.pkl'.format(meta_dst, 36 + i))
    
#     if i == 0:
#         print('total = {}\t num(url) = {}\t num(4k) = {}\t num(16k) = {}\t num(32k) = {}'.format(
#             num, len(url_dct), len(index_4k), len(index_16k), len(index_32k)))

# # srun -p llm_o --cpus-per-task=35 python 231119_0811_tokenize.py
# # total = 49471497         num(url) = 1895995      num(4k) = 52080         num(16k) = 11   num(32k) = 0

# Parallel(n_jobs=len(path_src_lst))(delayed(separate_jsonl)(iter) for iter in range(len(path_src_lst)))

# # Parallel(n_jobs=proc_num)(delayed(separate_jsonl)(iter) for iter in range(proc_num))

# step 3

proc_num = 70
url_dct, url_dct_inv = {}, {}
index_4k, index_16k, index_32k = [], [], []

for i in range(proc_num):
    url_dct_tmp = torch.load('{}/url_dct-{}.pkl'.format(meta_dst, i))
    index_4k_tmp = torch.load('{}/index_4k-{}.pkl'.format(meta_dst, i))
    index_16k_tmp = torch.load('{}/index_16k-{}.pkl'.format(meta_dst, i))
    index_32k_tmp = torch.load('{}/index_32k-{}.pkl'.format(meta_dst, i))
    
    index_4k.extend(index_4k_tmp)
    index_16k.extend(index_16k_tmp)
    index_32k.extend(index_32k_tmp)
    
    for data_url in url_dct_tmp:
        if data_url not in url_dct:
            url_dct[data_url] = url_dct_tmp[data_url]
        else:
            url_dct[data_url]['1'] += url_dct_tmp[data_url]['1']
            url_dct[data_url]['4k'] += url_dct_tmp[data_url]['4k']
            url_dct[data_url]['16k'] += url_dct_tmp[data_url]['16k']
            url_dct[data_url]['32k'] += url_dct_tmp[data_url]['32k']

    print('[{}]\t {} / {}\t num(url) = {}\t num(4k) = {}\t num(16k) = {}\t num(32k) = {}'.format(
        datetime.now().time(), i, proc_num,
        len(url_dct), len(index_4k), len(index_16k), len(index_32k)))
    sys.stdout.flush()

torch.save(url_dct, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/meta/url_dct.pkl')
torch.save(index_4k, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/meta/index_4k.pkl')
torch.save(index_16k, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/meta/index_16k.pkl')
torch.save(index_32k, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/meta/index_32k.pkl')

for data_url in url_dct:
    url_num = url_dct[data_url]['1']
    url_dct_inv[url_num] = 1 if url_num not in url_dct_inv else url_dct_inv[url_num] + 1

url_dct_inv = dict(sorted(url_dct_inv.items(), key=lambda x: x[0], reverse=True))
torch.save(url_dct_inv, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/meta/url_dct_inv.pkl')
print(url_dct_inv)
