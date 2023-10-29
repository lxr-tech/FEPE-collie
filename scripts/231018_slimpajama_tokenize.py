import os
import sys
import json

import torch
import sentencepiece as spm

from io import StringIO
from functools import reduce

from joblib import Parallel, delayed

from datetime import datetime

# 从 /mnt/petrelfs/share_data/llm_data/1006_subdomain_slimpajama/train/en/xxx 加载
# llama tokenize 写入 /mnt/petrelfs/share_data/llm_data/1006_slimpajama_llama/train/en/xxx
# 建立长度超过4096的索引；domain已经区分好，保存jsonl格式；先从最小的一个domain开始

enc = '/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model'

enc_sp = spm.SentencePieceProcessor()
enc_sp.load(enc)

root_src = '/mnt/petrelfs/share_data/llm_data/1006_subdomain_slimpajama/train/en'
root_dst = '/mnt/petrelfs/share_data/llm_data/1006_slimpajama_llama/train/en'

# domains = ['RedPajamaArXiv', 'RedPajamaBook', 'RedPajamaC4', 'RedPajamaCommonCrawl', 
#            'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaWikipedia', ]

domains = ['RedPajamaC4', 'RedPajamaCommonCrawl', ]

domain_total = {'RedPajamaArXiv': 1532627, 'RedPajamaBook': 199846, 
                'RedPajamaC4': 324264307, 'RedPajamaCommonCrawl': 186629088, 
                'RedPajamaGithub': 21223703, 'RedPajamaStackExchange': 29626962, 
                'RedPajamaWikipedia': 26918092, }

paths_src = {domain: '/'.join([root_src, domain]) for domain in domains}
paths_dst = {domain: '/'.join([root_dst, domain]) for domain in domains}

index_4k, index_16k = [], []
dp_size = 32

# for domain in domains:
#     os.mkdir('/'.join([root_dst, domain]))

# i, prefix = 0, domains[0]

def separate_jsonl(i, prefix):

    index_4k, index_16k = [], []
    
    path_dst = '/'.join([paths_dst[prefix], 'example_train_{}.jsonl'.format(i)])
    path_src = '/'.join([paths_src[prefix], 'example_train_{}.jsonl'.format(i)])
    file_dst = open(path_dst, mode='a+')
    file_src = open(path_src, mode='r')
    json_str, num, total = [], 0, domain_total[prefix] // dp_size

    while True:
        try:
            sample = json.loads(file_src.readline())
        except:
            file_dst.write(''.join(json_str))
            json_str = []
            break
        num += 1
        input_ids = enc_sp.encode(sample['text'])
        if len(input_ids) >= 4096:
            file_dst.write(''.join(json_str))
            offset = file_dst.tell()
            json_str = []
            index_4k.append({'file': path_dst, 'offset': offset, })
        if len(input_ids) >= 16384:
            index_16k.append({'file': path_dst, 'offset': offset, })
        json_str.append(json.dumps({'input_ids': input_ids}) + '\n')
        if i == 0 and num % 1000 == 0:
            print('[{}]\t {} / {}\t {}\t num(4k) = {}\t num(16k) = {}'.format(
                datetime.now().time(), num, total,
                prefix, len(index_4k), len(index_16k)))
            sys.stdout.flush()

    cache_root = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/slimpajama-train/'
    torch.save(index_4k, cache_root + '{}-llama-4096-{}.pkl'.format(domain, i))
    torch.save(index_16k, cache_root + '{}-llama-16384-{}.pkl'.format(domain, i))
    
    if i == 0:
        print('[total]\t {}\t num(4k) = {}\t num(16k) = {}'.format(
            domain, len(index_4k), len(index_16k)))

# domain = domains[0]

# srun -p llm_t --cpus-per-task=32 python scripts/231018_tokenize.py

for domain in domains:
    
    n_jobs = dp_size
        
    Parallel(n_jobs=n_jobs)(delayed(separate_jsonl)(iter, domain) for iter in range(n_jobs))

"""
[total]  RedPajamaStackExchange  num(4k) = 9108     num(16k) = 138
[total]  RedPajamaC4             num(4k) = 101014   num(16k) = 3963
[total]  RedPajamaCommonCrawl    num(4k) = 490140   num(16k) = 57520
[total]  RedPajamaBook           num(4k) = 6195     num(16k) = 6005                                                                                              
[total]  RedPajamaArXiv          num(4k) = 45757    num(16k) = 20898
[total]  RedPajamaWikipedia      num(4k) = 25629    num(16k) = 1638
[total]  RedPajamaGithub         num(4k) = 50239    num(16k) = 7122
"""
