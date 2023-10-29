import os
import sys
import json

from io import StringIO
from functools import reduce

from joblib import Parallel, delayed

from datetime import datetime

# 从 /mnt/petrelfs/share_data/llm_data/1006_raw_slimpajama/jsonl 加载
# 写入 /mnt/petrelfs/share_data/llm_data/1006_subdomain_slimpajama/en/xxx
# 分不同domain，保存jsonl格式，上限10G，从一个文件中读，a+方式写入另一个文件，不要内存记录

root_src = '/mnt/petrelfs/share_data/llm_data/1006_raw_slimpajama/jsonl/train/en'
root_dst = '/mnt/petrelfs/share_data/llm_data/1006_subdomain_slimpajama/train/en'

path_lst = ['/'.join([root_src, item]) for item in sorted(os.listdir(root_src))]
path_lst = [['/'.join([path, item]) for item in sorted(os.listdir(path))] for path in path_lst]
path_lst = reduce(lambda x, y: x + y, path_lst)

# print(path_lst[:10])
# print(len(path_lst))  # 59166

# path = path_lst[0]
# file = StringIO(open(path, 'r').read())
# data = json.loads(file.readline())
# print(type(data), list(data))  # ['text', 'meta']
# print(data)

meta, file = {}, {}
# path_src = path_lst[0]

dp_size = 32

domains = ['RedPajamaArXiv', 'RedPajamaBook', 'RedPajamaC4', 'RedPajamaCommonCrawl', 
           'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaWikipedia', ]
for domain in domains:
    meta[domain] = 0
    os.mkdir('/'.join([root_dst, domain]))
    path_dst = '/'.join([root_dst, domain])
    file[domain] = [open('/'.join([path_dst, 'example_train_{}.jsonl'.format(i)]), 'a+') 
                    for i in range(dp_size)]

def separate_jsonl(i, path_src):
    print('[{}]\t {} / {}\t size = {}'.format(datetime.now().time(), 
                                              i, len(path_lst), os.path.getsize(path_src)))
    sys.stdout.flush()
    json_lst = [json.loads(data) for data in open(path_src, 'r').readlines()]
    json_str = {domain: [] for domain in domains}
    
    for data in json_lst:
        meta[data['meta']['redpajama_set_name']] += 1
        json_str[data['meta']['redpajama_set_name']].append(
            json.dumps({'text': data['text']}, ensure_ascii=False) + '\n')
        
    for domain in domains:
        file[domain][i % dp_size].write(''.join(json_str[domain]))
        
Parallel(n_jobs=dp_size, require='sharedmem')(delayed(separate_jsonl)(iter, item) for iter, item in enumerate(path_lst))

# separate_jsonl(0, path_lst[0])
# separate_jsonl(1, path_lst[1])

print(meta)

# srun -p llm_t --cpus-per-task=64 python scripts/231016_slimpajama.py 
