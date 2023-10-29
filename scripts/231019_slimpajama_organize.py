import os
import sys
import json

import torch

from io import StringIO
from functools import reduce

from joblib import Parallel, delayed

from datetime import datetime

# https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama

root_src = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/slimpajama-train/'
root_dst = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/'

# domains = ['RedPajamaArXiv', ]
domains = ['RedPajamaBook', 'RedPajamaC4', 'RedPajamaCommonCrawl', 
           'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaWikipedia', ]

dp_size = 32

paths_4k_src = {domain: [root_src + '{}-llama-4096-{}.pkl'.format(domain, i) for i in range(dp_size)] for domain in domains}
paths_16k_src = {domain: [root_src + '{}-llama-16384-{}.pkl'.format(domain, i) for i in range(dp_size)] for domain in domains}

for domain in domains:
    
    paths_4k = paths_4k_src[domain]
    paths_16k = paths_16k_src[domain]

    index_4k, index_16k = [], []
    
    for i in range(dp_size):
        index_4k.extend(torch.load(paths_4k[i]))
        index_16k.extend(torch.load(paths_16k[i]))

    torch.save(index_4k, root_dst + 'slimpajama-train-{}-llama-4096.pkl'.format(domain))
    torch.save(index_16k, root_dst + 'slimpajama-train-{}-llama-16384.pkl'.format(domain))
    
    print('[total]\t {}\t num(4k) = {}\t num(16k) = {}'.format(
            domain, len(index_4k), len(index_16k)))
    
    # index_4k = sorted(index_4k, key=lambda x: x['file'])

"""
[total]  RedPajamaCommonCrawl    num(4k) = 15639953     num(16k) = 1835000
[total]  RedPajamaC4             num(4k) = 3231482      num(16k) = 125491
[total]  RedPajamaGithub         num(4k) = 1620590      num(16k) = 226782
[total]  RedPajamaBook           num(4k) = 197383       num(16k) = 190803
[total]  RedPajamaArXiv          num(4k) = 1457117      num(16k) = 663730
[total]  RedPajamaWikipedia      num(4k) = 810686       num(16k) = 52156
[total]  RedPajamaStackExchange  num(4k) = 300710       num(16k) = 5280
"""
"""
Commoncrawl     67.0%   87818
C4              15.0%   19661
GitHub          4.5%    5898
Books           4.5%    5898
ArXiv           2.5%    3277
Wikipedia       4.5%    5898
StackExchange   2.0%    2621
"""