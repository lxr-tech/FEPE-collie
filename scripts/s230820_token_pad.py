from rich.traceback import install

install(show_locals=True)

from datasets import Dataset
from transformers import AutoTokenizer

import sentencepiece as spm

from utils.clm_tools_pile import PileDataset

tokenizer = 'openlm-research/open_llama_7b'

train_length = 2048
test_lengths = [1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, ]

train_path = 'pile-train-{}-{}.pkl'.format('3B', 2048)
test_path = 'pile-test-{}-{}-books3.pkl'.format('3B', test_lengths[-1])

num_data = -1  # int((4 * 1024 * 1024 * 1024) / train_length)  # 2097152

# tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

train_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(train_path)
train_dataset = PileDataset(train_length=train_length, train_path=train_path, num_data=num_data)

for _ in range(10):
    item = train_dataset[_] 
    labels = item['labels']
    seqlen = item['seqlen']
    idx = 0
    for __ in range(seqlen.shape[0] - 1):
        idx += seqlen[__]
        print(_, __, idx, labels[idx - 1], labels[idx])

print(len(train_dataset))
num_token = 0

for _ in range(len(train_dataset)):
    num_token += len(item['input_ids'])
    
# num_data = 16969889
# num_token = 34754332672 ± 2048 ( 32B )
 
print(num_token)
print(num_token / 1024 / 1024 / 1024)