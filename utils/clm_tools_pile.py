from rich.traceback import install

install(show_locals=True)

import os
import json

from io import StringIO
from copy import deepcopy

import torch
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer

import sentencepiece as spm

from collie import env
from collie.driver.io import PetrelIODriver


def get_pile_for_perplexity(train_length, test_lengths, train_path, test_path, tokenizer, num_data):
    
    tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf', use_fast=False)
    
    # 如何从pile路径搞定一个on-the-fly的collie-datasset，我也不清楚迭代几个epoch，但知道多少step
    
    train_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(train_path)
    train_dataset = PileDataset(train_length=train_length, train_path=train_path, num_data=num_data)
    
    # 如何从pile路径找到books3子数据集，筛选出超过一定长度的，tokenize然后得到不同长度分段

    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(test_path)
    test_datasets = get_book_for_evaluate(test_path=test_path, test_lengths=test_lengths)
        
    return tokenizer, train_dataset, test_datasets


class PileDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_path, train_length=2048, num_data=-1):
        
        self.train_length = train_length 
        
        # self.path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/'
        
        self.cur_file_name, self.cache = None, None
        
        self.file_indices = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-train-llama.pkl')
        self.data_indices = torch.load(train_path)
        self.len = len(self.data_indices)
                    
        self.len = num_data if num_data >= 0 else self.len + 1 + num_data
        
        self.pivot, self.real_len = env.dp_rank, len(self.file_indices)
            
        if env.local_rank == 0:
            print("pretrain data num =", self.len, ", while the real num =", self.real_len)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, _):
        
        datadict = self.file_indices[self.data_indices[self.pivot]]
        self.pivot = (self.pivot + env.dp_size) % self.len

        if self.cur_file_name is None or self.cache is None or self.cur_file_name != datadict['file']:
            self.cur_file_name, self.cache = datadict['file'], StringIO(PetrelIODriver.load(datadict['file'], mode='r'))
        self.cache.seek(datadict['offset'])
        sample = json.loads(self.cache.readline())
        input_ids = sample['tokens'][:self.train_length]
        
        return {"input_ids": torch.tensor(input_ids).long(), "labels": torch.tensor(input_ids).long(), }  # 
            
        # while len(self.token_buffer) < self.train_length:
        #     self.pivot = (self.pivot + env.dp_size) % self.real_len
        #     datadict = self.indices[self.pivot]
        #     if self.cur_file_name is None or self.cache is None or self.cur_file_name != datadict['file']:
        #         self.cur_file_name, self.cache = datadict['file'], StringIO(PetrelIODriver.load(datadict['file'], mode='r'))
        #     self.cache.seek(datadict['offset'])
        #     sample = json.loads(self.cache.readline())    
        #     self.token_buffer.extend(sample['tokens'])
        #     self.index_buffer.extend(range(len(sample['tokens'])))
        #     self.seqlen.append(len(sample['tokens']))
            
        # input_ids = self.token_buffer[:self.train_length]
        # index_ids = self.index_buffer[:self.train_length]
        
        # seqlen = self.seqlen
        # extra_length = len(self.token_buffer) - self.train_length
        # seqlen[-1] -= extra_length

        # self.index_buffer = []
        # self.index_buffer.extend(range(extra_length))
        # self.seqlen = [] if extra_length == 0 else [extra_length]
        # self.token_buffer = self.token_buffer[self.train_length:]

        # assert len(self.token_buffer) == len(self.index_buffer)
         

def get_book_for_evaluate(test_path, test_lengths):

    if os.path.exists(test_path):
        return torch.load(test_path)
    
    path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/valid/en/pile_Books3/val.bin'
    assert PetrelIODriver.exists(path + '.meta')
    meta = np.load(PetrelIODriver.load_buffer(path + '.meta'))
    data = StringIO(PetrelIODriver.load(path, mode='r'))
    
    indices = []    
    if os.path.exists(test_path + '.meta'):
        indices = torch.load(test_path + '.meta')
    else:
        for sample in meta:
            if sample[1] >= test_lengths[-1]:
                indices.append({'offset': sample[0]})
        torch.save(indices, test_path + '.meta')

    dataset = []
    for datadict in indices:
        data.seek(datadict['offset'])
        sample = json.loads(data.readline())
        dataset.append({'input_ids': torch.tensor(sample['tokens'][:test_lengths[0]]).long(), 
                        'labels': torch.tensor(sample['tokens'][:test_lengths[0]]).long(), })
    dataset = Dataset.from_list(dataset)
    
    if env.local_rank == 0:
        print("evaluate data num =", len(dataset))

    test_datasets = {}
    for length in test_lengths:
        print(length)
        dataset = dataset.map(lambda instance: {'input_ids': instance['input_ids'][:length],
                                                'labels': instance['input_ids'][:length]})
        test_datasets[str(length)] = deepcopy(dataset)

    torch.save(test_datasets, test_path)
    return test_datasets


class DummyDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_path, train_length=2048, num_data=-1):
        
        self.len = 1572864 if num_data==-1 else num_data
        self.train_length = train_length 
     
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        return {'input_ids': torch.ones((self.train_length, )).long(), 
                'labels': torch.ones((self.train_length, )).long(), }


if __name__ == "__main__":
    
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model')
    
    dataset = PileDataset(train_path='/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-train-3B-2048.pkl', 
                          train_length=2048, num_data=3)
    sample = dataset[0]
    print(sample['cu_seqlens'])
    text = tokenizer.decode_ids(sample['input_ids'][sample['cu_seqlens'][1]:sample['cu_seqlens'][2]].tolist())
    print(text)
    sample = dataset[0]
    print(sample['cu_seqlens'])
    text = tokenizer.decode_ids(sample['input_ids'][sample['cu_seqlens'][0]:sample['cu_seqlens'][1]].tolist())
    print(text)
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-test-3B-20480-books3.pkl'
    # test_lengths = [20480, 18432, 16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, 1024, ]
    # get_book_for_evaluate(test_path, test_lengths)