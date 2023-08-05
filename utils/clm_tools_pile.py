from rich.traceback import install

install(show_locals=True)

import os
import json

from io import StringIO
from copy import deepcopy

import torch
import numpy as np

from transformers import AutoTokenizer
from datasets import Dataset

from collie import env
from collie.driver.io import PetrelIODriver


def get_pile_for_perplexity(train_length, test_lengths, train_path, test_path, tokenizer, num_data):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    
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
        
        self.path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/'
        
        datafiles = sorted(list(filter(lambda x: x.endswith(".bin"), PetrelIODriver.walk(self.path))))

        self.len, self.indices = 0, []
        self.cur_file_name, self.cache = None, None
        
        if os.path.exists(train_path):
            self.indices = torch.load(train_path)
            self.len = len(self.indices)
        else:
            for datafile in datafiles:
                datafile, metafile = self.path + datafile, self.path + datafile + '.meta'
                if PetrelIODriver.exists(metafile):
                    meta = np.load(PetrelIODriver.load_buffer(metafile))
                    self.len += meta.shape[0]
                    self.indices.extend([{'file': datafile, 'offset': sample[0]} for sample in meta])
                else:
                    buffer = StringIO(PetrelIODriver.load(datafile, 'r'))
                    offset = 0
                    self.indices.append({'file': datafile, 'offset': offset})
                    for line in buffer.readlines():
                        offset += len(line)
                        self.indices.append({'file': datafile, 'offset': offset})
                        self.len += 1
                    self.indices.pop(-1)
            torch.save(self.indices, train_path)
            
        self.len = num_data if num_data >= 0 else self.len + 1 + num_data
        self.indices = self.indices[:self.len]
            
        if env.local_rank == 0:
            print("pretrain data num =", self.len)
        self.max_retry = 100
    
    def __len__(self):
        return self.len
    
    # def __iter__(self): 
    #     for datafile in self.datafiles:
    #         dataset = StringIO(PetrelIODriver.load(self.path + datafile, mode='r'))
    #         for line in dataset.readlines():
    #             sample = json.loads(line.replace('\n', ''))
    #             yield {'input_ids': torch.tensor(sample['tokens'][:self.train_length]), 
    #                    'labels': torch.tensor(sample['tokens'][:self.train_length]), }
    
    def __getitem__(self, index):
        
        datadict = self.indices[index]
        if self.cur_file_name is None or self.cache is None or self.cur_file_name != datadict['file']:
            self.cur_file_name, self.cache = datadict['file'], StringIO(PetrelIODriver.load(datadict['file'], mode='r'))
            
        self.cache.seek(datadict['offset'])
        sample = json.loads(self.cache.readline())        
        return {'input_ids': torch.tensor(sample['tokens'][:self.train_length]).long(), 
                'labels': torch.tensor(sample['tokens'][:self.train_length]).long(), }
 

# class BookDataset(torch.utils.data.Dataset):
    
#     def __init__(self, test_path, test_lengths):
        
#         self.cur_test_idx = 0
#         self.test_lengths = test_lengths
        
#         self.path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/valid/en/pile_Books3/val.bin'

#         assert PetrelIODriver.exists(self.path + '.meta')
        
#         meta = np.load(PetrelIODriver.load_buffer(self.path + '.meta'))
#         self.len, self.indices = 0, []
         
#         if os.path.exists(test_path):
#             self.indices = torch.load(test_path)
#             self.len = len(self.indices)
#         else:
#             for sample in meta:
#                 if sample[1] >= test_lengths[-1]:
#                     self.indices.append({'offset': sample[0]})
#                     self.len += 1
#             torch.save(self.indices, test_path)
            
#         self.cache = StringIO(PetrelIODriver.load(self.path, mode='r'))
            
#         if env.local_rank == 0:
#             print("evaluate data num =", self.len)
    
#     def __len__(self):
#         return self.len
        
#     # def __iter__(self):
        
#     #     if env.local_rank == 0:
#     #         print('evaluate length = ', self.test_lengths[self.cur_test_idx])
        
#     #     dataset = StringIO(PetrelIODriver.load(self.path, mode='r'))
#     #     for line in dataset.readlines():
#     #         sample = json.loads(line.replace('\n', ''))
#     #         if len(sample['tokens']) < self.test_lengths[-1]:
#     #             continue
#     #         else:
#     #             yield {'input_ids': torch.tensor(sample['tokens'][:self.test_lengths[self.cur_test_idx]]), 
#     #                     'labels': torch.tensor(sample['tokens'][:self.test_lengths[self.cur_test_idx]]), }
        
#     #     self.cur_test_idx = (self.cur_test_idx + 1) % len(self.test_lengths)
        
#     def __getitem__(self, index):
        
#         cur_test_idx = self.cur_test_idx
        
#         if index == 0 and env.rank == 0:
#             print("cur_test_length :", self.test_lengths[cur_test_idx])
#         if index == len(self) - 1:
#             self.cur_test_idx = (self.cur_test_idx + 1) % len(self.test_lengths)
        
#         datadict = self.indices[index]
        
#         self.cache.seek(datadict['offset'])
#         sample = json.loads(self.cache.readline())
#         return {'input_ids': torch.tensor(sample['tokens'][:self.test_lengths[cur_test_idx]]).long(), 
#                 'labels': torch.tensor(sample['tokens'][:self.test_lengths[cur_test_idx]]).long(), }


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
    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-test-3B-20480-books3.pkl'
    test_lengths = [20480, 18432, 16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, 1024, ]
    get_book_for_evaluate(test_path, test_lengths)