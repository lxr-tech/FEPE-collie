import os
import json
from io import StringIO

import torch

from transformers import AutoTokenizer

from collie import env
from collie.driver.io import PetrelIODriver


def get_pile_for_perplexity(train_length, test_lengths, tokenizer):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    
    # 如何从pile路径搞定一个on-the-fly的collie-datasset，我也不清楚迭代几个epoch，但知道多少step
    
    train_dataset = PileDataset(train_length=train_length)
    
    # 如何从pile路径找到pg19子数据集，筛选出超过一定长度的，tokenize然后得到不同长度分段

    test_datasets = BookDataset(test_lengths=test_lengths)
    
    return tokenizer, train_dataset, test_datasets


class PileDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_length=2048, path='hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/'):
        
        self.path = path
        self.train_length = train_length
        self.datafiles = sorted(list(filter(lambda x: x.endswith(".bin"), PetrelIODriver.walk(self.path))))
    
    def __iter__(self):
        
        for datafile in self.datafiles:
            dataset = StringIO(PetrelIODriver.load(os.path.join(self.path, datafile)))
            for line in dataset.readlines():
                sample = json.loads(line.replace('\n', ''))
                yield {'input_ids': torch.tensor(sample['tokens'][:self.train_length]), 
                       'labels': torch.tensor(sample['tokens'][:self.train_length]), }

class BookDataset(torch.utils.data.Dataset):
    
    def __init__(self, test_lengths, path='hdd:s3://opennlplab_hdd/backup_trainig_data/valid/en/pile_Books3/val.bin'):
        
        self.path = path
        self.cur_test_idx = 0
        self.test_lengths = test_lengths
    
    def __iter__(self):
        
        if env.local_rank == 0:
            print('evaluate length = ', self.test_lengths[self.cur_test_idx])
        
        dataset = StringIO(PetrelIODriver.load(self.path))
        for line in dataset.readlines():
            sample = json.loads(line.replace('\n', ''))
            if len(sample['tokens']) < self.test_lengths[-1]:
                continue
            else:
                yield {'input_ids': torch.tensor(sample['tokens'][:self.test_lengths[self.cur_test_idx]]), 
                        'labels': torch.tensor(sample['tokens'][:self.test_lengths[self.cur_test_idx]]), }
        
        self.cur_test_idx = (self.cur_test_idx + 1) % len(self.test_lengths)
        
