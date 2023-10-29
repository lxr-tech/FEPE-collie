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


def get_pajama_for_perplexity(train_length, test_lengths, train_path, test_path, tokenizer, num_data):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
    train_dataset = PajamaDataset(train_length=train_length, num_data=num_data)
    
    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(test_path)
    test_datasets = get_book_for_evaluate(test_path=test_path, test_lengths=test_lengths)
        
    return tokenizer, train_dataset, test_datasets


class PajamaDataset(torch.utils.data.Dataset):
    
    def __init__(self, train_length=4096, num_data=-1):
        
        assert train_length in [4096, 16384]
        self.train_length = train_length
        
        self.domain_meta1 = {'RedPajamaCommonCrawl': 0.67, 'RedPajamaC4': 0.15, 
                             'RedPajamaGithub': 0.045, 'RedPajamaBook': 0.045, 
                             'RedPajamaWikipedia': 0.045, 'RedPajamaArXiv': 0.025, 
                             'RedPajamaStackExchange': 0.02, }
        self.domain_meta2 = {'RedPajamaCommonCrawl': (0, 0.67), 'RedPajamaC4': (0.67, 0.82), 
                             'RedPajamaGithub': (0.82, 0.865), 'RedPajamaBook': (0.865, 0.91), 
                             'RedPajamaWikipedia': (0.91, 0.955), 'RedPajamaArXiv': (0.955, 0.98), 
                             'RedPajamaStackExchange': (0.98, 1), }
        
        root = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches'

        self.domain_index = {domain: torch.load(f'{root}/slimpajama-train-{domain}-llama-{train_length}.pkl') 
                             for domain in self.domain_meta1}
        self.domain_cfile = {domain: None for domain in self.domain_meta1}
        self.domain_cache = {domain: None for domain in self.domain_meta1}
        
        self.domain_pivot = {domain: env.dp_rank for domain in self.domain_meta1}
        
        self.len = sum([len(self.domain_index[domain]) for domain in self.domain_meta1])
        self.len = num_data if num_data >= 0 else self.len + 1 + num_data
        
    def __len__(self):
        return self.len
    
    def __get_domain__(self):
        a = np.random.uniform()
        for domain in self.domain_meta2:
            if self.domain_meta2[domain][0] <= a < self.domain_meta2[domain][1]:
                return domain
    
    def __getitem__(self, _):
        
        domain = self.__get_domain__()
        
        datadict = self.domain_index[domain][self.domain_pivot[domain]]
        self.domain_pivot[domain] = self.domain_pivot[domain] + env.dp_size

        if self.domain_cfile[domain] is None or self.domain_cfile[domain] != datadict['file']:
            self.domain_cfile[domain] = datadict['file']
            self.domain_cache[domain] = open(datadict['file'], mode='r')
        self.domain_cache[domain].seek(datadict['offset'])
        sample = json.loads(self.domain_cache[domain].readline())
        input_ids = sample['input_ids'][:self.train_length]
        
        return {"input_ids": torch.tensor(input_ids).long(), "labels": torch.tensor(input_ids).long(), }  # 
         

def get_book_for_evaluate(test_path, test_lengths):

    if os.path.exists(test_path):
        return torch.load(test_path)
    
    path = 'p_ssd:s3://P_model_weights/liuxiaoran/backup_trainig_data/valid/en/pile_Books3/val.bin'
    # path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/valid/en/pile_Books3/val.bin'
    assert PetrelIODriver.exists(path + '.meta')
    meta = np.load(PetrelIODriver.load_buffer(path + '.meta'))
    data = StringIO(PetrelIODriver.load(path, mode='r'))
    
    indices = []    
    if os.path.exists(test_path + '.meta'):
        indices = torch.load(test_path + '.meta')
    else:
        for sample in meta:
            if sample[1] >= max(test_lengths):
                indices.append({'offset': sample[0]})
        torch.save(indices, test_path + '.meta')

    dataset = []
    for datadict in indices:
        data.seek(datadict['offset'])
        sample = json.loads(data.readline())
        dataset.append({'input_ids': torch.tensor(sample['tokens'][:max(test_lengths)]).long(), 
                        'labels': torch.tensor(sample['tokens'][:max(test_lengths)]).long(), })
    dataset = Dataset.from_list(dataset)
    
    if env.local_rank == 0:
        print("evaluate data num =", len(dataset))

    test_datasets = {}
    for length in sorted(test_lengths, reverse=True):
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
