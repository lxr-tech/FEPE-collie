import os
import json
import datetime

from io import StringIO
from copy import deepcopy

import torch
import numpy as np

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from collie import env
from collie.driver.io import PetrelIODriver


def get_code_for_perplexity(train_length, test_lengths, train_path, test_path, tokenizer, langs):
    
    assert len(test_lengths) == 1
    test_length = test_lengths[0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    
    init_code_indices(langs=langs, num_words=10 * 1024)

    train_datasets, test_datasets = [], {str(test_length): []}
    for lang in langs:
        load_path = 'star-{}-indices-10k.pkl'.format(lang)
        load_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(load_path)
        
        save_path = 'star-{}-test-{}.pkl'.format(lang, test_length)
        save_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(save_path)
        test_dataset = get_code_with_spec_lang(length=test_length, load_path=load_path, 
                                               save_path=save_path, tokenizer=tokenizer, lang=lang)
        
        save_path = 'star-{}-train-{}.pkl'.format(lang, train_length)
        save_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(save_path)
        train_dataset = get_code_with_spec_lang(length=train_length, load_path=load_path, 
                                                save_path=save_path, tokenizer=tokenizer, lang=lang)
        
        train_datasets.append(train_dataset.remove_columns(['max_stars_count', 'lang']))
        test_datasets[str(test_length)].append(test_dataset[str(test_length)].remove_columns(['max_stars_count', 'lang']))
    
    train_datasets = concatenate_datasets(train_datasets)
    test_datasets[str(test_length)] = concatenate_datasets(test_datasets[str(test_length)])
    return tokenizer, train_datasets, test_datasets


def init_code_indices(langs=['csharp', 'java', 'python', ], num_words=10 * 1024):

    # ['max_stars_repo_path', 'max_stars_repo_name', 'max_stars_count', 'id', 'content']
    root = 'p_ssd:s3://P_model_weights/llm_data/raw-starcoder-0821/train/code/'
    path = {'csharp': 'c-sharp/', 'java': 'java/', 'python': 'python/', } # PetrelIODriver.list(root)

    len_dct = {}

    for lang in langs:
        
        save_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-indices-10k.pkl'.format(lang)
        # print(save_path)
        if os.path.exists(save_path):
            continue
        print('missing !')
        
        files = PetrelIODriver.walk(root + path[lang])
        files = [file for file in files if file.endswith('.jsonl')]
        
        indices = []
        len_dct[lang] = {'num': 0, 'max': 0, 'avg': 0, '10k': 0, }
        
        for i, file in enumerate(files):
            cache = StringIO(PetrelIODriver.load(root + path[lang] + file, mode='r'))
            offset = cache.tell()
            sample = cache.readline()
            while sample != '':
                sample = json.loads(sample)
                length = len(sample['content'].split())
                if length >= num_words:
                    indices.append({'file': root + path[lang] + file, 'offset': offset, 
                                    'max_stars_count': sample['max_stars_count']})
                len_dct[lang]['num'] += 1
                len_dct[lang]['max'] = max(len_dct[lang]['max'], length)
                len_dct[lang]['avg'] *= (1 - 1 / len_dct[lang]['num'])
                len_dct[lang]['avg'] += length / len_dct[lang]['num']
                len_dct[lang]['10k'] += 1 if length > num_words else 0
                
                offset = cache.tell()
                sample = cache.readline()
                
            print('{} {}/{} {}'.format(lang, i, len(files), len_dct[lang]))
            
        torch.save(indices, save_path)
        
    """
    c-sharp/    44/45   {'num': 10801285, 'max': 342734, 'avg': 320.52475163825227, '10k': 14787}
    c/          52/53   {'num': 8536791, 'max': 1882386, 'avg': 653.6271375274099, '10k': 52416}
    html/       28/29   {'num': 3299965, 'max': 210608, 'avg': 728.4861197011846, '10k': 19119}
    java/       86/87   {'num': 20071773, 'max': 169527, 'avg': 361.7356222592061, '10k': 25497}
    javascript/ 64/65   {'num': 19544285, 'max': 600020, 'avg': 298.1298056695687, '10k': 30612}
    python/     58/59   {'num': 12866649, 'max': 300032, 'avg': 400.8048050427917, '10k': 15078}
    """

def get_code_with_spec_lang(length, load_path, save_path, tokenizer, lang):
    
    # print(save_path)
    if os.path.exists(save_path):
        return torch.load(save_path)
    print('missing !')
    
    file = torch.load(load_path)
    print('before filter', len(file))
    file = list(filter(lambda x: x['max_stars_count'] > 0, file))
    print('after filter ', len(file))
    
    cache_list, cache_file, cache_load = [], None, None
    
    for i, data_dict in enumerate(file):
        if cache_file != data_dict['file']:
            cache_file, cache_load = data_dict['file'], StringIO(PetrelIODriver.load(data_dict['file'], mode='r'))
        offset = cache_load.tell()
        sample = cache_load.readline()
        sample = json.loads(sample)
        cache_list.append({'content': sample['content'], 'max_stars_count': sample['max_stars_count']})
        if i % 1000 == 0:
            print('{} {}/{} {}'.format(lang, i, len(file), datetime.datetime.now()))
    
    dataset = Dataset.from_list(cache_list)
        
    def tokenize_function(examples):
        return tokenizer(examples['content'], max_length=length, truncation=True)

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['content'])
    dataset = dataset.filter(lambda x: len(x['input_ids']) >= length)
    
    dataset = dataset.map(lambda instance: {'input_ids': instance['input_ids'][:length],
                                            'attention_mask': instance['attention_mask'][:length],
                                            'labels': instance['input_ids'][:length], 
                                            'max_stars_count': instance['max_stars_count'], 'lang': lang})
    
    print(length, 'num_data', len(dataset), 'len_data', len(dataset[0]['input_ids']))
    torch.save(dataset, save_path)
    
    return dataset

# # tokenizer, train_datasets, test_datasets = get_code_for_perplexity(train_length=4096, test_lengths=[49152], train_path=None, test_path=None, 
# #                         tokenizer='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', 
# #                         langs=['csharp', 'java', 'python', ])
# # print(len(train_datasets), len(test_datasets['49152']))
# # import pdb; pdb.set_trace()

# path = {'csharp': 'c-sharp/', 'java': 'java/', 'python': 'python/', }

# tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', use_fast=False)

# train_length, test1_length, test2_length = 4 * 1024, 48 * 1024, 100 * 1024

# for lang in path:
       
#     load_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-indices-10k.pkl'.format(lang)
#     save_train = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-train.pkl'.format(lang)
#     # save_train = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-train-{}.pkl'.format(lang, train_length)
#     # save_test1 = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-test-{}.pkl'.format(lang, test1_length)
#     # save_test2 = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/star-{}-test-{}.pkl'.format(lang, test2_length)
    
#     file = torch.load(load_path)
#     print(load_path)
#     file = list(filter(lambda x: x['max_stars_count'] > 0, file))
#     print(lang, len(file))
    
#     cache_list, cache_file, cache_load = [], None, None
    
#     for i, data_dict in enumerate(file):
        
#         if cache_file != data_dict['file']:
#             cache_file, cache_load = data_dict['file'], StringIO(PetrelIODriver.load(data_dict['file'], mode='r'))
        
#         offset = cache_load.tell()
#         sample = cache_load.readline()
#         sample = json.loads(sample)
#         cache_list.append({'content': sample['content'], 'max_stars_count': sample['max_stars_count']})
        
#         if i % 1000 == 0:
#             print('{} {}/{} {}'.format(lang, i, len(file), datetime.datetime.now()))
    
#     dataset = Dataset.from_list(cache_list)
        
#     def tokenize_function(examples):
#         return tokenizer(examples['content'], max_length=test2_length, truncation=True)

#     dataset = dataset.map(tokenize_function, batched=True, remove_columns=['content'])
#     # dataset = dataset.filter(lambda x: len(x['input_ids']) >= train_length)
    
#     train_dataset = dataset.filter(lambda x: len(x['input_ids']) < test1_length)
#     train_dataset = train_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:train_length],
#                                                         'attention_mask': instance['attention_mask'][:train_length],
#                                                         'labels': instance['input_ids'][:train_length], 
#                                                         'max_stars_count': instance['max_stars_count'], 'lang': lang})
    
#     print(train_length, 'num_data', len(train_dataset), 'len_data', len(train_dataset[0]['input_ids']))
#     torch.save(train_dataset, save_train)
#     print(save_train)

#     # test1_dataset = dataset.filter(lambda x: test1_length <= len(x['input_ids']) < test2_length)
#     # test1_dataset = test1_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:test1_length],
#     #                                                     'attention_mask': instance['attention_mask'][:test1_length],
#     #                                                     'labels': instance['input_ids'][:test1_length], 
#     #                                                     'max_stars_count': instance['max_stars_count'], 'lang': lang})
    
#     # print(test1_length, 'num_data', len(test1_dataset), 'len_data', len(test1_dataset[0]['input_ids']))
#     # torch.save({str(int(test1_length)): test1_dataset}, save_test1)
#     # print(save_test1)
    
#     # test2_dataset = dataset.filter(lambda x: len(x['input_ids']) >= test2_length)
#     # test2_dataset = test2_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:test2_length],
#     #                                                     'attention_mask': instance['attention_mask'][:test2_length],
#     #                                                     'labels': instance['input_ids'][:test2_length], 
#     #                                                     'max_stars_count': instance['max_stars_count'], 'lang': lang})
    
#     # print(test2_length, 'num_data', len(test2_dataset), 'len_data', len(test2_dataset[0]['input_ids']))
#     # torch.save({str(int(test2_length)): test2_dataset}, save_test2)
#     # print(save_test2)


# """
# csharp_train  4096 num_data 416 len_data 4096
# csharp_test1  49152 num_data 4 len_data 49152
# csharp_test2  102400 num_data 3 len_data 102400
# java_train    4096 num_data 648 len_data 4096
# java_test1    49152 num_data 9 len_data 49152
# java_test2    102400 num_data 7 len_data 102400
# python_train  4096 num_data 625 len_data 4096
# python_test1  49152 num_data 7 len_data 49152
# python_test2  102400 num_data 2 len_data 102400
# """
