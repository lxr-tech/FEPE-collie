import os
from copy import deepcopy

import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def get_arxiv_for_perplexity(train_length, test_lengths, train_path, test_path, tokenizer):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

    train_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(train_path)
    train_dataset = get_arxiv_for_pretrain(tokenizer=tokenizer, train_path=train_path, train_length=train_length)
    
    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(test_path)
    test_datasets = get_arive_for_evaluate(tokenizer=tokenizer, test_path=test_path, test_lengths=test_lengths)

    return tokenizer, train_dataset, test_datasets


def get_arxiv_for_pretrain(tokenizer, train_path, train_length):

    if os.path.exists(train_path):
        return torch.load(train_path)
 
    train_dataset = load_dataset('ccdv/arxiv-summarization', name='document', split='train', 
                                cache_dir='/mnt/petrelfs/liuxiaoran/.huggingface/ccdv__arxiv-summarization')

    def tokenize_function(examples):

        return tokenizer(examples['article'], max_length=train_length, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['article', 'abstract'])
    train_dataset = train_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:train_length],
                                                        'attention_mask': instance['attention_mask'][:train_length],
                                                        'labels': instance['input_ids'][:train_length]})

    print('num_data', len(train_dataset), 'len_data', len(train_dataset[0]['input_ids']))

    torch.save(train_dataset, train_path)
    return train_dataset


def get_arive_for_evaluate(tokenizer, test_path, test_lengths):

    if os.path.exists(test_path):
        return torch.load(test_path)

    test_dataset = load_dataset('ccdv/arxiv-summarization', name='document', split='test', 
                                cache_dir='/mnt/petrelfs/liuxiaoran/.huggingface/ccdv__arxiv-summarization')

    def tokenize_function(examples):
        return tokenizer(examples['article'], max_length=test_lengths[-1], truncation=True)

    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['article', 'abstract'])
    test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) >= test_lengths[-1])
    
    print('num_data', len(test_dataset), 'len_data', len(test_dataset[0]['input_ids']))

    test_datasets = {}

    for length in test_lengths:

        dataset = test_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:length],
                                                     'attention_mask': instance['attention_mask'][:length],
                                                     'labels': instance['input_ids'][:length]})
        test_datasets[str(length)] = deepcopy(dataset)

    torch.save(test_datasets, test_path)
    return test_datasets


if __name__ == "__main__":
    train_length = 2048
    test_lengths = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, ]
    train_path = 'arxiv-train-{}-{}.pkl'.format('330M', train_length)
    test_path = 'arxiv-test-{}-{}.pkl'.format('330M', test_lengths[-1])

    tokenizer, train_dataset, test_datasets = get_arxiv_for_perplexity(tokenizer='openlm-research/open_llama_7b', 
        train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)

