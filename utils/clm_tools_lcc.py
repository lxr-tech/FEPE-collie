import os
from copy import deepcopy

import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def get_code_for_perplexity(train_length, lang, test_lengths, train_path, test_path, tokenizer):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    assert lang in ['python', 'csharp', 'java']

    train_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(train_path)
    train_dataset = get_code_for_pretrain(tokenizer=tokenizer, lang=lang, 
                                          train_path=train_path, train_length=train_length)
    
    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(test_path)
    test_datasets = get_code_for_evaluate(tokenizer=tokenizer, lang=lang, 
                                          test_path=test_path, test_lengths=test_lengths)

    return tokenizer, train_dataset, test_datasets


def get_code_for_pretrain(tokenizer, lang, train_path, train_length):

    if os.path.exists(train_path):
        return torch.load(train_path)
 
    train_dataset = load_dataset(f'microsoft/LCC_{lang}', split='train')

    def tokenize_function(examples):
        return tokenizer(examples['context'], max_length=train_length, truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['context', 'gt'])
    train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) >= train_length)
    train_dataset = train_dataset.map(lambda instance: {'input_ids': instance['input_ids'][:train_length],
                                                        'attention_mask': instance['attention_mask'][:train_length],
                                                        'labels': instance['input_ids'][:train_length]})

    print('num_data', len(train_dataset), 'len_data', len(train_dataset[0]['input_ids']))

    torch.save(train_dataset, train_path)
    return train_dataset


def get_code_for_evaluate(tokenizer, lang, test_path, test_lengths):

    if os.path.exists(test_path):
        return torch.load(test_path)

    test_dataset = load_dataset(f'microsoft/LCC_{lang}', split='validation')

    def tokenize_function(examples):
        return tokenizer(examples['context'], max_length=max(test_lengths), truncation=True)

    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['context', 'gt'])
    test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) >= max(test_lengths))
    
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
    train_length = 4096
    test_lengths = [102400, 81920, 65536, 49152, 32768, 4096]
    lang = 'csharp'  # ['python', 'csharp', 'java']
    train_path = 'lcc-{}-train-{}-{}.pkl'.format(lang, 'llama', train_length)
    test_path = 'lcc-{}-valid-{}-{}.pkl'.format(lang, 'llama', max(test_lengths))

    tokenizer, train_dataset, test_datasets = get_code_for_perplexity(
        tokenizer='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', lang=lang, 
        train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)

