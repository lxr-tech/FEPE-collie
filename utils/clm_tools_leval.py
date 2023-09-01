import os
from copy import deepcopy

import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def get_leval_for_perplexity(train_length, subset, test_lengths, train_path, test_path, tokenizer):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    
    assert subset in ["coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", 
                      "financial_qa", "legal_contract_qa", "multidoc_qa", "natural_question", "narrative_qa", "scientific_qa", 
                      "gov_report_summ", "meeting_summ", "news_summ", "paper_assistant", "patent_summ", "review_summ", "tv_show_summ"]
    
    test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/{}'.format(test_path)
    test_datasets = get_leval_for_evaluate(tokenizer=tokenizer, subset=subset, 
                                           test_path=test_path, test_lengths=test_lengths)

    return tokenizer, None, test_datasets


def get_leval_for_evaluate(tokenizer, subset, test_path, test_lengths):

    if os.path.exists(test_path):
        return torch.load(test_path)

    test_dataset = load_dataset("L4NLP/LEval", name=subset, split='test')
    
    def tokenize_function(examples):
        return tokenizer(examples['input'], max_length=max(test_lengths), truncation=True)

    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['input', 'instructions', 'outputs', 
                                                                                     'source', 'evaluation'])
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
    test_lengths = [49152, 40960, 32768, 24576, 20480, 16384, 8192, 4096]
    
    """
    "coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", 
    "financial_qa", "legal_contract_qa", "multidoc_qa", "natural_question", "narrative_qa", "scientific_qa", 
    "gov_report_summ", "meeting_summ", "news_summ", "paper_assistant", "patent_summ", "review_summ", "tv_show_summ"
    """
    
    subset = 'narrative_qa'
    test_path = 'leval-{}-{}-{}.pkl'.format(subset, 'llama', max(test_lengths))

    tokenizer, train_dataset, test_datasets = get_leval_for_perplexity(
        tokenizer='/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', subset=subset, 
        train_length=None, train_path=None, test_lengths=test_lengths, test_path=test_path)

