import os
from typing import Optional
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

# from torch.utils.data.distributed import DistributedSampler
#
# from transformers import Trainer
# from transformers import LlamaTokenizer
# from transformers.trainer_pt_utils import nested_numpify, nested_concat
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
#
# from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
# from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.training_args import TrainingArguments
#
# from transformers.trainer_callback import TrainerCallback
# from transformers.trainer_utils import EvalPrediction

from datasets import load_dataset
from dataclasses import dataclass, field


@dataclass
class DataCollatorForCausalLM:

    def __call__(self, features, return_tensors=None):

        input_ids, attention_mask = [], []
        max_length = [0, 0, ]
        for feature in features:
            input_ids.append(feature['input_ids'])
            max_length[0] = max(max_length[0], len(feature['input_ids']))
            attention_mask.append(feature['attention_mask'])
            max_length[1] = max(max_length[1], len(feature['attention_mask']))

        for i in range(2):
            each = (input_ids, attention_mask)[i]
            for item in each:
                item.extend([0] * (max_length[i] - len(item)))

        batched_features = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'labels': torch.tensor(input_ids).long(),
        }

        return batched_features


class MyDataset(Dataset):
    def __init__(self, dataset):
        Dataset.__init__(self)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return {
            'input_ids': self.dataset[item]['input_ids'],
            'attention_mask': self.dataset[item]['attention_mask']
        }


class MyDataloader:
    def __init__(self, max_length, tokenizer, dataset_info, split, test_note=''):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_info = dataset_info
        self.split = split
        self.finetune_cache = '/remote-home/xrliu/projects/FEPE-deepspeed/caches/{}-llama-{}-{}.pkl'. \
            format(dataset_info.dataset_name.replace('-', '_'), dataset_info.train_split, max_length)
        self.evaluate_cache = '/remote-home/xrliu/projects/FEPE-deepspeed/caches/{}-llama-{}-{}.pkl'. \
            format(dataset_info.dataset_name.replace('-', '_'), dataset_info.test_split, test_note)

        print(self.finetune_cache, self.evaluate_cache)

        if split == dataset_info.train_split:
            self.data = self.process_for_finetune()
        else:
            self.data = self.process_for_evaluate()

    def process_for_finetune(self):

        if os.path.exists(self.finetune_cache):
            return torch.load(self.finetune_cache)

        dataset = load_dataset(self.dataset_info.path, name=self.dataset_info.name, split=self.split)

        def tokenize_function(examples):

            return self.tokenizer(examples[self.dataset_info.input_key], max_length=self.max_length, truncation=True)

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=self.dataset_info.remove_columns)
        dataset = MyDataset(deepcopy(dataset))

        torch.save(dataset, self.finetune_cache)
        return dataset

    def process_for_evaluate(self):

        if os.path.exists(self.evaluate_cache):
            return torch.load(self.evaluate_cache)

        dataset = load_dataset(self.dataset_info.path, name=self.dataset_info.name, split=self.split)

        print(self.dataset_info.extrapolate_lengths[0])

        def tokenize_function(examples):
            return self.tokenizer(examples[self.dataset_info.input_key],
                                  max_length=self.dataset_info.extrapolate_lengths[0], truncation=True)

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=self.dataset_info.remove_columns)

        dataset = dataset.filter(lambda x: len(x['input_ids']) >= self.dataset_info.extrapolate_lengths[0])
        print(len(dataset))

        datasets = {}

        for length in self.dataset_info.extrapolate_lengths:

            print(length, len(dataset))
            dataset = dataset.map(lambda instance: {'input_ids': instance['input_ids'][:length],
                                                    'attention_mask': instance['attention_mask'][:length]})
            datasets[str(length)] = MyDataset(deepcopy(dataset))

        torch.save(datasets, self.evaluate_cache)
        return datasets


@dataclass
class DatasetInfo:
    path: str = None
    dataset_name: str = None
    train_split: str = None
    valid_split: str = None
    test_split: str = None
    name: str = None
    input_key: str = None
    remove_columns: list = None
    extrapolate_lengths: list = None


def get_dataset_info(dataset_name):
    if dataset_name == 'arxiv':
        return DatasetInfo(
            dataset_name='arxiv',
            path='ccdv/arxiv-summarization',
            name='document',
            train_split='train',
            test_split='test',
            input_key='article',
            remove_columns=['article', 'abstract'],
            extrapolate_lengths=[7168, 6656, 6144, 5632, 5120, 4608, 4096, 3584,
                                 3072, 2560, 2048, 1536, 1024, 768, 512, 256, 128, ]
        )
    elif dataset_name == 'pg19':
        return DatasetInfo(
            dataset_name='pg19',
            path='pg19',
            train_split='train',
            test_split='test',
            input_key='text',
            remove_columns=['short_book_title', 'publication_date', 'url', 'text'],
            extrapolate_lengths=[10240, 9216, 8192, 7168, 6144, 5120,
                                 4096, 3072, 2048, 1024, 512, 256, ]
        )
    else:
        raise NotImplementedError


if __name__ == '__main__':

    from transformers import AutoTokenizer

    max_length = 8192

    tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
    tokenizer.pad_token_id = 0

    dataset_info = get_dataset_info('arxiv')
    dataset_info.extrapolate_lengths = [10240, 9216, 8192, 7168, 6144, 5120, 4096, 3072, 2048, 1024, 512, 128]

    test = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data
    # train = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data

    # max_length = 2048
    #
    # tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B')
    # tokenizer.pad_token_id = 0
    #
    # dataset_info = get_dataset_info('pg19')
    #
    # test = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.test_split, test_note='extra').data
    # train = MyDataloader(max_length, tokenizer, dataset_info, split=dataset_info.train_split).data
