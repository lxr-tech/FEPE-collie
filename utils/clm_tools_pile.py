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
        datadict['file'] = datadict['file'].replace('hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/', 
                                                    'p_ssd:s3://P_model_weights/liuxiaoran/backup_trainig_data/train/en/pile/')
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


def get_gen_book_for_evaluate(gen_path, gen_lengths):

    if os.path.exists(gen_path):
        return torch.load(gen_path)
    
    path = 'p_ssd:s3://P_model_weights/liuxiaoran/backup_trainig_data/valid/en/pile_Books3/val.bin'
    # path = 'hdd:s3://opennlplab_hdd/backup_trainig_data/valid/en/pile_Books3/val.bin'
    assert PetrelIODriver.exists(path + '.meta')
    meta = np.load(PetrelIODriver.load_buffer(path + '.meta'))
    data = StringIO(PetrelIODriver.load(path, mode='r'))
    
    indices = []    
    if os.path.exists(gen_path + '.meta'):
        indices = torch.load(gen_path + '.meta')
    else:
        for sample in meta:
            if sample[1] >= max(gen_lengths):
                indices.append({'offset': sample[0]})
        torch.save(indices, gen_path + '.meta')

    dataset = []
    for datadict in indices:
        data.seek(datadict['offset'])
        sample = json.loads(data.readline())
        dataset.append({'input_ids': torch.tensor(sample['tokens'][:max(gen_lengths)]).long(), })
    dataset = Dataset.from_list(dataset)
    
    if env.local_rank == 0:
        print("evaluate data num =", len(dataset))

    test_datasets = {}
    for length in sorted(gen_lengths, reverse=True):
        print(length)
        dataset = dataset.map(lambda instance: {'input_ids': instance['input_ids'][:length-256],
                                                'target': instance['input_ids'][length-256:length]})
        test_datasets[str(length)] = deepcopy(dataset)
        
    for length in gen_lengths:
        print(length, len(test_datasets[str(length)][0]['input_ids']), 
                      len(test_datasets[str(length)][0]['target']), )

    torch.save(test_datasets, gen_path)
    return test_datasets


def get_extra_book_for_evaluate(test_path, test_length, test_num):

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
            indices.append({'offset': sample[0]})
        torch.save(indices, test_path + '.meta')

    dataset, cache = [], {'tokens': [], }
    for datadict in indices:
        if len(dataset) >= test_num:
            break
        data.seek(datadict['offset'])
        sample = json.loads(data.readline())
        cache['tokens'].extend(sample['tokens'])
        if len(cache['tokens']) >= test_length:
            dataset.append({'input_ids': torch.tensor(cache['tokens'][:test_length]).long(), 
                            'labels': torch.tensor(cache['tokens'][:test_length]).long(), })
            cache = {'tokens': [], }
        print(len(dataset), len(cache['tokens']))
    dataset = Dataset.from_list(dataset)
    
    if env.local_rank == 0:
        print("evaluate data num =", len(dataset))

    torch.save(dataset, test_path)
    return dataset


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
    
    # import sentencepiece as spm
    # tokenizer = spm.SentencePieceProcessor()
    # tokenizer.load('/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model')
    
    # dataset = PileDataset(train_path='/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-train-3B-2048.pkl', 
    #                       train_length=2048, num_data=3)
    # sample = dataset[0]
    # print(sample['cu_seqlens'])
    # text = tokenizer.decode_ids(sample['input_ids'][sample['cu_seqlens'][1]:sample['cu_seqlens'][2]].tolist())
    # print(text)
    # sample = dataset[0]
    # print(sample['cu_seqlens'])
    # text = tokenizer.decode_ids(sample['input_ids'][sample['cu_seqlens'][0]:sample['cu_seqlens'][1]].tolist())
    # print(text)
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-49152.pkl'
    # test_lengths = [49152, 45056, 40960, 36864, 32768, 28672, 24576, 20480, 16384, 12288, 8192, 4096]
    # dataset = get_book_for_evaluate(test_path, test_lengths)
    # print(len(dataset['49152']))
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-65536.pkl'
    # test_lengths = [65536, 49152, 32768, 16384, 4096, ]
    # dataset = get_book_for_evaluate(test_path, test_lengths)
    # print(len(dataset['65536']))
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-102400.pkl'
    # test_lengths = [102400, 81920, 65536, 49152, 32768, 4096]
    # dataset = get_book_for_evaluate(test_path, test_lengths)
    # print(len(dataset['102400']))
    
    # train_length = 16 * 1024
    # file_indices = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-train-llama.pkl')
    # # demo: file_indices[0] = {'file': 'hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/train_111.bin', 'offset': 0}
    # data_indices = []
    # # demo: data_indices[0] = 17
    # cur_file_name, cache = None, None
    
    # for index, datadict in enumerate(file_indices):
    
    #     datadict['file'] = datadict['file'].replace('hdd:s3://opennlplab_hdd/backup_trainig_data/train/en/pile/', 
    #                                                 'p_ssd:s3://P_model_weights/liuxiaoran/backup_trainig_data/train/en/pile/')

    #     if cur_file_name is None or cache is None or cur_file_name != datadict['file']:
    #         cur_file_name, cache = datadict['file'], StringIO(PetrelIODriver.load(datadict['file'], mode='r'))
    #     cache.seek(datadict['offset'])
    #     sample = json.loads(cache.readline())
    #     if len(sample['tokens']) >= train_length:
    #         data_indices.append(index)
        
    #     if index % 1000 == 0:
    #         print(f"{index} / {len(file_indices)}, {len(data_indices)}")
            
    # print(len(data_indices))
    # torch.save(data_indices, f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/pile-train-llama-{train_length}.pkl')
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-131072.pkl'
    # test_lengths = [131072, 110592, 4096]
    # dataset = get_book_for_evaluate(test_path, test_lengths)
    # print(len(dataset['131072']))
    
    # test_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-1M-extra.pkl'
    # test_length = 1024 * 1024
    # dataset = get_extra_book_for_evaluate(test_path, test_length, test_num=32)
    # print(len(dataset))
    
    # dataset = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-1M-extra.pkl')
    # max_length = 262144
    
    # test_datasets = {}
    # dataset = dataset.map(lambda instance: {'input_ids': instance['input_ids'][:max_length],
    #                                         'labels': instance['input_ids'][:max_length]})
    # test_datasets[str(max_length)] = deepcopy(dataset)

    # torch.save(test_datasets, '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-262144.pkl')

    gen_lengths = [102400 + 256, 81920 + 256, 65536 + 256, 49152 + 256, 
                   32768 + 256, 16384 + 256, 4096 + 256, 2048 + 256]
    gen_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-gen-llama-102656.pkl'
    dataset = get_gen_book_for_evaluate(gen_path, gen_lengths)
    print(len(dataset['102656']))

"""
0 242632
0 419241
0 548339
0 640280
0 959974
1 0
1 184264
1 329261
1 427924
1 494648
1 608334
1 808495
1 955748
2 0
2 199933
2 306372
2 493261
2 858956
2 982788
3 0
3 160500
3 352511
3 424530
3 622196
3 677149
3 959021
3 1026090
4 0
4 62562
4 159905
4 233720
4 507482
4 571005
4 625215
4 706457
4 706459
4 930296
5 0
5 88828
5 211720
5 353507
5 430564
5 692842
5 844818
5 934371
5 1038409
6 0
6 252483
6 504966
6 559054
6 721331
6 869465
6 1041431
7 0
7 341012
7 570600
7 679329
7 808838
7 991980
8 0
8 161742
8 232666
8 384565
8 454281
8 615582
8 709260
8 851090
9 0
9 292265
9 330206
9 465938
9 589073
9 777577
9 790596
9 863232
10 0
10 319386
10 438134
10 594772
10 708952
10 727209
10 790707
10 840431
11 0
11 84661
11 184672
11 192830
11 281752
11 424165
11 980783
11 1034414
12 0
12 104185
12 661963
12 743925
12 888004
13 0
13 158268
13 316536
13 409950
13 555953
13 671243
13 815322
13 946030
14 0
14 143250
14 293727
14 405922
14 554038
14 824895
14 868189
15 0
15 225532
15 406035
15 486354
15 488832
15 666346
15 710715
15 976001
16 0
16 181720
16 599459
16 744920
16 865583
16 960333
17 0
17 180030
17 209637
17 420966
17 647866
17 762496
17 890796
17 1011261
18 0
18 83339
18 793358
18 816762
18 930286
18 976281
19 0
19 139300
19 489081
19 511410
19 689952
19 858220
19 1027494
20 0
20 88502
20 110924
20 270725
20 326069
20 798472
20 1040713
21 0
21 27356
21 145925
21 350462
21 880921
21 1026118
21 1026120
21 1045634
22 0
22 73990
22 253879
22 413496
22 463532
22 600871
22 642634
22 950246
23 0
23 161522
23 187077
23 260542
23 477516
23 661229
23 840473
23 840475
23 999824
24 0
24 96677
24 232759
24 345058
24 434985
24 750188
24 906981
25 0
25 104660
25 234834
25 308258
25 420166
25 690257
26 0
26 298263
26 674037
26 825564
26 1007220
27 0
27 196810
27 406164
27 542755
27 757693
27 898442
27 1026995
28 0
28 273011
28 400234
28 486735
28 613681
28 643261
28 774658
28 805478
28 984265
29 0
29 283142
29 467986
29 762086
29 937142
30 0
30 290876
30 413636
30 598970
30 914503
30 1043170
31 0
31 125035
31 566860
31 695322
31 801001
32 0
"""
