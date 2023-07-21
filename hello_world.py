import torch

from torch.utils.data import Dataset

# from collie import CollieDatasetForTraining, EvaluatorForPerplexity, PPLMetric

from transformers import AutoTokenizer, AutoConfig
from transformers import TrainingArguments

from configs.clm_train_config import model_args, train_args

from models.llama_with_pe import LlamaForCausalLM
from utils.clm_trainer import TrainerForCausalLM
from utils.clm_tools import DataCollatorForCausalLM

import os

torch.set_default_dtype(torch.bfloat16)  # float32, bfloat16

max_length = 512
model_tag = 'clm_arxiv_1'

key = 'pre_xpos_inv_2d_raw'

assert model_tag in model_args and (model_tag, max_length) in train_args

model_args, train_args = model_args[model_tag], train_args[(model_tag, max_length)]

head_dim = model_args['hidden_size'] // model_args['num_attention_heads']

pe_config = {'1d': key.__contains__('1d'), 'exp': key.__contains__('xpos'),
             'imp': key.__contains__('imp'), 'log': key.__contains__('log'),
             'flash_train': False, 'flash_test': True,
             'post': key.__contains__('post'), 'both': key.__contains__('both'),
             'init': key.__contains__('init'), }

model_path = f'/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/{key}/train_last/pytorch_model.bin'

config = AutoConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.gradient_checkpointing = True
config.torch_dtype = torch.bfloat16
config.hidden_size = model_args['hidden_size']  # 4096
config.intermediate_size = model_args['intermediate_size']  # 11008
config.num_attention_heads = model_args['num_attention_heads']  # 32
config.num_hidden_layers = model_args['num_hidden_layers']  # 32

model = LlamaForCausalLM(config=config, pe_config=pe_config).bfloat16()  # float()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
# model = LlamaForCausalLM.from_pretrained(model_path)

train_args['bf16'] = True
train_args['bf16_full_eval'] = True
train_args['fp16'] = False
train_args['fp16_full_eval'] = False

rank = int(os.environ["LOCAL_RANK"])
size = int(os.environ["WORLD_SIZE"])

train_args['per_device_train_batch_size'] = 1
train_args['per_device_eval_batch_size'] = 1

if rank == 0:
    print('model type is', key, '\n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print('model is over !', 'at epoch 1', '\n')

tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/llama_hf/7B', use_fast=False)
tokenizer.pad_token_id = 0

if rank == 0:
    print('tokenizer is over !', '\n')
    # print(tokenizer("I have a story that I have a story ."))  
    # [1, 306, 505, 263, 5828, 393, 306, 505, 263, 5828, 869]


class UpperDataset(Dataset):
    def __init__(self, length, number):
        Dataset.__init__(self)
        self.length = length
        data = [306, 505, 263, 5828, 393, ] * (length // 5 + 1)
        data.insert(0, 1)
        data[-1] = 869
        self.dataset = [data] * number

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return {
            'input_ids': self.dataset[item][: self.length],
            'attention_mask': [1] * self.length
        }
    
if rank == 0:
    print(UpperDataset(length=10, number=size)[0])

eval_datasets = {}
prefix_list = ['32', '512', '10240', '40960', '51200', '61440', '65536', '71680', '81920']  #
# prefix_list = ['512', '1024', '2048', '3072', '4096', '5120', '6144', '7168', '8192', '9216', '10240', ]  #

"""
'torch init version & bf16': {
    'eval_32_acc': 0.225806 , 'eval_32_ppl': 4608.0 ,
    'eval_512_acc': 0.426614 , 'eval_512_ppl': 237.0 ,
    'eval_10240_acc': 0.070222 , 'eval_10240_ppl': 3168.0 ,
    'eval_40960_acc': 0.017383 , 'eval_40960_ppl': 9792.0 ,
    'eval_51200_acc': 0.013946 , 'eval_51200_ppl': 10432.0 ,
    'eval_61440_acc': 0.011621 , 'eval_61440_ppl': 11072.0 ,
    'eval_65536_acc': 0.01091 , 'eval_65536_ppl': 11072.0 ,
    'eval_71680_acc': 0.0 , 'eval_71680_ppl': nan ,
    'eval_81920_acc': 0.0 , 'eval_81920_ppl': nan ,
}
'torch init version & bf16': {
    'eval_32_acc': 0.129032 , 'eval_32_ppl': 7136.0 ,
    'eval_512_acc': 0.367906 , 'eval_512_ppl': 776.0 ,
    'eval_10240_acc': 0.246118 , 'eval_10240_ppl': 1408.0 ,
    'eval_40960_acc': 0.211675 , 'eval_40960_ppl': 2048.0 ,
    'eval_51200_acc': 0.209379 , 'eval_51200_ppl': 2176.0 ,
    'eval_61440_acc': 0.207783 , 'eval_61440_ppl': 2256.0 ,
    'eval_65536_acc': 0.207309 , 'eval_65536_ppl': 2256.0 ,
    'eval_71680_acc': 0.0 , 'eval_71680_ppl': nan ,
    'eval_81920_acc': 0.0 , 'eval_81920_ppl': nan ,
}
"""

for prefix in prefix_list:
    eval_datasets[prefix] = UpperDataset(length=int(prefix), number=size)
train_dataset = eval_datasets['10240']

if rank == 0:
    print('dataset is over !')

training_args = TrainingArguments(report_to='none', **train_args)

trainer = TrainerForCausalLM(model=model, args=training_args, tokenizer=tokenizer,
                             train_dataset=train_dataset, eval_dataset=eval_datasets,
                             data_collator=DataCollatorForCausalLM(), )

# if rank == 0:
#     import pdb
#     pdb.set_trace()
# torch.distributed.barrier()
#
# if rank == 0:
#     print(f'\'{key}\'\n')
#
if isinstance(trainer.eval_dataset, dict):
    metrics = {}
    for eval_dataset_name, eval_dataset in trainer.eval_dataset.items():
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=None,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        metrics.update(dataset_metrics)
else:
    metrics = trainer.evaluate(ignore_keys=None)
