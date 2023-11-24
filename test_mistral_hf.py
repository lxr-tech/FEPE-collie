import os
import sys

import torch
import numpy as np
import random

from datetime import datetime

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.deepspeed import deepspeed_init

from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from models.modeling_mistral import MistralForCausalLM

from utils.arg_parser import arg_parse

tag, path, group, pp_size, tp_size, task, pe_config, data_config, model_args, train_args = arg_parse()

file_name = './csv_logs/{}-{}.txt'.format(group, tag)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.bfloat16)

config_path = '/mnt/petrelfs/share_data/llm_data/mistral-7b-hf'
model_path = '/mnt/petrelfs/share_data/llm_data/mistral-7b-hf'

max_length = max(data_config['ext_lengths'])

config = AutoConfig.from_pretrained(config_path)
config.gradient_checkpointing = True
config.torch_dtype = torch.bfloat16

model = MistralForCausalLM.from_pretrained(model_path, config=config)  # , device_map='auto'

ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e7,
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 1,
    "train_batch_size": 32, 
    "train_micro_batch_size_per_gpu": 4, 
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

dschf = HfDeepSpeedConfig(ds_config)

# deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config)

tokenizer = AutoTokenizer.from_pretrained(model_args['model_path_or_name'], use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
train_length = train_args['max_length']
test_lengths = data_config['ext_lengths']
test_dataset = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-mistral-{}.pkl'.format(max(test_lengths)))


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous().to(logits.device)
    pred = torch.max(logits, dim=-1)[1]    
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    cur_loss = loss_func(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    cur_loss = cur_loss.reshape((-1, max_length - 1)).float()
    cum_loss = torch.cumsum(cur_loss, dim=-1)
    cum_loss = cum_loss / torch.arange(1, max_length, 1, device='cuda').reshape((1, -1))

    cur_acc = torch.cumsum(torch.equal(pred, labels), axis=-1)
    cur_acc = cur_acc / torch.arange(1, max_length, 1, device='cuda').reshape((1, -1))

    return {'cum_ppl': cum_loss, 'cur_acc': cur_acc, }


training_args = TrainingArguments(deepspeed=ds_config, report_to='none', bf16=True, bf16_full_eval=True, 
                                  per_device_train_batch_size=4, per_device_eval_batch_size=1, output_dir='checkpoints')

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer,
                  train_dataset=test_dataset, eval_dataset=test_dataset, 
                  compute_metrics=compute_metrics)  # , is_model_parallel=True

# deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(trainer, num_training_steps=trainer.args.max_steps)
# trainer.model = deepspeed_engine.module
# trainer.model_wrapped = deepspeed_engine
# trainer.deepspeed = deepspeed_engine
# trainer.optimizer = optimizer
# trainer.lr_scheduler = lr_scheduler
# trainer.model_wrapped = trainer._wrap_model(trainer.model_wrapped)

metrics = trainer.evaluate()
metrics = {'cum_ppl': torch.exp(metrics['cum_ppl']), 'cur_acc': metrics['cur_acc']}

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

if rank == 0:
    file = open(file_name, 'a')
    file.write(str(datetime.now()) + '\n\n')
    file.write('model type : {} \n'.format(tag))
    file.write('{}\n'.format(pe_config))
    file.write('{}\n'.format(model_args))
    file.write('{}\n'.format(train_args))
    file.write('{}\n\n'.format(config))
    file.write('{}\n\n'.format(metrics))
    file.close()
