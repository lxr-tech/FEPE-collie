
import os
import math

import torch
from torch.optim import AdamW

from datetime import datetime

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

# import sys
# sys.path.append('../collie/')

from collie import CollieConfig, Trainer, env
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import ColliePadder, CheckpointCallback, GPTLMLoss

from models.collie_llama_with_pe import LlamaForCausalLM
from models.tokenization_internlm import InternLMTokenizer

from utils.arg_parser import arg_parse
from utils.clm_tools_acc import EvaluatorForExtrapolation, CumGPTLMLoss, CumPPLMetric, CumAccMetric

tag, path, group, pp_size, task, pe_config, ds_config, model_args, train_args = arg_parse()

config = CollieConfig.from_pretrained(name_or_path='/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/models', 
                                      protocol='petrel')

config.model_config.use_cache = False
config.checkpointing = True

if not group.__contains__('rand'):
    config.seed = 42
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

config.pp_size = pp_size
config.train_micro_batch_size = train_args['train_micro_batch_size'] // pp_size if task['training'] else train_args['train_micro_batch_size'] 
config.gradient_accumulation_steps = pp_size if task['training'] else 1
config.eval_batch_size = 1
config.train_epochs = 1
config.eval_per_n_epochs = train_args['eval_per_n_epochs']
config.eval_per_n_steps = train_args['eval_per_n_steps']
config.low_cpu_mem_usage = False

config.use_flash = True
config.ds_config = {
    'bf16': {
        'enabled': True,
    },
    'train_micro_batch_size_per_gpu': config.train_micro_batch_size,  # * config.gradient_accumulation_steps,
    'monitor_config': {
        'enabled': True,
        'tag': f'{model_args["size"]}-{tag}-',  # tag
        'csv_monitor': {  # wandb
            'enabled': task['training'],
            'output_path': '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_monitor/', 
            'job_name': f'newton_20B-{tag}'
            # 'team': 'xrliu',
            # 'project': 'fepe_collie',
            # 'group': group
        }
    },
    'gradient_clipping': train_args['max_grad_norm'],
    'zero_optimization': {
        "stage": 3 if pp_size == 1 else 0, 
    },
}

config.__setattr__('pe_config', pe_config)

file_name = './csv_logs/{}-{}.txt'.format(group, tag)
    
config.__setattr__('file_name', file_name)

tokenizer = InternLMTokenizer('/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model')

model_path_or_name = 'ssd:s3://model_weights/exported_transformers/Newton_20B_0.6.2_HF/21500'
model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path_or_name, 
                                         protocol='petrel', config=config)

rank = env.rank  #  int(os.environ["rank"])

train_path = 'pile-train-llama-{}.pkl'.format(train_args['max_length'])

test_dataset = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-V7-49152.pkl')

optimizer = None
 
lr_scheduler = None

evaluators = []

evaluators.append(EvaluatorForExtrapolation(model=model, dataset=test_dataset, monitors=[EvalMonitor(config) ], 
                                            config=config, loss_fn=CumGPTLMLoss(max_len=49152, ignore_index=1), 
                                            dynamic_enabled=(pe_config['ntk_option'] == 'dynamic'), dynamic_stride=train_args['max_length'],
                                            metrics={'cum#acc': CumAccMetric(gather_result=True), 
                                                     'cum#ppl': CumPPLMetric(gather_result=True)}))

model_size = sum([param.nelement() for param in model.parameters()]) / 1e6

if rank == 0:
    print('model type :', tag, '\n')
    print('model size :', model_size, 'M \n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print(config, '\n')

callbacks = []

trainer = Trainer(model=model, tokenizer=tokenizer, config=config,
                  optimizer=optimizer, lr_scheduler=lr_scheduler,
                  loss_fn=GPTLMLoss(ignore_index=1),
                  train_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 1, "labels": 1}, padding_left=False),
                  eval_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 1, "labels": 1}, padding_left=False),
                  train_dataset=None, evaluators=evaluators, 
                  monitors=[LossMonitor(config), LRMonitor(config), TGSMonitor(config), MemoryMonitor(config), ], 
                  callbacks=callbacks)

if env.rank == 0:
    file = open(file_name, 'a')
    file.write(str(datetime.now()) + '\n\n')
    file.write('model type : {} , model size : {}M \n'.format(tag, model_size))
    file.write('{}\n'.format(pe_config))
    file.write('{}\n'.format(model_args))
    file.write('{}\n'.format(train_args))
    file.write('{}\n\n'.format(config))
    file.write("'{}': {}\n".format(tag, '{'))
    file.close()

try:
    trainer.eval()
except BaseException as e:
    import sys
    import traceback
    from rich.console import Console
    file = open("traceback.log", 'a+')
    file.write(str(datetime.now()) + "\n\n")
    sys.stdout = file
    traceback.print_exc(file=file)
    file.write("\n\n")
    Console().print_exception()
    raise e

if env.rank == 0:
    file = open(file_name, 'a')
    file.write('}\n\n')
    file.close()

