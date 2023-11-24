
import os
import math

import torch
from torch.optim import AdamW

from datetime import datetime

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from collie import CollieConfig, Trainer, env
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import ColliePadder, CheckpointCallback, GPTLMLoss

from models.collie_llama_with_pe4 import LlamaForCausalLM

from utils.arg_parser import arg_parse
from utils.clm_tools_acc import EvaluatorForExtrapolation, CumGPTLMLoss, CumPPLMetric, CumAccMetric

tag, path, group, args, task, pe_config, ds_config, model_args, train_args = arg_parse()
"""
paths = {'scaling_rope-1B_b500_v0-ckpt_s4000': 'qianxuesen_1B/xingshuhao_v_0_2_4/4000/', 
         'scaling_rope-1B_b500_v0-ckpt_s12000': 'qianxuesen_1B/xingshuhao_v_0_2_4/12000/', 
         'scaling_rope-1B_b500_v1-ckpt_s2000': 'qianxuesen_1B/xingshuhao_v_0_2_5/2000', 
         'scaling_rope-1B_b500_v1-ckpt_s4000': 'qianxuesen_1B/xingshuhao_v_0_2_5/4000/', 
         'scaling_rope-1B_b500_v1-ckpt_s8000': 'qianxuesen_1B/xingshuhao_v_0_2_5/8000/', 
         'scaling_rope-1B_b500_v1-ckpt_s16000': 'qianxuesen_1B/xingshuhao_v_0_2_5/16000/', 
         'scaling_rope-1B_b500_v1-ckpt_s32000': 'qianxuesen_1B/xingshuhao_v_0_2_5/32000/', 
         'scaling_rope-1B_b500_v1-ckpt_s64000': 'qianxuesen_1B/xingshuhao_v_0_2_5/64000/', 
         'scaling_rope-7B_b500_v0-ckpt_s18000': 'official_qianxuesen_base_500_7B_v1.0.0/18000', 
         'scaling_rope-7B_b500_v1-ckpt_s11000': 'official_qianxuesen_base_500_7B_v1.0.1/11000', 
         'scaling_rope-7B_b500_v1-ckpt_s18000': 'official_qianxuesen_base_500_7B_v1.0.1/18000', 
         'scaling_rope-7B_b500_v1-ckpt_s18000_ft': 'official_qianxuesen_base_500_7B_v1.0.1_18000step_base_10000_fp/1024/', 
         }
"""
paths = {'shuxingbei_1B_b10000': 'official_Shuxingbei_1B_b10000', 
         'shuxingbei_1B_b10000_log': 'official_Shuxingbei_1B_b10000_log', 
         'shuxingbei_1B_b2608': 'official_Shuxingbei_1B_b2608', 
         'shuxingbei_1B_b2608_log': 'official_Shuxingbei_1B_b2608_log', 
         'shuxingbei_1B_b1304': 'official_Shuxingbei_1B_b1304', 
         'shuxingbei_1B_b1304_log': 'official_Shuxingbei_1B_b1304_log', 
         'shuxingbei_llama2_7B_b500000_fp': 'official_Shuxingbei_7B_llama2base500000', 
         'shuxingbei_llama2_7B_b2000000_fp': 'official_Shuxingbei_7B_llama2base2000000',
         'shuxingbei_llama2_7B_b2000000_fp2': 'official_Shuxingbei_7B_llama2base2000000_fix_weightdecay',
         'shuxingbei_llama2_7B_b2000000_fp2_16K': 'official_Shuxingbei_7B_llama2base2000000_fix_weightdecay_16k_context',
         'qianxuesen_7B': 'official_qianxuesen_7B_v1.0.0', 
         'shuxingbei_7B_b500': 'official_Shuxingbei_7B_b500', 
         'shuxingbei_7B_b500-ckpt50000_b10000_fp': 'official_Shuxingbei_7B_b500_fp10000', 
         'shuxingbei_7B_b500-ckpt80000_b10000_fp': 'official_Shuxingbei_7B_b500_fp10000_from80000', 
        #  'shuxingbei_7B_b500_b10000_fp': 'official_Shuxingbei_7B_b500_fp10000',
         }

root_f = '/mnt/petrelfs/share_data/xingshuhao/exported_transformers'
root_p = 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints'

config_path = f'{root_f}/{paths[group]}/{args.ckpt}/'

model_path = f'{root_f}/{paths[group]}/{args.ckpt}/' if path in paths else '{}/{}-ckpt_{}-{}/epoch_1/'.format(root_p, group, args.ckpt, path)

config = CollieConfig.from_pretrained(config_path, trust_remote_code=True)

config.model_config.use_cache = False
config.checkpointing = True

if not group.__contains__('rand'):
    config.seed = 42
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

config.pp_size = args.pp_size
config.tp_size = args.tp_size
config.train_micro_batch_size = train_args['train_micro_batch_size'] // args.pp_size if task['training'] else train_args['train_micro_batch_size'] 
config.gradient_accumulation_steps = args.pp_size if task['training'] else 1
config.eval_batch_size = train_args['eval_batch_size']
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
        'enabled': task['training'],
        'tag': f'{group}-ckpt_{args.ckpt}-{tag}-',  # tag
        'csv_monitor': {  # wandb
            'enabled': task['training'],
            'output_path': '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_monitor/', 
            'job_name': f'{group}-ckpt_{args.ckpt}-{tag}'
        }
    },
    'gradient_clipping': train_args['max_grad_norm'],
    'zero_optimization': {
        "stage": 3 if args.pp_size == 1 else 0, 
    },
}

config.__setattr__('pe_config', pe_config)

file_name = './csv_logs/{}-ckpt_{}-{}.txt'.format(group, args.ckpt, tag)
    
config.__setattr__('file_name', file_name)

tokenizer = model_args['model_path_or_name']

model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path, config=config, trust_remote_code=True, 
                                         protocol='petrel' if model_path.__contains__(':') else 'file')

rank = env.rank

if ds_config['dataset'] == 'pile':
    train_length = train_args['max_length']
    test_lengths = ds_config['ext_lengths']

    train_path = 'pile-train-llama-{}.pkl'.format(train_args['max_length'])
    test_path = 'books3-test-llama-{}.pkl'.format(max(test_lengths))

    num_training_steps, num_warmup_steps = train_args['train_steps'], train_args['warmup_steps']
    num_data = train_args['train_micro_batch_size'] * env.dp_size * num_training_steps
    
    print('num_training_steps', num_training_steps, 'num_warmup_steps', num_warmup_steps)
    print('num_data', num_data, 'num_token', num_data * train_length)

    from utils.clm_tools_pile import get_pile_for_perplexity
    
    tokenizer, train_dataset, test_datasets = get_pile_for_perplexity(tokenizer=tokenizer, num_data=num_data, 
        train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)
    
elif ds_config['dataset'] == 'pajama':
    train_length = train_args['max_length']
    test_lengths = ds_config['ext_lengths']

    test_path = 'books3-test-llama-{}.pkl'.format(max(test_lengths))

    num_training_steps, num_warmup_steps = train_args['train_steps'], train_args['warmup_steps']
    num_data = train_args['train_micro_batch_size'] * env.dp_size * num_training_steps
    
    print('num_training_steps', num_training_steps, 'num_warmup_steps', num_warmup_steps)
    print('num_data', num_data, 'num_token', num_data * train_length)

    from utils.clm_tools_pajama import get_pajama_for_perplexity
    
    tokenizer, train_dataset, test_datasets = get_pajama_for_perplexity(tokenizer=tokenizer, num_data=num_data, 
        train_length=train_length, train_path=None, test_lengths=test_lengths, test_path=test_path)

else:
    sys.exit()

if train_args['optim'] == 'AdamW':
    optimizer = AdamW(model.parameters(), lr=train_args['learning_rate'])  # max_grad_norm=train_args['max_grad_norm'])
else:
    optimizer = None
    
if train_args['lr_scheduler_type'] == 'CosineAnnealingWarmup':
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps)
elif train_args['lr_scheduler_type'] == 'WarmupDecayLR':
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps)
elif train_args['lr_scheduler_type'] == 'ConstantWarmup':
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, 
                                                     num_warmup_steps=num_warmup_steps)
else:
    lr_scheduler = None

evaluators = []

item = str(max(test_lengths))
evaluators.append(EvaluatorForExtrapolation(model=model, dataset=test_datasets[item], monitors=[EvalMonitor(config) ], 
                                            config=config, loss_fn=CumGPTLMLoss(max_len=max(test_lengths), ignore_index=1), 
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

if task['training'] and not group.__contains__('rand') and not group.__contains__('debug'):
    callbacks = [CheckpointCallback(folder='p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/{}-ckpt_{}-{}'.format(group, args.ckpt, tag), model_only=True, 
                                    every_n_epochs=train_args['save_every_n_epochs'], protocol='petrel')]
else:
    callbacks = []

trainer = Trainer(model=model, tokenizer=tokenizer, config=config,
                  optimizer=optimizer, lr_scheduler=lr_scheduler,
                  loss_fn=GPTLMLoss(ignore_index=1),
                  train_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 1, "labels": 1}, padding_left=False),
                  eval_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 1, "labels": 1}, padding_left=False),
                  train_dataset=train_dataset, evaluators=evaluators, 
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
    if task['training']:
        trainer.train()
    else:
        trainer.eval()
except BaseException as e:
    if env.rank == 0:
        import sys
        import traceback
        from rich.console import Console
        file = open("traceback.log", 'a+')
        file.write(str(datetime.now()) + "\n\n")
        sys.stdout = file
        traceback.print_exc(file=file)
        file.write("\n\n")
        # Console().print_exception()
        raise e

if env.rank == 0:
    file = open(file_name, 'a')
    file.write('}\n\n')
    file.close()

