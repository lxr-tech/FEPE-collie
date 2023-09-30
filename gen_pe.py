
import os
import math

import torch
from torch.optim import AdamW

from datetime import datetime

# import sys
# sys.path.append('../collie/')

from collie import CollieConfig, Trainer, env
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import ColliePadder, CheckpointCallback, GPTLMLoss

from models.collie_llama_with_pe3 import LlamaForCausalLM

from utils.arg_parser import arg_parse
from collie import EvaluatorForGeneration, DecodeMetric
from transformers.generation.utils import GenerationConfig

tag, path, group, pp_size, task, pe_config, ds_config, model_args, train_args = arg_parse()

assert task['training'] == False

config = CollieConfig.from_pretrained(model_args['model_path_or_name'])

# assert config.model_config.hidden_size == model_args['hidden_size']
# assert config.model_config.intermediate_size == model_args['intermediate_size']
# assert config.model_config.num_attention_heads == model_args['num_attention_heads']
# assert config.model_config.num_hidden_layers == model_args['num_hidden_layers']

config.model_config.use_cache = False
config.checkpointing = True

if not group.__contains__('rand'):
    config.seed = 42
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

config.pp_size = pp_size
config.train_micro_batch_size = train_args['train_micro_batch_size'] 
config.gradient_accumulation_steps = 1
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
    'train_micro_batch_size_per_gpu': config.train_micro_batch_size,
    'monitor_config': {
        'enabled': True,
        'tag': tag,
    },
    'gradient_clipping': train_args['max_grad_norm'],
}

config.__setattr__('pe_config', pe_config)

file_name = './csv_logs/{}-{}.txt'.format(group, tag)
    
config.__setattr__('file_name', file_name)

tokenizer = model_args['model_path_or_name']

if path in ['llama2-7B', 'llama2-13B', ]:
    model_path_or_name = model_args['model_path_or_name']
    model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path_or_name, config=config)
else:
    model_path_or_name = 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/pjlab_fepe_{}_4096-{}/epoch_1'.format(model_args['size'].replace('-', '_'), path)
    model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path_or_name, 
                                             protocol='petrel', config=config)

rank = env.rank  #  int(os.environ["rank"])

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
else:
    sys.exit()

optimizer = None
    
lr_scheduler = None

callbacks = []

evaluators = []

test_lengths.reverse()

save_path = '/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/gen_monitor'

for item in range(len(test_lengths) - 1):
    cur, nex = test_lengths[item], test_lengths[item+1]
    evaluators.append(EvaluatorForGeneration(model=model, tokenizer=torch, 
                                             dataset=test_datasets[str(cur)], monitors=[EvalMonitor(config) ], 
                                             config=config, metrics={'dec': DecodeMetric(verbose=True, save_to_file=True,
                                                                                         save_path=f'{save_path}/{group}-{tag}.txt')}, 
                                             generation_config=GenerationConfig(eos_token_id=tokenizer.eos_token_id, 
                                                                                pad_token_id=tokenizer.pad_token_id, 
                                                                                num_beams=1, do_sample=False, use_chche=True, 
                                                                                max_new_tokens=256, ), ))  # nex-cur
        
model_size = sum([param.nelement() for param in model.parameters()]) / 1e6

if rank == 0:
    print('model type :', tag, '\n')
    print('model size :', model_size, 'M \n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print(config, '\n')

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

