
import os
import math
import torch

from datetime import datetime

from transformers import get_linear_schedule_with_warmup

from collie import CollieConfig, Trainer, env, setup_distribution
from collie import EvaluatorForPerplexity, PPLMetric, AccuracyMetric
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import CheckpointCallback
from collie import ColliePadder, GPTLMLoss

from models.hf_ds_llama_with_pe import LlamaForCausalLM

from utils.arg_parser import arg_parse
from utils.clm_tools_acc import EvaluatorForExtrapolation
from utils.clm_tools_arxiv import get_arxiv_for_perplexity

tag, group, pe_config, model_args, train_args = arg_parse()

rank = env.local_rank  #  int(os.environ["LOCAL_RANK"])
size = env.world_size  #  int(os.environ["WORLD_SIZE"])

config = CollieConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.model_config.hidden_size = model_args['hidden_size']
config.model_config.intermediate_size = model_args['intermediate_size']
config.model_config.num_attention_heads = model_args['num_attention_heads']
config.model_config.num_hidden_layers = model_args['num_hidden_layers']
config.model_config.use_cache = False
tokenizer = model_args['tokenizer']

config.init_method = lambda x: torch.nn.init.normal_(x, mean=0., std=0.002) if x.ndim == 2 else torch.nn.init.ones_(x)
config.seed = 42

config.train_micro_batch_size = train_args['train_micro_batch_size'] // size
config.eval_batch_size = train_args['eval_batch_size']
config.train_epochs = train_args['train_epochs']
config.eval_per_n_epochs = train_args['eval_per_n_epochs']
config.eval_per_n_steps = train_args['eval_per_n_steps']

train_length = train_args['max_length']
test_lengths = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, ]
train_path = 'arxiv-train-{}-{}.pkl'.format(model_args['size'], train_args['max_length'])
test_path = 'arxiv-test-{}-{}.pkl'.format(model_args['size'], test_lengths[-1])

tokenizer, train_dataset, test_datasets = get_arxiv_for_perplexity(tokenizer=tokenizer, 
    train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)

num_training_batchs = math.floor(len(train_dataset) / config.train_micro_batch_size)
num_training_steps = math.floor(num_training_batchs / size) * train_args['train_epochs']
num_warmup_steps = int(num_training_steps * train_args['warmup_ratio'])

config.use_flash = False
config.ds_config = {
    'bf16': {'enabled': True},
    'monitor_config': {
        'enabled': True,
        'tag': tag,  # tag 表示 一次run的名字，对应图表中 一条线
        'wandb': {
            'enabled': True,
            'team': 'xrliu',
            'project': 'fepe_collie',
            'group': group  # group是run的集合，对应一张图表，不同monitor不同的子图
        }
    },
    'optimizer': {
        'type': train_args['optim'],  # 'FusedLamb',  
        'params': {
            'lr': train_args['learning_rate'],
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': train_args['weight_decay']
        }
    }, 
    'gradient_clipping': train_args['max_grad_norm'], 
    'scheduler': {
        'type': 'WarmupLR',
        'params': {
            'warmup_min_lr': 0,
            'warmup_max_lr': train_args['learning_rate'],
            'warmup_num_steps': num_warmup_steps,
            'warmup_type': 'linear'
        }
    }
}

config.__setattr__('pe_config', pe_config)

file_name = './csv_logs/{}-{}.txt'.format(group, tag)
    
config.__setattr__('file_name', file_name)

model = LlamaForCausalLM(config=config.model_config, pe_config=pe_config)

setup_distribution(config)

evaluators = []

for item in test_datasets:
    evaluators.append(EvaluatorForExtrapolation(model=model, dataset=test_datasets[item], 
                                                config=config, monitors=[EvalMonitor(config) ], 
                                                metrics={'{}'.format(item): AccuracyMetric(gather_result=True), 
                                                         '{}#ppl'.format(item): PPLMetric(gather_result=True)}))

model_size = sum([param.nelement() for param in model.parameters()]) / 1e6

if rank == 0:
    print('model type :', tag, '\n')
    print('model size :', model_size, 'M \n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print(config, '\n')

trainer = Trainer(model=model, tokenizer=tokenizer, config=config,
                #   optimizer=optimizer, lr_scheduler=lr_scheduler,
                  loss_fn=GPTLMLoss(ignore_index=0),
                  train_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 0, "labels": 0}, padding_left=False),
                  eval_dataset_collate_fn=ColliePadder(padding_token_id={"attention_mask": 0, "labels": 0}, padding_left=False),
                  train_dataset=train_dataset, evaluators=evaluators, 
                  monitors=[LossMonitor(config), LRMonitor(config), 
                            TGSMonitor(config), MemoryMonitor(config), ], 
                  callbacks=[CheckpointCallback(folder='checkpoints/{}-{}'.format(group, tag), model_only=True, 
                                                every_n_epochs=train_args['save_every_n_epochs'])]
                  )

if env.local_rank == 0:
    file = open(file_name, 'a')
    file.write(str(datetime.now()) + '\n\n')
    file.write('model type : {} , model size : {}M \n'.format(tag, model_size))
    file.write('{}\n'.format(pe_config))
    file.write('{}\n'.format(model_args))
    file.write('{}\n'.format(train_args))
    file.write('{}\n\n'.format(config))
    file.write("'{}': {}\n".format(tag, '{'))
    file.close()

trainer.train()

if env.local_rank == 0:
    file = open(file_name, 'a')
    file.write('}\n\n')
    file.close()

