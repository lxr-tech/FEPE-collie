
import os
import math
import torch

from transformers import get_linear_schedule_with_warmup

from collie import CollieConfig, Trainer
from collie import EvaluatorForPerplexity, PPLMetric
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import CheckpointCallback, LoadBestModelCallback

# from collie import LlamaForCausalLM
from models.collie_llama_with_pe import LlamaForCausalLM

from utils.arg_parser import arg_parse
from utils.clm_tools_arxiv import get_arxiv_for_perplexity

key, pe_config, model_args, train_args = arg_parse()

rank = int(os.environ["LOCAL_RANK"])
size = int(os.environ["WORLD_SIZE"])

config = CollieConfig.from_pretrained('/remote-home/share/llama_hf/7B')
config.model_config.hidden_size = model_args['hidden_size']
config.model_config.intermediate_size = model_args['intermediate_size']
config.model_config.num_attention_heads = model_args['num_attention_heads']
config.model_config.num_hidden_layers = model_args['num_hidden_layers']
tokenizer = model_args['tokenizer']

config.train_micro_batch_size = train_args['train_micro_batch_size']
config.eval_batch_size = train_args['eval_batch_size']
config.train_epochs = train_args['train_epochs']
config.eval_per_n_epochs = train_args['eval_per_n_epochs']
config.eval_per_n_steps = train_args['eval_per_n_steps']

config.use_flash = False
config.ds_config = {
    'bf16': {'enable': True},
    'monitor_config': {
        'enabled': True,
        'wandb': {
            'enabled': False,
            'team': 'xrliu',
            'project': 'fepe_collie',
            'group': 'test_wandb_collie'
        }
    },
}

train_length = train_args['max_length']
test_lengths = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, ]
train_path = 'arxiv-train-{}-{}.pkl'.format(model_args['tag'], train_args['max_length'])
test_path = 'arxiv-test-{}-{}.pkl'.format(model_args['tag'], test_lengths[-1])

tokenizer, train_dataset, test_datasets = get_arxiv_for_perplexity(tokenizer=tokenizer, 
    train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)

model = LlamaForCausalLM.from_config(config=config)

if train_args['optim'] == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args['learning_rate'], 
                                  weight_decay=train_args['weight_decay'])
else:
    raise KeyError

num_training_steps = math.floor(len(train_dataset) / train_args['train_micro_batch_size'])
num_warmup_steps = int(num_training_steps * train_args['warmup_ratio'])

if train_args['lr_scheduler_type'] == 'linear':
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                   num_training_steps=num_training_steps,
                                                   num_warmup_steps=num_warmup_steps)
else:
    lr_scheduler = None

evaluators = [EvaluatorForPerplexity(model=model, config=config, dataset=test_datasets[item],
                                     monitors=[EvalMonitor(config) ],
                                     metrics={'ppl#{}'.format(item): PPLMetric(gather_result=True)}
                                     ) for item in test_datasets]

model_size = sum([param.nelement() for param in model.parameters()]) / 1e6

if rank == 0:
    print('model type :', key, '\n')
    print('model size :', model_size, 'M \n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')

trainer = Trainer(model=model, tokenizer=tokenizer, config=config,
                  optimizer=optimizer, lr_scheduler=lr_scheduler,
                  train_dataset=train_dataset, evaluators=evaluators, 
                  monitors=[LossMonitor(config), LRMonitor(config), 
                            TGSMonitor(config), MemoryMonitor(config), ], 
                  callbacks=[CheckpointCallback(folder='checkpoints/{}'.format(key), model_only=True, 
                                                every_n_epochs=train_args['save_every_n_epochs']), 
                             LoadBestModelCallback(folder='checkpoints/{}/best'.format(key), 
                                                   monitor='ppl#{}'.format(train_length), larger_better=False) ]
                  )

trainer.train()