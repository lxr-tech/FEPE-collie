import torch

from datetime import datetime

from collie import CollieConfig, Trainer
from collie import EvaluatorForPerplexity, PPLMetric, AccuracyMetric
from collie import EvalMonitor, LossMonitor, LRMonitor, TGSMonitor, MemoryMonitor
from collie import env, setup_distribution

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

config.train_micro_batch_size = train_args['train_micro_batch_size']
config.eval_batch_size = train_args['eval_batch_size']
config.train_epochs = train_args['train_epochs']
config.eval_per_n_epochs = train_args['eval_per_n_epochs']
config.eval_per_n_steps = train_args['eval_per_n_steps']

config.use_flash = True
config.ds_config = {
    'bf16': {'enabled': True},
}

config.__setattr__('pe_config', pe_config)

file_name = './csv_logs/{}-{}.txt'.format(group, tag)
    
config.__setattr__('file_name', file_name)

pe_config['flash_train'] = True
pe_config['flash_test'] = True
pe_config['init'] = True
pe_config['post'] = False
pe_config['both'] = False

## old run loaded in huggingface version
# import sys
# sys.path.append('/remote-home/xrliu/projects/FEPE-deepspeed/')
# from models.llama_with_pe import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path='/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/init_pre_{}/train_last/'.format(tag), 
#                                          pe_config=pe_config)

## old run loaded in collie version (from_pretrained), wrong !
# from models.collie_llama_with_pe import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained(model_path_or_name='/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/init_pre_{}/train_last/'.format(tag), 
#                                          config=config)

## old run loaded in collie version (from_config)
import torch
from models.collie_llama_with_pe import LlamaForCausalLM
state_dict1 = torch.load('/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints/init_pre_{}/train_last/pytorch_model.bin'.format(tag)) 
state_dict2 = {}

for key, value in state_dict1.items():
    if key.startswith('model.'):
        state_dict2[key[6:]] = value
    else:
        state_dict2[key] = value
model = LlamaForCausalLM.from_config(config=config)
model.load_state_dict(state_dict2)

## new run loaded in collie version
# model = LlamaForCausalLM.from_pretrained(model_path_or_name='./checkpoints/{}-{}/epoch_1'.format(group, tag), 
#                                          config=config)

setup_distribution(config)

train_length = train_args['max_length']
test_lengths = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, ]
train_path = 'arxiv-train-{}-{}.pkl'.format(model_args['size'], train_args['max_length'])
test_path = 'arxiv-test-{}-{}.pkl'.format(model_args['size'], test_lengths[-1])

tokenizer, train_dataset, test_datasets = get_arxiv_for_perplexity(tokenizer=tokenizer, 
    train_length=train_length, train_path=train_path, test_lengths=test_lengths, test_path=test_path)

model_size = sum([param.nelement() for param in model.parameters()]) / 1e6

if rank == 0:
    print('model type :', tag, '\n')
    print('model size :', model_size, 'M \n')
    print(pe_config, '\n')
    print(model_args, '\n')
    print(train_args, '\n')
    print(config, '\n')

evaluators = []

for item in [test_datasets]:  # ['512', '1024', '2048', '4096']:  
    evaluators.append(EvaluatorForExtrapolation(model=model, dataset=test_datasets[item], 
                                                config=config, monitors=[EvalMonitor(config) ], 
                                                metrics={'{}'.format(item): AccuracyMetric(gather_result=True), 
                                                         '{}#ppl'.format(item): PPLMetric(gather_result=True)}))

trainer = Trainer(model=model, tokenizer=tokenizer, config=config,
                  train_dataset=train_dataset, evaluators=evaluators)

if env.local_rank == 0:
    file = open(file_name, 'a')
    file.write(str(datetime.now()) + '\n\n')
    file.write("'{}': {}\n".format(tag, '{'))
    file.close()

trainer.eval()

if env.local_rank == 0:
    file = open(file_name, 'a')
    file.write('}\n\n')
    file.close()

