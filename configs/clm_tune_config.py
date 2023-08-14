
model_args = {
    'llama-7B': {
        'size': 'llama2-7B', 'tokenizer': 'decapoda-research/llama-7b-hf',  # "vocab_size": 32000
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },    
    'llama2-7B': {
        'size': 'llama2-7B', 'tokenizer': 'decapoda-research/llama-7b-hf',  # "vocab_size": 32000
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },    
}

train_args = {
    ('llama-7B', 2048): {        
        'max_length': 2048, 'world_size': 32, 'train_micro_batch_size': 8, 'eval_batch_size': 2, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 2.5, 'weight_decay': 0, 
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'train_steps': 1024, 'warmup_steps': 2,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
    ('llama2-7B', 4096): {        
        'max_length': 4096, 'world_size': 32, 'train_micro_batch_size': 4, 'eval_batch_size': 2, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 2.5, 'weight_decay': 0, 
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'train_steps': 1024, 'warmup_steps': 2,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
    ('llama2-7B', 8192): {        
        'max_length': 8192, 'world_size': 32, 'train_micro_batch_size': 2, 'eval_batch_size': 2, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 2.5, 'weight_decay': 0,   
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'train_steps': 1024, 'warmup_steps': 2,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
}
