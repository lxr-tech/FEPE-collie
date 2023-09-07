
model_args = {
    'llama2-7B': {
        'size': 'llama2-7B', 'model_path_or_name': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',  # "vocab_size": 32000
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },    
    'llama2-13B': {
        'size': 'llama2-13B', 'model_path_or_name': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-13b-hf/',  # "vocab_size": 32000
        'hidden_size': 5120, 'intermediate_size': 13824, 'num_attention_heads': 40, 'num_hidden_layers': 40,
    },    
}

# '/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf/',  # in s cluster
# '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',  # in a800 cluster


train_args = {
    ('llama2-7B', 4096): {        
        'max_length': 4096, 'world_size': 32, 'train_micro_batch_size': 4, 'eval_batch_size': 1, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 2.5, 'weight_decay': 0, 
        'lr_scheduler_type': 'none', 'train_steps': 1024, 'warmup_steps': 0,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
    ('llama2-7B', 8192): {        
        'max_length': 8192, 'world_size': 32, 'train_micro_batch_size': 2, 'eval_batch_size': 1, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 2.5, 'weight_decay': 0, 
        'lr_scheduler_type': 'none', 'train_steps': 1024, 'warmup_steps': 0,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
    ('llama2-13B', 4096): {        
        'max_length': 4096, 'world_size': 32, 'train_micro_batch_size': 4, 'eval_batch_size': 1, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 1, 'weight_decay': 0, 
        'lr_scheduler_type': 'none', 'train_steps': 1024, 'warmup_steps': 0, 
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
    ('llama2-13B', 8192): {        
        'max_length': 8192, 'world_size': 32, 'train_micro_batch_size': 2, 'eval_batch_size': 1, 
        'optim': 'AdamW', 'learning_rate': 0.00002, 'max_grad_norm': 1, 'weight_decay': 0, 
        'lr_scheduler_type': 'none', 'train_steps': 1024, 'warmup_steps': 0, 
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
}
