
model_args = {
    '330M': {
        'size': '330M', 'model_path_or_name': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/', # "vocab_size": 32000
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    '3B': {
        'size': '3B', 'model_path_or_name': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',  # "vocab_size": 32000
        'hidden_size': 3072, 'intermediate_size': 9216, 'num_attention_heads': 32, 'num_hidden_layers': 26,
    },
    '7B': {
        'size': '7B', 'model_path_or_name': '/mnt/petrelfs/share_data/llm_llama2/llm_llama2/llama-2-7b-hf/',  # "vocab_size": 32000
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },
}

train_args = {
    ('330M', 512): {        
        'max_length': 512, 'train_micro_batch_size': 12, 'eval_batch_size': 2,   # 8 cards  
        'train_epochs': 2, 'optim': 'AdamW', 'learning_rate': 0.000075, 'weight_decay': 0.01, 
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'warmup_ratio': 0.5, 'max_grad_norm': 2.5,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 2,
    },
    ('330M', 2048): {
        'max_length': 2048, 'train_micro_batch_size': 12, 'eval_batch_size': 2,   # 8 cards 
        'train_epochs': 2, 'optim': 'AdamW', 'learning_rate': 0.00015, 'weight_decay': 0.01, 
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'warmup_ratio': 0.5, 'max_grad_norm': 5.,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 2,
    },
    ('3B', 2048): {
        'max_length': 2048, 'train_micro_batch_size': 16, 'eval_batch_size': 4,   # 16 cards
        'train_epochs': 1, 'optim': 'AdamW', 'learning_rate': 0.00015, 'weight_decay': 0.1, 
        'lr_scheduler_type': 'CosineAnnealingWarmup', 'warmup_ratio': 0.1, 'max_grad_norm': 1.,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,
    },
}

