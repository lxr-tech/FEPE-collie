"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4 , H=256, A=4; L=4 , H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
  - batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4
"""

model_args = {
    'clm_arxiv_0': {
        'hidden_size': 896, 'intermediate_size': 3584, 'num_attention_heads': 12, 'num_hidden_layers': 16,
    },
    'clm_arxiv_1': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    'clm_arxiv_2': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 8, 'num_hidden_layers': 16,
    },
    'clm_arxiv_8192': {
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    'clm_llama_f': {
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },
}

train_args = {
    ('clm_arxiv_0', 512): {
        'per_device_train_batch_size': 4, 'per_device_eval_batch_size': 4, 'num_train_epochs': 10,
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.00012, 'weight_decay': 0, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('clm_arxiv_1', 512): {
        'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 10,  # 4 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.00015, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
        # 'per_device_train_batch_size': 48, 'per_device_eval_batch_size': 24, 'num_train_epochs': 10,  # 4 cards
        # 'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        # 'learning_rate': 0.0003, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        # 'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        # 'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('clm_arxiv_2', 512): {
        'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 10,  # 4 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.00015, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('clm_arxiv_1', 2048): {
        'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 10,  # 4 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.0004, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('clm_arxiv_2', 2048): {
        'per_device_train_batch_size': 6, 'per_device_eval_batch_size': 6, 'num_train_epochs': 10,  # 6 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.0003, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('clm_llama_f', 2048): {
        'per_device_train_batch_size': 2, 'per_device_eval_batch_size': 2, 'num_train_epochs': 10,  # 6 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.0003, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
}

"""
configs = {
    ('ccdv/arxiv-summarization', 'exp_flash', 'EleutherAI/gpt-j-6b'):  {
        'num_layers': 12, 'd_model': 896, 'n_heads': 16, 'ffn_dim_rate': 4,
        'lr': 0.00008, 'batch_size': 16, 'warmup_steps': 0.05, 'optim_type': 'adamW', 'weight_decay': 0,
        'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.1, 'num_epoch': 20,
    },
    ('wikitext-103-v1', 'exp_flash', 'EleutherAI/gpt-j-6b'):  {
        'num_layers': 12, 'd_model': 896, 'n_heads': 16, 'ffn_dim_rate': 4,
        'lr': 0.00008, 'batch_size': 16, 'warmup_steps': 0.05, 'optim_type': 'adamW', 'weight_decay': 0,
        'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.1, 'num_epoch': 20,
    },
    ('ccdv/arxiv-summarization', 'alibi', 'EleutherAI/gpt-j-6b'):  {
        'num_layers': 12, 'd_model': 896, 'n_heads': 16, 'ffn_dim_rate': 4,
        'lr': 0.00005, 'batch_size': 8, 'warmup_steps': 0.1, 'optim_type': 'adamW', 'weight_decay': 0,
        'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.1, 'num_epoch': 10,
    },
    ('wikitext-103-v1', 'alibi', 'EleutherAI/gpt-j-6b'): {
        'num_layers': 12, 'd_model': 896, 'n_heads': 16, 'ffn_dim_rate': 4,
        'lr': 0.00003, 'batch_size': 8, 'warmup_steps': 0.1, 'optim_type': 'adamW', 'weight_decay': 0,
        'after_norm': 1, 'dropout': 0.15, 'fc_dropout': 0.1, 'num_epoch': 10,
    },
}
"""
