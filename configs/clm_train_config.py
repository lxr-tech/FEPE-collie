"""
  - BERT_base (L=12, H=768, A=12, #Para=110M) and BERT_large (L=24, H=1024, A=16, #Para=340M)
  - FNet hyper L=2, H=128, A=2; L=2, H=256, A=4; L=4 , H=256, A=4; L=4 , H=512, A= 8;
               L=8, H=256, A=4; L=8, H=512, A=8; L=12, H=512, A=8; L=12, H=768, A=12;
  - batch size: 16, 32; Learning rate (Adam): 5e-5, 3e-5, 2e-5; Number of epochs: 2, 3, 4
"""

model_args = {
    '330M': {
        'size': '330M', 'tokenizer': '/remote-home/share/llama_hf/7B',  # "vocab_size": 32000
        'hidden_size': 1024, 'intermediate_size': 4096, 'num_attention_heads': 16, 'num_hidden_layers': 16,
    },
    '3B': {
        'size': '3B', 'tokenizer': 'openlm-research/open_llama_3b',  # "vocab_size": 32000
        'hidden_size': 3200, 'intermediate_size': 8640, 'num_attention_heads': 32, 'num_hidden_layers': 26,
    },
    '7B': {
        'size': '7B', 'tokenizer': 'openlm-research/open_llama_7b',  # "vocab_size": 32000
        'hidden_size': 4096, 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_hidden_layers': 32,
    },
}

train_args = {
    ('330M', 512): {
        'max_length': 512, 'train_micro_batch_size': 48, 'eval_batch_size': 1, 
        'train_epochs': 1, 'optim': 'adamw', 'learning_rate': 0.00015, 'weight_decay': 0.01, 
        'lr_scheduler_type': 'linear', 'warmup_ratio': 0.4, 'max_grad_value': 1.0,
        'eval_per_n_steps': 0, 'eval_per_n_epochs': 1, 'save_every_n_epochs': 1,

        # 'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 10,  # 4 cards
        # 'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        # 'learning_rate': 0.00015, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        # 'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        # 'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',

        # 'per_device_train_batch_size': 48, 'per_device_eval_batch_size': 24, 'num_train_epochs': 10,  # 4 cards
        # 'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        # 'learning_rate': 0.0003, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        # 'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        # 'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('330M', 2048): {
        'per_device_train_batch_size': 12, 'per_device_eval_batch_size': 12, 'num_train_epochs': 10,  # 4 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.0004, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
    ('7B', 2048): {
        'per_device_train_batch_size': 2, 'per_device_eval_batch_size': 2, 'num_train_epochs': 10,  # 6 cards
        'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 'optim': 'adamw_hf',
        'learning_rate': 0.0003, 'weight_decay': 0.01, 'lr_scheduler_type': 'linear', 'warmup_ratio': 0.1,
        'evaluation_strategy': 'epoch', 'eval_accumulation_steps': 1, 'max_grad_norm': 5.0,
        'logging_strategy': 'steps', 'logging_steps': 10, 'output_dir': 'checkpoints', 'save_strategy': 'epoch',
    },
}

