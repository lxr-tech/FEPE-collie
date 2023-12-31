
import argparse
import torch
import sys

ext_dict = {
    '10k': [10240, 9216, 8192, 7168, 6144, 5120, 4096, 3072, 2048, 1024, 512, ],
    '20k': [20480, 18432, 16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, 1024, ],
    '32k': [32768, 30720, 28672, 26624, 24576, 22528, 20480, 18432, 
            16384, 14336, 12288, 10240, 8192, 6144, 4096, 2048, ],
    '48k': [49152, 45056, 40960, 36864, 32768, 28672, 
            24576, 20480, 16384, 12288, 8192, 4096], 
    '64k': [65536, 49152, 32768, 16384, 4096, ], 
    '100k': [102400, 81920, 65536, 49152, 32768, 4096], 
    '128k': [131072, 110592, 4096], 
    '256k': [262144], 
    'gen_100k': [102400 + 256, 81920 + 256, 65536 + 256, 49152 + 256, 
                 32768 + 256, 16384 + 256, 4096 + 256, 2048 + 256], 
}

def arg_parse():
    
    parser = argparse.ArgumentParser(description='define pe fp config')
    
    parser.add_argument('--task_a', type=str, default='pretrain', choices=['pretrain', 'finetune'])
    parser.add_argument('--task_b', type=str, default='training', choices=['training', 'testing'])

    parser.add_argument('--dim', type=str, default='2d', choices=['2d', '1d'])
    parser.add_argument('--exp', type=str, default='rope', choices=['rope', 'xpos'])
    parser.add_argument('--imp', type=str, default='inv', choices=['inv', 'imp'])
    parser.add_argument('--ln', type=str, default='raw', choices=['raw', 'log'])
    
    parser.add_argument('--log_base', type=float, default=torch.e)
    parser.add_argument('--log_clip', type=float, default=1)
    parser.add_argument('--exp_base', type=float, default=512.)
    
    parser.add_argument('--base', type=float, default=10000.)
    parser.add_argument('--pi_lambda', type=float, default=1.)

    parser.add_argument('--ntk_option', type=str, default='none', choices=['none', 'fixed', 'dynamic', 'dynamic_new'])
    parser.add_argument('--ntk_alpha', type=float, default=1.)

    # parser.add_argument('--post_norm_attn', type=str, default='false')
    # parser.add_argument('--post_norm_ffn', type=str, default='false')
    # parser.add_argument('--post_init', type=str, default='true')

    # parser.add_argument('--qk_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    # parser.add_argument('--vo_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    # parser.add_argument('--pe_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    # parser.add_argument('--ffn_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    # parser.add_argument('--norm_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', type=int, default=0)

    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--group', type=str, default='')
    
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)

    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_size', type=str, default='330M', choices=['330M', '3B', 'llama-7B',  
                                                                           'llama2-7B', 'llama2-13B', 'llama2-70B'])
    parser.add_argument('--dataset', type=str, default='pile', choices=['arxiv', 'pile', 'leval', 
                                                                        'code', 'pajama', ])
    parser.add_argument('--ext_length', type=str, default='32k', choices=list(ext_dict))

    args = parser.parse_args()
    
    if args.task_a == 'pretrain':
        from configs.clm_train_config import model_args, train_args
    elif args.task_a == 'finetune':
        from configs.clm_tune_config import model_args, train_args
    else:
        sys.exit()
    
    task = {'pretrain': args.task_a == 'pretrain', 'training': args.task_b == 'training'}

    # fp = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

    # fp_config = {'pe_fp': fp[args.pe_fp], 'qk_fp': fp[args.qk_fp], 'vo_fp': fp[args.vo_fp],
    #              'norm_fp': fp[args.norm_fp], 'ffn_fp': fp[args.ffn_fp], }

    pe_config = {'exp': True if args.exp == 'xpos' else False, '1d': True if args.dim == '1d' else False, 
                 'imp': True if args.imp == 'imp' else False, 'log': True if args.ln == 'log' else False, 
                 'exp_base': args.exp_base, 'log_base': args.log_base, 'log_clip': args.log_clip, 
                 'max_length': args.max_length, 'pi_lambda': args.pi_lambda, 'base': args.base, 
                 'ntk_option': args.ntk_option, 'ntk_alpha': args.ntk_alpha, }
    
    ds_config = {'dataset': args.dataset, 'ext_lengths': ext_dict[args.ext_length], }

    # hp_config = {'init': args.post_init, 'post': args.post_norm_attn, 'both': args.post_norm_ffn, }

    model_size, max_length = args.model_size, args.max_length

    assert args.tag != '' and args.path != '' and args.group != ''

    tag, path, group = args.tag, args.path, args.group

    assert model_size in model_args and (model_size, max_length) in train_args

    model_arg, train_arg = model_args[model_size], train_args[(model_size, max_length)]

    return tag, path, group, args, task, pe_config, ds_config, model_arg, train_arg  # fp_config, hp_config


