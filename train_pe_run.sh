set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='inv' --log='raw' --tag='bf16_rope_inv_2d_raw' --group='test_fepe_collie' \
 > nohup_logs/clm_330M_512_bf16_rope_inv_2d_raw.log 2>&1
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='xpos' --imp='inv' --log='raw' --tag='bf16_xpos_inv_2d_raw' --group='test_fepe_collie' \
 > nohup_logs/clm_330M_512_bf16_xpos_inv_2d_raw.log 2>&1
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='imp' --log='raw' --tag='bf16_rope_imp_2d_raw' --group='test_fepe_collie' \
 > nohup_logs/clm_330M_512_bf16_rope_imp_2d_raw.log 2>&1
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='inv' --log='log' --tag='bf16_rope_inv_2d_log' --group='test_fepe_collie' \
 > nohup_logs/clm_330M_512_bf16_rope_inv_2d_log.log 2>&1


# parser.add_argument('--dim', type=str, default='2d', choices=['2d', '1d'])
# parser.add_argument('--exp', type=str, default='rope', choices=['rope', 'xpos'])
# parser.add_argument('--imp', type=str, default='inv', choices=['inv', 'imp'])
# parser.add_argument('--log', type=str, default='raw', choices=['raw', 'log'])

# parser.add_argument('--post_norm_attn', type=bool, default=False)
# parser.add_argument('--post_norm_ffn', type=bool, default=False)
# parser.add_argument('--post_init', type=bool, default=True)

# parser.add_argument('--qk_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
# parser.add_argument('--vo_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
# parser.add_argument('--pe_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
# parser.add_argument('--ffn_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
# parser.add_argument('--norm_fp', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])

# parser.add_argument('--key', type=str, default=' ')
# parser.add_argument('--max_length', type=int, default=512)
# parser.add_argument('--model_size', type=str, default='330M', choices=['330M', '3B', '7B'])
