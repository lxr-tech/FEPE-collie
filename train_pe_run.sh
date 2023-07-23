set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='inv' --log='raw' --key='bf16_rope_inv_2d_raw'

# parser.add_argument('--dim', type=bool, default=False)
# parser.add_argument('--exp', type=bool, default=False)
# parser.add_argument('--imp', type=bool, default=False)
# parser.add_argument('--log', type=bool, default=False)

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
