set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=8 test_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --tag='rope_inv_2d_raw' --group='pjlab_fepe_3B_2048'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=8 test_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' --tag='xpos_inv_2d_raw' --group='pjlab_fepe_3B_2048'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=8 test_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_2d_log' --group='pjlab_fepe_3B_2048'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=8 test_pe.py --model_size='3B' --max_length=2048 \
 --dim='1d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_1d_log' --group='pjlab_fepe_3B_2048'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=8 test_pe.py --model_size='3B' --max_length=2048 \
 --dim='1d' --exp='xpos' --imp='imp' --ln='log' --tag='xpos_imp_1d_log' --group='pjlab_fepe_3B_2048'
