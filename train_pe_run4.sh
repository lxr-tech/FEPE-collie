set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --tag='rope_inv_2d_raw_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='inv' --ln='log' --tag='rope_inv_2d_log_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='rope' --imp='inv' --ln='raw' --tag='rope_inv_1d_raw_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='rope' --imp='inv' --ln='log' --tag='rope_inv_1d_log_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' --tag='xpos_inv_2d_raw_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_2d_log_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='inv' --ln='raw' --tag='xpos_inv_1d_raw_rand4' --group='pjlab_fepe_512_rand4'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_1d_log_rand4' --group='pjlab_fepe_512_rand4'
