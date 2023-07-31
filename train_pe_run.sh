set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='imp' --ln='raw' --tag='rope_imp_2d_raw' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='rope' --imp='imp' --ln='log' --tag='rope_imp_2d_log' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='rope' --imp='imp' --ln='raw' --tag='rope_imp_1d_raw' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='rope' --imp='imp' --ln='log' --tag='rope_imp_1d_log' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='xpos' --imp='imp' --ln='raw' --tag='xpos_imp_2d_raw' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='2d' --exp='xpos' --imp='imp' --ln='log' --tag='xpos_imp_2d_log' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='imp' --ln='raw' --tag='xpos_imp_1d_raw' --group='pjlab_fepe_512'
wait
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29404 \
 --nnodes=1 --nproc_per_node=4 train_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='imp' --ln='log' --tag='xpos_imp_1d_log' --group='pjlab_fepe_512'

