set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
 --nnodes=1 --nproc_per_node=4 debug.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='imp' --ln='log' --tag='xpos_imp_1d_log_cl' --group='debug'
