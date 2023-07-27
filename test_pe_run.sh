set -x
port=$(shuf -i25000-30000 -n1)

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
 --nnodes=1 --nproc_per_node=4 test_pe.py --model_size='330M' --max_length=512 \
 --dim='1d' --exp='xpos' --imp='imp' --log='log' --tag='xpos_imp_1d_log' --group='test_fepe_collie'
# wait
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
#  --nnodes=1 --nproc_per_node=4 test_pe.py --model_size='330M' --max_length=512 \
#  --dim='2d' --exp='rope' --imp='imp' --log='raw' --tag='rope_imp_2d_raw' --group='test_fepe_collie'
# wait
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
#  --nnodes=1 --nproc_per_node=4 test_pe.py --model_size='330M' --max_length=512 \
#  --dim='1d' --exp='rope' --imp='inv' --log='raw' --tag='rope_inv_1d_raw' --group='test_fepe_collie'
# wait
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29403 \
#  --nnodes=1 --nproc_per_node=4 test_pe.py --model_size='330M' --max_length=512 \
#  --dim='2d' --exp='rope' --imp='inv' --log='log' --tag='rope_inv_2d_log' --group='test_fepe_collie'