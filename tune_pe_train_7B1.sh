
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=2048 \
 --tag='rope_inv_2d_raw' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='1d' --exp='rope' --imp='inv' --ln='raw' --log_base=2048 \
 --tag='rope_inv_1d_raw' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=2048 \
 --tag='rope_inv_2d_log' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='1d' --exp='rope' --imp='inv' --ln='log' --log_base=2048 \
 --tag='rope_inv_1d_log' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' --log_base=2048 \
 --tag='xpos_inv_2d_raw' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='1d' --exp='xpos' --imp='inv' --ln='raw' --log_base=2048 \
 --tag='xpos_inv_1d_raw' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=2048 \
 --tag='xpos_inv_2d_log' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='1d' --exp='xpos' --imp='inv' --ln='log' --log_base=2048 \
 --tag='xpos_inv_1d_log' --group='pjlab_fepe_llama2_7B_4096'