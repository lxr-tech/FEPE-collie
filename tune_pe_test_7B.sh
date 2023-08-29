
# srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
#  --tag='rope_inv_2d_raw' --group='pjlab_fepe_llama2_13B_4096' --ntk_option='none'
# wait
# srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
#  --tag='rope_inv_2d_raw_ntk_fixed' --group='pjlab_fepe_llama2_13B_4096' --ntk_option='fixed' --ntk_alpha=2
# wait
# srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
#  --tag='rope_inv_2d_raw_ntk_fixed' --group='pjlab_fepe_llama2_13B_4096' --ntk_option='fixed' --ntk_alpha=8
# wait
# srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
#  --tag='rope_inv_2d_raw_ntk_dynamic' --group='pjlab_fepe_llama2_13B_4096' --ntk_option='dynamic'
# wait
srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='log' \
 --base=1000 --log_base=4096 \
 --tag='hang_1000-log' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' \
 --base=1000 --exp_base=4096 \
 --tag='hang_1000-xpos' --group='pjlab_fepe_llama2_7B_4096'
wait
srun -p p4_test --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='xpos' --imp='inv' --ln='log' \
 --base=1000 --log_base=4096 --exp_base=4096 \
 --tag='hang_1000-xpos_log' --group='pjlab_fepe_llama2_7B_4096'
