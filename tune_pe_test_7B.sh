
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='baseline' --path='llama2-7B' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='fixed' --ntk_alpha=8 \
#  --tag='ntk_fixed_8' --path='llama2-7B' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='finetune' --path='rope_inv_2d_raw' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
#  --tag='pi_2' --path='rope_inv_2d_raw_pi_2' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000' --path='hang_1000' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000_xpos_log' --path='hang_1000' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='hang_500' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_xpos_log' --path='hang_500' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='ntk_dynamic' --path='llama2-7B' --group='llama2_7B_100k'
# wait
# srun -p p4_test --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='leval' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='ntk_dynamic-leval' --path='llama2-7B' --group='llama2_7B_48k'
# wait
# srun -p p4_test --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='code' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='ntk_dynamic-code' --path='llama2-7B' --group='llama2_7B_48k'
wait
srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='leval' --ext_length='48k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='base_500-leval' --path='hang_500' --group='llama2_7B_48k'
wait
srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='code' --ext_length='48k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='base_500-code' --path='hang_500' --group='llama2_7B_48k'
wait
srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='leval' --ext_length='48k' \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
 --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='base_500_xpos_log-leval' --path='hang_500' --group='llama2_7B_48k'
wait
srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='code' --ext_length='48k' \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
 --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='base_500_xpos_log-code' --path='hang_500' --group='llama2_7B_48k'