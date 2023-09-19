# srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python gen_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='hang_500' --group='llama2_7B_gen_48k'
# wait
# srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python gen_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000' --path='hang_1000000' --group='llama2_7B_gen_48k'
# wait
# srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python gen_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='llama2-7B' --path='llama2-7B' --group='llama2_7B_gen_48k'
# wait
srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python gen_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='48k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='base_10000' --path='rope_inv_2d_raw' --group='llama2_7B_gen_48k'