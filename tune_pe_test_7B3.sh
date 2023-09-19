# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[150-153] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_1304_dynamic_xpos_log' --path='hang_1304' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[150-153] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_652_dynamic_xpos_log' --path='hang_652' --group='llama2_7B_100k' --pp_size=4
