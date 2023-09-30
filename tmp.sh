# srun -p llm_t --ntasks=8 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-180 python test_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='first92' --path='llama2-7B' --group='llama2_7B-qk_100k'
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[181-182] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_xpos_log' --path='hang_500' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[141-142] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k_xpos_log' --path='hang_500_16k' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[141-142] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k_log' --path='hang_500_16k' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[139-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=16384 --exp_base=16384 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k' --path='hang_500_16k' --group='llama2_7B_256k' --pp_size=16
# wait
# srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[139-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=16384 --exp_base=16384 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000_16k' --path='hang_1000000_16k' --group='llama2_7B_256k' --pp_size=16
# wait
# srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[139-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=16384 --exp_base=16384 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k_log' --path='hang_500_16k' --group='llama2_7B_256k' --pp_size=16
# wait
# # srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[139-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=16384 --exp_base=16384 \
# #  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_500_16k_xpos_log' --path='hang_500_16k' --group='llama2_7B_256k' --pp_size=16
# # wait
# # srun -p llm_t --ntasks=160 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[135-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='256k' \
# #  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=16384 --exp_base=16384 \
# #  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_500_16k' --path='hang_500_16K' --group='llama2_13B_256k' --pp_size=20
# # wait
# # srun -p llm_t --ntasks=160 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[135-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='256k' \
# #  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=16384 --exp_base=16384 \
# #  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_1000000_16k' --path='hang_1000000_16k' --group='llama2_13B_256k' --pp_size=20
# # wait
# # srun -p llm_t --ntasks=160 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[135-146,150-157] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='256k' \
# #  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=16384 --exp_base=16384 \
# #  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_500_16k_log' --path='hang_500_16k' --group='llama2_13B_256k' --pp_size=20
