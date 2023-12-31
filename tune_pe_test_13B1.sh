# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_8000_xpos_log' --path='hang_8000' --group='llama2_13B_100k' 
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_8000_log' --path='hang_8000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_6000_xpos_log' --path='hang_6000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_6000_log' --path='hang_6000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_4000_xpos_log' --path='hang_4000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_4000_log' --path='hang_4000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2608_xpos_log' --path='hang_2608' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2608_log' --path='hang_2608' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2000_xpos_log' --path='hang_2000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2000_log' --path='hang_2000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1304_xpos_log' --path='hang_1304' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1304_log' --path='hang_1304' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000_xpos_log' --path='hang_1000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000_log' --path='hang_1000' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_652_xpos_log' --path='hang_652' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_652_log' --path='hang_652' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_xpos_log' --path='hang_500' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_log' --path='hang_500' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_log' --path='rope_inv_2d_raw' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_xpos_log' --path='rope_inv_2d_raw' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='llama2-13B' --group='llama2_13B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='xpos_log' --path='llama2-13B' --group='llama2_13B_100k'
