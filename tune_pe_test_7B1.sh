# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_8000_xpos_log' --path='hang_8000' --group='llama2_7B_100k' 
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_8000_log' --path='hang_8000' --group='llama2_7B_100k' --pp_size=2
# wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_6000_xpos_log' --path='hang_6000' --group='llama2_7B_100k'
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_6000_log' --path='hang_6000' --group='llama2_7B_100k' --pp_size=2
# wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_4000_xpos_log' --path='hang_4000' --group='llama2_7B_100k'
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_4000_log' --path='hang_4000' --group='llama2_7B_100k' --pp_size=2
# wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_2608_xpos_log' --path='hang_2608' --group='llama2_7B_100k'
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2608_log' --path='hang_2608' --group='llama2_7B_100k' --pp_size=2
# wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_2000_xpos_log' --path='hang_2000' --group='llama2_7B_100k'
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2000_log' --path='hang_2000' --group='llama2_7B_100k' --pp_size=2
# wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
# #  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
# #  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='base_1304_xpos_log' --path='hang_1304' --group='llama2_7B_100k'
# # wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1304_log' --path='hang_1304' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000_xpos_log' --path='hang_1000' --group='llama2_7B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000_log' --path='hang_1000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_652_xpos_log' --path='hang_652' --group='llama2_7B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_652_log' --path='hang_652' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_xpos_log' --path='hang_500' --group='llama2_7B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_log' --path='hang_500' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_log' --path='rope_inv_2d_raw' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_xpos_log' --path='rope_inv_2d_raw' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[86-89] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='llama2-7B' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='xpos_log' --path='llama2-7B' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000_log' --path='hang_1000000' --group='llama2_7B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[186,193-195] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='cut_dim' --path='llama2-7B' --group='llama2_7B_100k'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-156] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000_xpos_log' --path='hang_600000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-156] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000_log' --path='hang_600000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-156]  python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_92' --path='hang_10000_92' --group='llama2_7B-qk_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-156]  python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_92_log' --path='hang_10000_92' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-156] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_92_xpos_log' --path='hang_10000_92' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000_xpos_log' --path='hang_1000000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=400000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_400000_xpos_log' --path='hang_400000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=400000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_400000_log' --path='hang_400000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_160000_xpos_log' --path='hang_160000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_160000_log' --path='hang_160000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_80000_xpos_log' --path='hang_80000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_80000_log' --path='hang_80000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_40000_xpos_log' --path='hang_40000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_40000_log' --path='hang_40000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[185-187,193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=89377 --exp_base=89377 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000_new_xpos_log' --path='hang_600000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=89377 --exp_base=89377 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000_new_log' --path='hang_600000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=66782 --exp_base=66782 \
#  --base=400000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_400000_new_xpos_log' --path='hang_400000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=66782 --exp_base=66782 \
#  --base=400000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_400000_new_log' --path='hang_400000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=34565 --exp_base=34565 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_160000_new_xpos_log' --path='hang_160000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=34565 --exp_base=34565 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_160000_new_log' --path='hang_160000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=12762 --exp_base=12762 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_40000_new_xpos_log' --path='hang_40000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=12762 --exp_base=12762 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_40000_new_log' --path='hang_40000' --group='llama2_7B_100k' --pp_size=2
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=12762 --exp_base=12762 \
#  --base=40000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_40000_new_dynamic_xpos_log' --path='hang_40000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=12762 --exp_base=12762 \
#  --base=40000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_40000_new_dynamic' --path='hang_40000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=12762 --exp_base=12762 \
#  --base=40000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_40000_new_dynamic' --path='hang_40000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-148] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000_16k' --path='hang_1000000_16k' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-148] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k' --path='hang_500_16k' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-148] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_log' --path='hang_500' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=48 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[143-148] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=89377 --exp_base=89377 \
#  --base=600000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_600000_new_dynamic' --path='hang_600000' --group='llama2_7B_128k' --pp_size=6
