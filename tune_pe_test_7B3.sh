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
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[177-180] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=34565 --exp_base=34565 \
#  --base=160000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_160000_new_dynamic_xpos_log' --path='hang_160000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[153-155,157] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=34565 --exp_base=34565 \
#  --base=160000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_160000_new_dynamic' --path='hang_160000' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='hang_10000_92_dynamic' --path='hang_10000_92' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='xpos' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='hang_10000_92_dynamic_xpos_log' --path='hang_10000_92' --group='llama2_7B_100k' --pp_size=4
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000' --path='hang_600000' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=800000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_800000' --path='hang_800000' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000' --path='hang_1000000' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='hang_500' --group='llama2_7B_128k' --pp_size=2
# wait
# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=100 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_100' --path='hang_100' --group='llama2_7B_128k' --pp_size=2
wait
# srun -p llm_t --ntasks=48 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[177-180,195-196] python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='128k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=109907 --exp_base=109907 \
#  --base=800000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='base_800000_new_dynamic' --path='hang_800000' --group='llama2_7B_128k' --pp_size=6