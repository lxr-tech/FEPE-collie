srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='hang_500000' --path='hang_500000' --group='pjlab_fepe_llama2_70B_4096'
wait
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='hang_1000000' --path='hang_1000000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_8000' --path='hang_8000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_6000' --path='hang_6000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_4000' --path='hang_4000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_2000' --path='hang_2000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=8192 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
#  --tag='rope_inv_2d_raw_pi_2' --path='rope_inv_2d_raw_pi_2' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1000000' --path='hang_1000000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1000000_16k' --path='hang_1000000_16k' --group='pjlab_fepe_llama2_70B_4096'
# # wait
# # srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
# #  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
# #  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
# #  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
# #  --base=100 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
# #  --tag='hang_100' --path='hang_100' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_40000' --path='hang_40000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_80000' --path='hang_80000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=120000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_120000' --path='hang_120000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_160000' --path='hang_160000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=20000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_20000' --path='hang_20000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_500000' --path='hang_500000' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=100 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_100' --path='hang_100' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_16k' --path='hang_10000_16K' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=20000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_20000_16k' --path='hang_20000_16K' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_40000_16k' --path='hang_40000_16K' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_80000_16k' --path='hang_80000_16K' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_92' --path='hang_10000_92' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-185,187,193] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=120000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_120000_16k' --path='hang_120000_16K' --group='pjlab_fepe_llama2_70B_4096'
wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=4 --ntk_option='none' --ntk_alpha=1 \
#  --tag='rope_inv_2d_raw_pi_4' --path='rope_inv_2d_raw_pi_4' --group='pjlab_fepe_llama2_70B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[182-183,185-186] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-70B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_500_16k' --path='hang_500_16K' --group='pjlab_fepe_llama2_70B_4096'
