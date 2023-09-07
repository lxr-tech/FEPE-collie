# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='rope_inv_2d_raw' --path='rope_inv_2d_raw' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=8000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_8000' --path='hang_8000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=6000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_6000' --path='hang_6000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=4000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_4000' --path='hang_4000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_2000' --path='hang_2000' --group='pjlab_fepe_llama2_13B_4096'
wait
srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[113-116] python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-13B' --max_length=8192 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=2 --ntk_option='none' --ntk_alpha=1 \
 --tag='rope_inv_2d_raw_pi_2' --path='rope_inv_2d_raw_pi_2' --group='pjlab_fepe_llama2_13B_4096'