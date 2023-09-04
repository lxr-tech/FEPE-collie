# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-66-[19,22-25,62,66,69] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_652' --path='hang_652' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-66-[19-25,62,66,69] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1304' --path='hang_1304' --group='pjlab_fepe_llama2_7B_4096'
# wait
srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='hang_2608' --path='hang_2608' --group='pjlab_fepe_llama2_7B_4096'

# HOST-10-140-66-[19,22-25,62,66,69]