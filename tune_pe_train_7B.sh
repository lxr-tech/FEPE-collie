
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=8192 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --pi_lambda=2 \
#  --tag='rope_inv_2d_raw_pi_2' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --base=500 \
#  --tag='rope_inv_2d_raw_500' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --base=1000 \
#  --tag='rope_inv_2d_raw_1000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=4 --ntk_option='none' --ntk_alpha=1 \
#  --tag='rope_inv_2d_raw_pi_4' --path='rope_inv_2d_raw_pi_4' --group='pjlab_fepe_llama2_7B_4096'