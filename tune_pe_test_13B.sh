# srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='code' --ext_length='48k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='ntk_dynamic-code' --path='llama2-13B' --group='llama2_13B_48k'
# wait
srun -p llm_t --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-13B' --max_length=4096 --dataset='leval' --ext_length='48k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='ntk_dynamic-leval' --path='llama2-13B' --group='llama2_13B_48k'
wait
srun -p llm_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='ntk_dynamic-books3' --path='llama2-13B' --group='llama2_13B_100k'
