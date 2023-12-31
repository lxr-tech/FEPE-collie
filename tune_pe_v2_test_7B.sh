srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[181-184,193-196] python tune_pe_v2.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='dynamic_new' --ntk_alpha=1 \
 --tag='llama2_7B' --path='llama2-7B' --group='test_dynamic_new0' --pp_size=4
wait 
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[181-184,193-196] python tune_pe_v2.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='dynamic_new' --ntk_alpha=1 \
 --tag='llama2_7B-log' --path='llama2-7B' --group='test_dynamic_new0' --pp_size=4
# wait 
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-173,178-179,194-196] python tune_pe_v2.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='llama2_7B-log' --path='llama2-7B' --group='test_dynamic' --pp_size=4