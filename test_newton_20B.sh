# srun -p llm_t --ntasks=96 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[179-186,193-195] python test_newton.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='newton_20B' --path='llama2-7B' --group='jiaofu'
# wait
srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[157-159,177-186,193-195] python test_newton.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
 --tag='newton_20B' --path='llama2-7B' --group='jiaofu' --pp_size=4
# wait
# srun -p llm_t --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[157-159,177-186,193-195] python test_newton.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
#  --tag='newton_20B_dynamic' --path='llama2-7B' --group='jiaofu' --pp_size=4