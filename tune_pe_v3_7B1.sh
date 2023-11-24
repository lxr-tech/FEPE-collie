# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='pretrained' --path='scaling_rope-7B_b500_v1-ckpt_s11000' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='scaling_rope-7B_b500_v1-ckpt_s11000' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000' --path='scaling_rope-7B_b500_v1-ckpt_s11000' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='scaling_rope-7B_b500_v1-ckpt_s11000' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_log' --path='base_10000' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_log' --path='base_500' --group='scaling_rope-7B_b500_v1-ckpt_s11000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='pretrained' --path='scaling_rope-7B_b500_v1-ckpt_s18000' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='scaling_rope-7B_b500_v1-ckpt_s18000' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000' --path='scaling_rope-7B_b500_v1-ckpt_s18000' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='scaling_rope-7B_b500_v1-ckpt_s18000' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_log' --path='base_10000' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_log' --path='base_500' --group='scaling_rope-7B_b500_v1-ckpt_s18000'
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[187,194-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000' --path='scaling_rope-7B_b500_v1-ckpt_s18000_ft' --group='scaling_rope-7B_b500_v1-ckpt_s18000_ft'
# waits
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[187,194-196] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_log' --path='scaling_rope-7B_b500_v1-ckpt_s18000_ft' --group='scaling_rope-7B_b500_v1-ckpt_s18000_ft'