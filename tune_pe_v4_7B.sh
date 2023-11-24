# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[148,155,159,184-187,195] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[148,155,159,184-187,195] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[148,155,159,184-187,195] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='base2000000_7B' --group='base2000000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[148,155,159,184-187,195] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='base2000000_7B_16k' --group='base2000000_7B_16k-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log4096' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=16384 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log16384' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=16  # -w HOST-10-140-66-[184-187,193-196]
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log4096' --path='base2000000_7B' --group='base2000000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[152,156,158,167-168,171-172,174] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=16384 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log16384' --path='base2000000_7B_16k' --group='base2000000_7B_16k-pile_256k' --pp_size=16  #
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log78399' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log78399' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='fixed' --ntk_alpha=4 \
#  --tag='fixed78399' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='raw' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='fixed' --ntk_alpha=4 \
#  --tag='fixed78399' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=16
# wait
# srun -p llm_o --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='fixed' --ntk_alpha=4 \
#  --tag='fixed78399_log' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=16
# wait
# srun -p llm_o --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1' python tune_pe_v4.py \
#  --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
#  --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='fixed' --ntk_alpha=4 \
#  --tag='fixed78399_log' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k --pp_size=16
# wait
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
 --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
 --exp='rope' --ln='raw' --log_base=78399 --exp_base=4096 \
 --base=500000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='dynamic78399' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=32
wait
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
 --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
 --exp='rope' --ln='raw' --log_base=78399 --exp_base=4096 \
 --base=500000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='dynamic78399' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=32
wait
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
 --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
 --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
 --base=500000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='dynamic78399_log' --path='base500000_7B' --group='base500000_7B-pile_256k' --pp_size=32
wait
srun -p llm3_t --ntasks=64 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187,193-196] python tune_pe_v4.py \
 --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='256k' \
 --exp='rope' --ln='log' --log_base=78399 --exp_base=4096 \
 --base=500000 --pi_lambda=1 --ntk_option='dynamic' --ntk_alpha=1 \
 --tag='dynamic78399_log' --path='base500000_7B_16k' --group='base500000_7B_16k-pile_256k' --pp_size=32
