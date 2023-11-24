# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='llama2_7B_b500000_fp' --group='llama2_7B_b500000_fp' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='llama2_7B_b500000_fp' --group='llama2_7B_b500000_fp' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='qianxuesen_7B' --group='qianxuesen_7B' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='qianxuesen_7B' --group='qianxuesen_7B' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[171-174] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500_b10000_fp' --group='shuxingbei_7B_b500_b10000_fp' --ckpt=26000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500_b10000_fp' --group='shuxingbei_7B_b500_b10000_fp' --ckpt=26000
# wait
# srun -p llm_o --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='256k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 --pp_size=16 \
#  --tag='books3_256k' --path='shuxingbei_7B_b500_b10000_fp' --group='shuxingbei_7B_b500_b10000_fp' --ckpt=26000
# wait
# srun -p llm_o --ntasks=128 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='256k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 --pp_size=16 \
#  --tag='log-books3_256k' --path='shuxingbei_7B_b500_b10000_fp' --group='shuxingbei_7B_b500_b10000_fp' --ckpt=26000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='llama2_7B_b2000000_fp' --group='llama2_7B_b2000000_fp' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='llama2_7B_b2000000_fp' --group='llama2_7B_b2000000_fp' --ckpt=28000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_llama2_7B_b2000000_fp' --group='shuxingbei_llama2_7B_b2000000_fp' --ckpt=48000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_llama2_7B_b2000000_fp' --group='shuxingbei_llama2_7B_b2000000_fp' --ckpt=48000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=64000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=64000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=80000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=80000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=100000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184-187] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=100000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[178-179,181-182] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500-ckpt50000_b10000_fp' --group='shuxingbei_7B_b500-ckpt50000_b10000_fp' --ckpt=40000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[178-179,181-182] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500-ckpt50000_b10000_fp' --group='shuxingbei_7B_b500-ckpt50000_b10000_fp' --ckpt=40000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[178-179,181-182] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500-ckpt80000_b10000_fp' --group='shuxingbei_7B_b500-ckpt80000_b10000_fp' --ckpt=40000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=spot \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[178-179,181-182] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500-ckpt80000_b10000_fp' --group='shuxingbei_7B_b500-ckpt80000_b10000_fp' --ckpt=40000
# wait
# srun -p llm_o --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=160000
# wait
# srun -p llm_o --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_7B_b500' --group='shuxingbei_7B_b500' --ckpt=160000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[180-183] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_llama2_7B_b2000000_fp2' --group='shuxingbei_llama2_7B_b2000000_fp2' --ckpt=10000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[180-183] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_llama2_7B_b2000000_fp2' --group='shuxingbei_llama2_7B_b2000000_fp2' --ckpt=10000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[180-183] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=16384 --exp_base=16384 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_llama2_7B_b2000000_fp2_16K' --group='shuxingbei_llama2_7B_b2000000_fp2_16K' --ckpt=10000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[180-183] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=16384 --exp_base=16384 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_llama2_7B_b2000000_fp2_16K' --group='shuxingbei_llama2_7B_b2000000_fp2_16K' --ckpt=10000
