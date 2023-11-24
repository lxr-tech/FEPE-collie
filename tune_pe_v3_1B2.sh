# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_1B_b10000' --group='shuxingbei_1B_b10000' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b10000' --group='shuxingbei_1B_b10000' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b10000_log' --group='shuxingbei_1B_b10000_log' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_1B_b2608' --group='shuxingbei_1B_b2608' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b2608' --group='shuxingbei_1B_b2608' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b2608_log' --group='shuxingbei_1B_b2608_log' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='-' --path='shuxingbei_1B_b1304' --group='shuxingbei_1B_b1304' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b1304' --group='shuxingbei_1B_b1304' --ckpt=20000
# wait
# srun -p llm_o --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[147-150] python tune_pe_v3.py --task_a='finetune' --task_b='testing' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='log' --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='log' --path='shuxingbei_1B_b1304_log' --group='shuxingbei_1B_b1304_log' --ckpt=20000
