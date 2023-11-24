# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-60-[4-5,11,21] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_652' --path='base_652' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-60-[1-46] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000' --path='base_1000000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-60-[1-46] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000' --path='base_10000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --exclude=HOST-10-140-60-[1-46] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500' --path='base_500' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1304' --path='base_1304' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2608' --path='base_2608' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=100 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_100' --path='base_100' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_1000000_16k' --path='base_1000000_16k' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_10000_16k' --path='base_10000_16k' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_16k' --path='base_500_16k' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=100 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_100_16k' --path='base_100_16k' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_40000' --path='base_40000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=120000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_120000' --path='base_120000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[184,187,193-194] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_600000' --path='base_600000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[134,181,186,195] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_500_bf32' --path='base_500_bf32' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2000000' --path='base_2000000' --group='scaling_rope-llama2_7B'
# wait
# srun -p llm2_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[193-196] python tune_pe_v2.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pajama' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='base_2000000_16k' --path='base_2000000_16k' --group='scaling_rope-llama2_7B'

# squeue | grep liuxiao | awk '{print $1}' | xargs scancel
