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
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_2608' --path='hang_2608' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1000000' --path='hang_1000000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1000000_16k' --path='hang_1000000_16k' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=100 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_100' --path='hang_100' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_500_16k' --path='hang_500_16k' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=20000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_20000' --path='hang_20000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_40000' --path='hang_40000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_80000' --path='hang_80000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=120000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_120000' --path='hang_120000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=160000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_160000' --path='hang_160000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=300000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_300000' --path='hang_300000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=240000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_240000' --path='hang_240000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2000000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_2000000' --path='hang_2000000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=320000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_320000' --path='hang_320000' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=20000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_20000_16k' --path='hang_20000_16K' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=40000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_40000_16k' --path='hang_40000_16K' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=80000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_80000_16k' --path='hang_80000_16K' --group='pjlab_fepe_llama2_7B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[149-152] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-7B' --max_length=16384 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_10000_16k' --path='hang_10000_16K' --group='pjlab_fepe_llama2_7B_4096'
