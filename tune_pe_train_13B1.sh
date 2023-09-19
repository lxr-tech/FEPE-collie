# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=500 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_500' --path='hang_500' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=652 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_652' --path='hang_652' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1000' --path='hang_1000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=1304 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_1304' --path='hang_1304' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[126-129] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=2608 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_2608' --path='hang_2608' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=200000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_200000' --path='hang_200000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=320000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_320000' --path='hang_320000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[105-108] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=400000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_400000' --path='hang_400000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[166-169] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=240000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_240000' --path='hang_240000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[166-169] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=600000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_600000' --path='hang_600000' --group='pjlab_fepe_llama2_13B_4096'
# wait
# srun -p llm_t --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 -w HOST-10-140-66-[166-169] python tune_pe.py --task_a='finetune' --task_b='training' \
#  --model_size='llama2-13B' --max_length=4096 --dataset='pile' --ext_length='100k' \
#  --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
#  --base=800000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1 \
#  --tag='hang_800000' --path='hang_800000' --group='pjlab_fepe_llama2_13B_4096'
