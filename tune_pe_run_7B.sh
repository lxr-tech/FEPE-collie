
# srun -p llm --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
#  --kill-on-bad-exit=1 --pty python tune_pe.py --task='finetune' \
#  --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
#  --tag='rope_inv_2d_raw_ntk_dynamic' --group='pjlab_fepe_llama2_7B_4096' --ntk_option='dynamic' --ntk_alpha=16

srun -p llm --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 --pty python tune_pe.py --task='finetune' \
 --model_size='llama-7B' --max_length=2048 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=2048 \
 --tag='rope_inv_2d_raw' --group='pjlab_fepe_llama_7B_2048'

#  -w SH-IDC1-10-140-1-[22,172,176,177]
