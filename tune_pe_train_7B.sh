
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-60-[136-139] python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 \
 --tag='rope_inv_2d_raw_500' --group='pjlab_fepe_llama2_7B_4096'
