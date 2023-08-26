
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=8192 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --pi_lambda=2 \
 --tag='rope_inv_2d_raw_pi_2' --group='pjlab_fepe_llama2_7B_4096'