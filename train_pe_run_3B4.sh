
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty -w SH-IDC1-10-140-1-[79,80,82,88] --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_2d_log_v1.0.4' --group='pjlab_fepe_3B2_2048'

# -w SH-IDC1-10-140-0-237,SH-IDC1-10-140-0-238,SH-IDC1-10-140-0-239,SH-IDC1-10-140-0-240