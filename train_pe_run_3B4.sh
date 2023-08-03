
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty -w SH-IDC1-10-140-1-20,SH-IDC1-10-140-1-21,SH-IDC1-10-140-1-22,SH-IDC1-10-140-1-23 \
 --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_2d_log' --group='pjlab_fepe_3B3_2048'
