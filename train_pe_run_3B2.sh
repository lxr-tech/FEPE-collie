
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty -w SH-IDC1-10-140-1-6,SH-IDC1-10-140-1-7,SH-IDC1-10-140-1-8,SH-IDC1-10-140-1-9 \
 --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' --tag='xpos_inv_2d_raw' --group='pjlab_fepe_3B3_2048'
