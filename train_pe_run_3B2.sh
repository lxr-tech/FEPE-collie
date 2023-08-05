
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='inv' --ln='raw' --tag='xpos_inv_2d_raw' --group='pjlab_fepe_3B_2048'

#  -w SH-IDC1-10-140-1-82,SH-IDC1-10-140-1-86,SH-IDC1-10-140-1-88,SH-IDC1-10-140-1-90 