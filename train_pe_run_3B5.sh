
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='1d' --exp='xpos' --imp='inv' --ln='log' --tag='xpos_inv_1d_log' --group='pjlab_fepe_3B_2048'
