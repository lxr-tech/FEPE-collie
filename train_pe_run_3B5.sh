
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty -w SH-IDC1-10-140-1-[79,80,82,88] --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='xpos' --imp='imp' --ln='log' --tag='xpos_imp_2d_log_v1.0.4_16B' --group='pjlab_fepe_3B2_2048'
