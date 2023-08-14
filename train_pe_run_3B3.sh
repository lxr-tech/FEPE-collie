
srun -p llm --ntasks=16 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 -w SH-IDC1-10-140-0-[143,144] --kill-on-bad-exit=1 --pty python train_pe.py \
 --model_size='3B' --max_length=2048 --dim='2d' --exp='xpos' --imp='imp' --ln='log' \
 --tag='xpos_imp_2d_log' --group='pjlab_fepe_3B3_2048'
