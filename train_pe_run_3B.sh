
srun -p llm --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 --quotatype=reserved \
 --pty --kill-on-bad-exit=1 python train_pe.py --model_size='3B' --max_length=2048 \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --tag='rope_inv_2d_raw' --group='pjlab_fepe_3B_2048'

# srun -p llm --quotatype=spot -w SH-IDC1-10-140-0-181 --pty bash
# ps -ef | grep /mnt/petrelfs/liuxiaoran | grep -v 'grep' | awk '{print $2}' | xargs kill -9
# wait
# srun -p llm --quotatype=spot -w SH-IDC1-10-140-0-185 --pty bash
# ps -ef | grep /mnt/petrelfs/liuxiaoran | grep -v 'grep' | awk '{print $2}' | xargs kill -9
# wait
# srun -p llm --quotatype=spot -w SH-IDC1-10-140-0-186 --pty bash
# ps -ef | grep /mnt/petrelfs/liuxiaoran | grep -v 'grep' | awk '{print $2}' | xargs kill -9
# wait
# srun -p llm --quotatype=spot -w SH-IDC1-10-140-0-203 --pty bash
# ps -ef | grep /mnt/petrelfs/liuxiaoran | grep -v 'grep' | awk '{print $2}' | xargs kill -9
# wait