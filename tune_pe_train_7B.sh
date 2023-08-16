
srun -p p4_test --ntasks=32 --ntasks-per-node=8 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 python tune_pe.py --task_a='finetune' --task_b='training' \
 --model_size='llama2-7B' --max_length=4096 --dim='1d' --exp='xpos' --imp='imp' --ln='log' --log_base=4096 \
 --tag='xpos_imp_1d_log' --group='pjlab_fepe_llama2_7B_4096'
