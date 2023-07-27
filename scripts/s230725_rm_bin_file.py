import os

base = '/remote-home/xrliu/projects/FEPE-deepspeed/checkpoints'

keys = ['rope_inv_2d_raw', 'rope_inv_1d_raw', 'rope_inv_2d_log', 'rope_inv_1d_log', 
        'rope_imp_2d_raw', 'rope_imp_1d_raw', 'rope_imp_2d_log', 'rope_imp_1d_log', 
        'xpos_inv_2d_raw', 'xpos_inv_1d_raw', 'xpos_inv_2d_log', 'xpos_inv_1d_log', 
        'xpos_imp_2d_raw', 'xpos_imp_1d_raw', 'xpos_imp_2d_log', 'xpos_imp_1d_log', ]

for key in keys:
    try:
        os.remove('{}/init_pre_{}/train_last/training_args.bin'.format(base, key))
    except FileNotFoundError:
        pass
