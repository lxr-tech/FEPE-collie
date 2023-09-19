import torch
import pylab
import seaborn
import matplotlib.pyplot as plt

# pylab.rcParams['font.sans-serif'] = ['Times New Roman']
# pylab.rcParams['axes.unicode_minus'] = False

# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)

item_lst = ['qk1_a_layer31_avg', 'qk1_a_layer31_std', 
            'qk1_b_layer31_avg', 'qk1_b_layer31_std', 
            'qk1_o_layer31_avg', 'qk1_o_layer31_std', 
            'q1k_a_layer31_avg', 'q1k_a_layer31_std', 
            'q1k_b_layer31_avg', 'q1k_b_layer31_std', 
            'q1k_o_layer31_avg', 'q1k_o_layer31_std']

seq_len, num_head = 102400, 32  # for llama2-7B

x = torch.tensor(range(seq_len))
# c = get_cmap(num_head)

fig = plt.figure(figsize=(22, 30), dpi=200)

label_dct = {'llama2_7B-qk_100k--.pkl': ('base=10000', [4096]),
             'llama2_7B-qk_100k-hang_10000.pkl': ('base=10000', [4096]), 
             'llama2_7B-qk_100k-hang_1000000.pkl': ('base=1000000', [4096]), 
             'llama2_7B-qk_100k-hang_160000.pkl': ('base=160000', [4096, 33564]), 
             'llama2_7B-qk_100k-hang_40000.pkl': ('base=40000', [4096, 11902]), 
             'llama2_7B-qk_100k-hang_500.pkl': ('base=500', [4096]), }

title_dct = {('qk1', 'avg'): r'mean of $q_t^Tk_1$', 
             ('qk1', 'std'): r'std of $q_t^Tk_1$', 
             ('q1k', 'avg'): r'mean of $q_T^Tk_s$', 
             ('q1k', 'std'): r'std of $q_T^Tk_s$'}

for col, label in [(0, 'llama2_7B-qk_100k--.pkl'), (1, 'llama2_7B-qk_100k-hang_10000.pkl'), 
                   (2, 'llama2_7B-qk_100k-hang_1000000.pkl'), (3, 'llama2_7B-qk_100k-hang_160000.pkl'), 
                   (4, 'llama2_7B-qk_100k-hang_40000.pkl'), (5, 'llama2_7B-qk_100k-hang_500.pkl'), ]:
    
    json_dct = torch.load(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/{label}')
    
    print(label)
    
    for row, pre, suf in [(0, 'qk1', 'avg'), (1, 'qk1', 'std'), (2, 'q1k', 'avg'), (3, 'q1k', 'std'), ]:

        print(label, pre, suf)

        ax = fig.add_subplot(6, 4, col * 4 + row + 1)
        for i in range(4):
            ax.scatter(x, json_dct[f'{pre}_a_layer31_{suf}'][:,i], s=0.01, c='b')  # c(i))
            ax.scatter(x, json_dct[f'{pre}_b_layer31_{suf}'][:,i], s=0.01, c='r')  # c(i))
            ax.scatter(x, json_dct[f'{pre}_o_layer31_{suf}'][:,i], s=0.01, c=seaborn.xkcd_rgb['violet'])  # c(i))
        base, poss = label_dct[label]
        for pos in poss:
            plt.axvline(pos, 0, 1, c='k', ls='-.')
        ax.set_title(f'{title_dct[(pre, suf)]} for {base}', fontsize=20)

plt.tight_layout()
# plt.subplots_adjust(wspace=0.25)
plt.savefig('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/scaling_rope-dim-1.jpg')