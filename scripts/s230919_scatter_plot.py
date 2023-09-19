import json

import torch
import numpy as np

import pylab
import seaborn
import matplotlib.pyplot as plt

# pylab.rcParams['font.sans-serif'] = ['Times New Roman']
# pylab.rcParams['axes.unicode_minus'] = False

def get_cmap(n, name='hsv'):  # https://stackoverflow.com/a/25628397
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)  # 

with open('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/data_2309_llama2_7B_base.json', 'r') as file1:
    file1 = file1.read()
    file1 = json.loads(file1)
with open('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/data_2309_llama2_7B_base2.json', 'r') as file2:
    file2 = file2.read()
    file2 = json.loads(file2)
file1.update(file2)

item_lst = ['qk1_a_layer31_avg', 'qk1_a_layer31_std', 
            'qk1_b_layer31_avg', 'qk1_b_layer31_std', 
            'qk1_o_layer31_avg', 'qk1_o_layer31_std', 
            'q1k_a_layer31_avg', 'q1k_a_layer31_std', 
            'q1k_b_layer31_avg', 'q1k_b_layer31_std', 
            'q1k_o_layer31_avg', 'q1k_o_layer31_std']

seq_len, num_head, head_dim = 102400, 32, 128  # for llama2-7B

x = torch.tensor(range(seq_len))
c = ['blue', 'red', 'dark orange']
d = [0, 50, 75, 90, 92, 108, 126]
# m = get_cmap(len(d), name='brg')

fig = plt.figure(figsize=(25, 20), dpi=200)

label_dct = {'llama2_7B-qk_100k--.pkl': ('llama2_7B-books3_100k', 'LLaMA2 7B', [4096], 10000),
             'llama2_7B-qk_100k-dynamic.pkl': ('llama2_7B-books3_100k-dynamic_ntk', 'Dynamic NTK', [4096], 10000), 
             'llama2_7B-qk_100k-hang_1000000.pkl': ('llama2_7B-books3_100k-1000000', 'base=1000000', [4096, 102400], 1000000), 
             'llama2_7B-qk_100k-hang_320000.pkl': ('llama2_7B-books3_100k-320000', 'base=320000', [4096, 56076], 320000), 
             'llama2_7B-qk_100k-hang_80000.pkl': ('llama2_7B-books3_100k-80000', 'base=80000', [4096, 19803], 80000), 
             'llama2_7B-qk_100k-hang_10000.pkl': ('llama2_7B-books3_100k-10000', 'base=10000 (ft)', [4096], 10000), 
             'llama2_7B-qk_100k-hang_2608.pkl': ('llama2_7B-books3_100k-2608', 'base=2608', [4096], 2608), 
             'llama2_7B-qk_100k-hang_500.pkl': ('llama2_7B-books3_100k-500', 'base=500', [4096], 500), }

for col, label in enumerate(list(label_dct)):
    
    if col == 1:
        continue
    
    json_dct = torch.load(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/{label}')
    
    print(label)
    
    item, title, poss, base = label_dct[label]

    ax1 = fig.add_subplot(4, 4, col * 2 + 1)
    color = 'tab:brown'  # https://matplotlib.org/3.4.3/gallery/color/color_demo.html

    for i in range(4):
        ax1.scatter(x[0::40], json_dct['qk1_a_layer31_avg'][0::40,i], s=1, c=seaborn.xkcd_rgb[c[0]])
        s1 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[0]])
        ax1.scatter(x[13::40], json_dct['qk1_b_layer31_avg'][13::40,i], s=1, c=seaborn.xkcd_rgb[c[1]])
        s2 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[1]])
        ax1.scatter(x[26::40], json_dct['qk1_o_layer31_avg'][26::40,i], s=1, c=seaborn.xkcd_rgb[c[2]], zorder=0)
        s3 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[2]])
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend([s1, s2, s3], ['first 92 dim', 'last 36 dim', 'overall dim', ], loc='upper right', fontsize=12, labelcolor=color)
    ax1.set_ylabel('Attention score', fontsize=16, color=color)
    ax1.set_ylim((-6.5, 6.5) if col < 2 else (-17.5, 17.5))
    ax1.set_title(r'mean of $q_t^Tk_1$ for %s' % title, fontsize=18)
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    
    ax2.plot(x[1:], file1[item]['cum#ppl'], c=seaborn.xkcd_rgb['purple'], lw=2, ls='-')
    for pos in poss:
        if pos < 102400:
            plt.axvline(pos, 0, 1, c=seaborn.xkcd_rgb['black'], lw=2, ls='-.')
    ax2.set_ylim((-2.1, 102.1))
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Perplexity', fontsize=16, color=color)

    ax3 = fig.add_subplot(4, 4, col * 2 + 2)
    color = 'tab:brown'  # https://matplotlib.org/3.4.3/gallery/color/color_demo.html
    pos = int(min(poss[-1] * 1.5, 102400))

    for i, dim in enumerate(d):
        period = base ** (dim / head_dim) * np.pi * 2
        xd = np.linspace(0, min(period, pos), 1000)
        ax3.plot(xd, np.cos(base ** (-dim / head_dim) * xd) + 2.2 * i, c=seaborn.xkcd_rgb['black'], lw=1, ls='-')
    ax3.set_ylabel('Cos wave', fontsize=16, color=color)
    ax3.set_title(r'cos of different $\theta$ for %s' % title, fontsize=18)
    
    ax2 = ax3.twinx()
    color = 'tab:purple'
    
    ax2.plot(x[1:pos], file1[item]['cum#ppl'][:pos-1], c=seaborn.xkcd_rgb['purple'], lw=2, ls='-', zorder=0)
    for pos in poss:
        if pos < 102400:
            plt.axvline(pos, 0, 1, c=seaborn.xkcd_rgb['black'], lw=2, ls='-.')
    ax2.set_ylim((-2.1, 102.1))
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Perplexity', fontsize=16, color=color)

plt.tight_layout()
plt.subplots_adjust(wspace=0.36, hspace=0.22)
plt.savefig('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/scaling_rope-dim++.jpg')