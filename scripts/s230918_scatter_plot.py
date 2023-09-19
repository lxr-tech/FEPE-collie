import json

import torch
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

seq_len, num_head = 102400, 32  # for llama2-7B

x = torch.tensor(range(seq_len))
c = ['blue', 'red', 'dark orange']
# c = get_cmap(3, name='tab10')  # ['#FFB800', '#FF4F00', '#00D2FF', '#7F00FF']

fig = plt.figure(figsize=(25, 20), dpi=200)

label_dct = {'llama2_7B-qk_100k--.pkl': ('llama2_7B-books3_100k', 'LLaMA2 7B', [4096]),
             'llama2_7B-qk_100k-dynamic.pkl': ('llama2_7B-books3_100k-dynamic_ntk', 'Dynamic NTK', [4096]), 
             'llama2_7B-qk_100k-hang_1000000.pkl': ('llama2_7B-books3_100k-1000000', 'base=1000000', [4096]), 
             'llama2_7B-qk_100k-hang_320000.pkl': ('llama2_7B-books3_100k-320000', 'base=320000', [4096, 56076]), 
             'llama2_7B-qk_100k-hang_80000.pkl': ('llama2_7B-books3_100k-80000', 'base=80000', [4096, 19803]), 
             'llama2_7B-qk_100k-hang_10000.pkl': ('llama2_7B-books3_100k-10000', 'base=10000 (ft)', [4096]), 
             'llama2_7B-qk_100k-hang_2608.pkl': ('llama2_7B-books3_100k-2608', 'base=2608', [4096]), 
             'llama2_7B-qk_100k-hang_500.pkl': ('llama2_7B-books3_100k-500', 'base=500', [4096]), }

title_dct = {('qk1', 'avg'): (r'mean of $q_t^Tk_1$', (-17.5, 17.5)), 
             ('qk1', 'std'): (r'std of $q_t^Tk_1$', (-2.5, 22.5)), 
             ('q1k', 'avg'): (r'mean of $q_T^Tk_s$', (-7.5, 7.5)), 
             ('q1k', 'std'): (r'std of $q_T^Tk_s$', (-2.5, 15))}

llama_ylim = [(-6.5, 6.5), (-0.2, 3.2)]

for col, label in [(0, 'llama2_7B-qk_100k--.pkl'), (1, 'llama2_7B-qk_100k-dynamic.pkl'), 
                   (2, 'llama2_7B-qk_100k-hang_1000000.pkl'), (3, 'llama2_7B-qk_100k-hang_320000.pkl'), 
                   (4, 'llama2_7B-qk_100k-hang_80000.pkl'), (5, 'llama2_7B-qk_100k-hang_10000.pkl'), 
                   (6, 'llama2_7B-qk_100k-hang_2608.pkl'), (7, 'llama2_7B-qk_100k-hang_500.pkl'), ]:
    
    json_dct = torch.load(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/{label}')
    
    print(label)
    
    for row, pre, suf in [(0, 'qk1', 'avg'), (1, 'qk1', 'std')]:

        print(label, pre, suf)

        item, base, poss = label_dct[label]
        title, ylim = title_dct[(pre, suf)]

        ax1 = fig.add_subplot(4, 4, col * 2 + row + 1)
        color = 'tab:brown'  # https://matplotlib.org/3.4.3/gallery/color/color_demo.html

        for i in range(4):
            ax1.scatter(x[0::40], json_dct[f'{pre}_a_layer31_{suf}'][0::40,i], s=1, c=seaborn.xkcd_rgb[c[0]])  # c(0)
            s1 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[0]])
            ax1.scatter(x[13::40], json_dct[f'{pre}_b_layer31_{suf}'][13::40,i], s=1, c=seaborn.xkcd_rgb[c[1]])  # c(1)
            s2 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[1]])
            ax1.scatter(x[26::40], json_dct[f'{pre}_o_layer31_{suf}'][26::40,i], s=1, c=seaborn.xkcd_rgb[c[2]], zorder=0)  # c(2)  
            s3 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[2]])
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend([s1, s2, s3], ['first 92 dim', 'last 36 dim', 'overall dim', ], 
                   loc='upper right', fontsize=12, labelcolor=color)
        ax1.set_ylabel('Attention score', fontsize=16, color=color)
        ax1.set_ylim(llama_ylim[row] if col < 2 else ylim)
        ax1.set_title(f'{title} for {base}', fontsize=18)
        
        ax2 = ax1.twinx()
        color = 'tab:purple'
        
        ax2.plot(x[1:], file1[item]['cum#ppl'], c=seaborn.xkcd_rgb['purple'], lw=2, ls='-')
        for pos in poss:
            plt.axvline(pos, 0, 1, c=seaborn.xkcd_rgb['black'], lw=2, ls='-.')
        ax2.set_ylim((-2.1, 102.1))
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Perplexity', fontsize=16, color=color)

        
plt.tight_layout()
plt.subplots_adjust(wspace=0.36, hspace=0.22)
plt.savefig('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/scaling_rope-dim+.jpg')