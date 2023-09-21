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
with open('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/data_2309_llama2_7B_base3.json', 'r') as file3:
    file3 = file3.read()
    file3 = json.loads(file3)

file1.update(file2)
file1.update(file3)

item_lst = ['qk1_a_layer31_avg', 'qk1_a_layer31_std', 
            'qk1_b_layer31_avg', 'qk1_b_layer31_std', 
            'qk1_o_layer31_avg', 'qk1_o_layer31_std', 
            'q1k_a_layer31_avg', 'q1k_a_layer31_std', 
            'q1k_b_layer31_avg', 'q1k_b_layer31_std', 
            'q1k_o_layer31_avg', 'q1k_o_layer31_std']

seq_len, num_head, head_dim = 102400, 32, 128  # for llama2-7B

x = torch.tensor(range(seq_len))
c = ['blue', 'red', 'dark orange']
d = [60, 75, 90, 92, 108, 126]
m = get_cmap(len(d), name='Oranges')

label_dct1 = {'llama2_7B-qk_100k--.pkl': ('llama2_7B-books3_100k', 'LLaMA2 7B', [4096], 10000),
              'llama2_7B-qk_100k-dynamic.pkl': ('llama2_7B-books3_100k-dynamic_ntk', 'Dynamic NTK', [4096], 10000), 
              'llama2_7B-qk_100k-first92.pkl': ('llama2_7B-books3_100k-first92', 'Only 92 Dims', [4096], 10000), }
label_dct2 = {'llama2_7B-qk_100k-hang_1000000.pkl': ('llama2_7B-books3_100k-1000000', 'base=1000000', [4096, 102400], 1000000), 
              'llama2_7B-qk_100k-hang_240000.pkl': ('llama2_7B-books3_100k-240000', 'base=240000', [4096, 45132], 240000), 
              'llama2_7B-qk_100k-hang_40000.pkl': ('llama2_7B-books3_100k-40000', 'base=40000', [4096, 11902], 40000), 
              'llama2_7B-qk_100k-hang_10000.pkl': ('llama2_7B-books3_100k-10000', 'base=10000 (ft)', [4096], 10000), 
              'llama2_7B-qk_100k-hang_2608.pkl': ('llama2_7B-books3_100k-2608', 'base=2608', [4096], 2608), 
              'llama2_7B-qk_100k-hang_500.pkl': ('llama2_7B-books3_100k-500', 'base=500', [4096], 500), }

for size, a, b, label_dct, file_name in [((17, 15), 3, 3, label_dct1, 'pretrained'), 
                                         ((38, 15), 3, 6, label_dct2, 'finetuned')]:

    fig = plt.figure(figsize=size, dpi=200)
    
    for col, label in enumerate(list(label_dct)):
        
        json_dct = torch.load(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/{label}')
        
        print(label)
        
        item, title, poss, base = label_dct[label]
        
        for row in range(2):
            
            max_len = 100 * 1024 if row == 0 else 16 * 1024
            sam_int = 40 if row == 0 else 4

            ax1 = fig.add_subplot(a, b, row * b + col + 1)
            color = 'tab:brown'
            
            for i in range(num_head):
                ax1.scatter(x[0:max_len:sam_int], json_dct['qk1_a_layer31_avg'][0:max_len:sam_int,i], s=1, c=seaborn.xkcd_rgb[c[0]])
                s1 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[0]])
                s = sam_int // 3
                ax1.scatter(x[s:max_len:sam_int], json_dct['qk1_b_layer31_avg'][s:max_len:sam_int,i], s=1, c=seaborn.xkcd_rgb[c[1]])
                s2 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[1]])
                s = sam_int // 3 * 2 
                ax1.scatter(x[s:max_len:sam_int], json_dct['qk1_o_layer31_avg'][s:max_len:sam_int,i], s=1, c=seaborn.xkcd_rgb[c[2]], zorder=0)
                s3 = ax1.scatter([], [], c=seaborn.xkcd_rgb[c[2]])
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend([s1, s2, s3], ['First 92 Dims', 'Last 36 Dims', 'Overall Dims', ], 
                       loc='upper right', fontsize=12, labelcolor=color, framealpha=1).set_zorder(100000)
            ax1.set_ylabel('Attention score', fontsize=16, color=color)
            ax1.set_ylim((-6.5, 6.5) if file_name == 'pretrained' else (-17.5, 17.5))
            ax1.set_title(r'mean of $q_t^Tk_0$ for %s' % title, fontsize=18)
            
            ax2 = ax1.twinx()
            color = seaborn.xkcd_rgb['purple']  # 'tab:purple'
            
            ax2.plot(x[1:max_len], file1[item]['cum#ppl'][:max_len-1], c=seaborn.xkcd_rgb['purple'], lw=2, ls='-')
            for pos in poss:
                if pos < max_len:
                    plt.axvline(pos, 0, 1, c=seaborn.xkcd_rgb['black'], lw=2, ls='-.')
            ax2.set_ylim((-2.1, 102.1))
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylabel('Perplexity', fontsize=16, color=color)

        ax3 = fig.add_subplot(a, b, 2 * b + col + 1)
        color = 'tab:brown'
        max_len = 18 * 1024  # int(min(poss[-1] * 1.5, 102400))

        ls = []
        for i, dim in enumerate(d):
            xd = np.linspace(0, max_len, max_len)
            if title != 'Dynamic NTK':
                freq = base ** (-dim / head_dim)
                yd = np.cos(freq * xd) + 2.5 * i
            else:
                alpha = np.clip(2 ** np.ceil(np.log2(xd / 4096) + 1) - 1, a_min=1, a_max=10000)
                freq = (base * alpha ** (head_dim / (head_dim - 2))) ** (-dim / head_dim)
                yd = np.cos(freq * xd) + 2.5 * i
            l, = ax3.plot(xd, yd, c=m(i+len(d)//2 if i+len(d)//2 < len(d)-1 else 3*len(d)//2-i-1), lw=1, ls='-')
            ls.append(l)
        ax3.legend(ls, [r'$cos(\theta_nt)$ n=%d' % n for n in d], framealpha=1).set_zorder(100000)
        ax3.set_ylabel('Cos wave', fontsize=16, color=color)
        ax3.set_title(r'cos of different $\theta$ in %s' % title, fontsize=18)
        
        ax2 = ax3.twinx()
        color = seaborn.xkcd_rgb['purple']  # 'tab:purple'
        
        ax2.plot(x[1:max_len], file1[item]['cum#ppl'][:max_len-1], c=seaborn.xkcd_rgb['purple'], lw=2, ls='-', zorder=0)
        for pos in poss:
            if pos < max_len:
                plt.axvline(pos, 0, 1, c=seaborn.xkcd_rgb['black'], lw=2, ls='-.')
        ax2.set_ylim((-2.1, 102.1))
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Perplexity', fontsize=16, color=color)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.36, hspace=0.22)
    plt.savefig(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/scripts/scaling_rope-dim-{file_name}.jpg')
    
    plt.clf()