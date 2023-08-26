import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
import seaborn


def count_idea(start, end, imag=False):
    t = torch.arange(start, end).reshape((-1, ))
    si1, ci1 = scipy.special.sici(t)
    si2, ci2 = scipy.special.sici(t / 10000)
    if imag:
        idea = 1 / np.log(10000) * (si1 - si2)  # ci1 can be neglected
    else:
        idea = 1 / np.log(10000) * (ci1 - ci2)  # ci1 can be neglected
    return idea


def count_rope(start, end, head_dim, scaled, dtype1=torch.float, dtype2=torch.float):
    t = torch.arange(start, end, device='cuda', dtype=dtype1).reshape((-1, 1))
    i = torch.linspace(1, head_dim // 2, head_dim // 2, device='cuda', dtype=dtype1).reshape((1, -1))
    rope = torch.cos((10000 ** (-2 * i / head_dim) * t), dtype=dtype2)
    rope = rope * ((2 * i + 0.4 * head_dim) / (1.4 * head_dim)) ** (t / 512) if scaled else rope
    return torch.mean(rope, dim=-1).float().detach().cpu().numpy()


c = ['blue', 'dark orange', 'red', 'black', ]

max_lens, head_dims = [0, 512, 1024, 2048, 4096], [128 ]  # , 256

fig = plt.figure(figsize=(20, 5), dpi=300)


for m in range(len(max_lens)-1):
    start, end = max_lens[m], max_lens[m+1]
    x = np.arange(start, end)
    ax = fig.add_subplot(1, 3, m+1)
        
    idea = count_idea(start, end, )
    fp32 = count_rope(start, end, head_dim=128, scaled=False, dtype1=torch.float)
    fpbf = count_rope(start, end, head_dim=128, scaled=False, dtype1=torch.bfloat16, dtype2=torch.float)
    bf16 = count_rope(start, end, head_dim=128, scaled=False, dtype1=torch.bfloat16, dtype2=torch.bfloat16)
    
    # print(f'{max_len} (idea-fp32)^2 :', np.mean((idea-fp32) ** 2, dim=-1))
    # print(f'{max_len} (idea-fp16)^2 :', np.mean((idea-fpbf) ** 2, dim=-1))
    # print(f'{max_len} (idea-bf16)^2 :', np.mean((idea-bf16) ** 2, dim=-1))
    # print(f'{max_len} (fp32-fp16)^2 :', np.mean((fp32-fpbf) ** 2, dim=-1))
    # print(f'{max_len} (fp32-bf16)^2 :', np.mean((fp32-bf16) ** 2, dim=-1))
    
    l1, = ax.plot(x, fp32, c='r', lw=2, ls='-')
    l2, = ax.plot(x, fpbf, c='m', lw=2, ls='-')
    l3, = ax.plot(x, bf16, c='b', lw=2, ls='-')
    l0, = ax.plot(x, idea, c='k', lw=2, ls='--')
    ax.grid()
    ax.legend([l0, l1, l2, l3], ['idea', 'RoPE fp32', 'RoPE bf16 cache', 'RoPE bf16', ], fontsize=12, loc='upper right')  # , framealpha=1
    ax.set_xlabel('relative distance', fontsize=14)
    if m == 0:
        ax.set_ylabel('attention bias expectation', fontsize=14, labelpad=12)
    ax.set_title(f'from {start} to {end}', fontsize=16)

plt.tight_layout()
plt.savefig('extrapolate_upper_bound_b2.jpg')
