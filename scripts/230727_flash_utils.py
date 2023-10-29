import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from flash_attn.bert_padding import index_first_axis, unpad_input

torch.manual_seed(0)
repeats = 30
batch_size = 64
nheads = 16
seqlen = 1024
n = 1024
d = n // nheads
dropout_p = 0.1
causal = False
dtype = torch.float16
device = 'cuda'

x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)

lengths = torch.randint(seqlen - 20, seqlen, (batch_size, 1), device='cuda')
lengths[-1][0] = seqlen 
attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) < lengths
attention_mask = attention_mask_bool.int()

x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask_bool)

print("lengths", lengths)
print("x_unpad", x_unpad)
print("indices", indices)
print("cu_seqlens", cu_seqlens)
print("max_seqlen_in_batch", max_seqlen_in_batch)

seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
max_seqlen_in_batch = seqlens_in_batch.max().item()
cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))

x_unpad = index_first_axis(rearrange(x, 'b s ... -> (b s) ...'), indices)

# x_unpad_1 = torch.gather(x.flatten(), 0, indices).reshape(-1,)
# print(x_unpad == x_unpad_1)

print("lengths", lengths)
print("x_unpad", x_unpad)
print("indices", indices)
# print("position", position)

position = torch.arange(max_seqlen_in_batch, device='cuda').reshape((1, -1)) * attention_mask
position = torch.gather(position.flatten(), 0, indices).reshape(-1,)

print("position", position, indices.shape, position.shape, int(position.max()))
print("cu_seqlens", cu_seqlens)
print("max_seqlen_in_batch", max_seqlen_in_batch)

"""
x_unpad tensor([[-0.9248, -0.4253, -2.6445,  ..., -1.2363, -1.1992, -0.1084],
        [-1.0322, -0.8892, -0.1914,  ...,  1.4658, -0.0562,  0.0457],
        [ 1.4951, -1.5723,  0.3313,  ..., -0.6592,  2.3184,  0.4822],
        ...,
        [-0.7598, -0.2815, -1.7314,  ..., -0.5435, -0.4978, -1.1318],
        [-0.4604,  0.5435,  0.4023,  ..., -0.1267, -0.4702,  0.3462],
        [ 0.5518,  0.2021, -0.4985,  ..., -0.2632,  1.6035,  0.0717]],
       device='cuda:0', dtype=torch.float16, grad_fn=<IndexFirstAxisBackward>)
indices tensor([    0,     1,     2,  ..., 65517, 65518, 65519], device='cuda:0')
cu_seqlens tensor([    0,  1014,  2030,  3050,  4073,  5085,  6094,  7102,  8109,  9114,
        10129, 11140, 12159, 13168, 14188, 15206, 16211, 17219, 18224, 19240,
        20249, 21255, 22274, 23296, 24301, 25307, 26317, 27330, 28340, 29362,
        30366, 31379, 32399, 33419, 34432, 35450, 36456, 37467, 38480, 39484,
        40505, 41517, 42540, 43555, 44569, 45579, 46598, 47610, 48618, 49623,
        50629, 51650, 52663, 53673, 54695, 55717, 56732, 57746, 58761, 59766,
        60774, 61792, 62815, 63830, 64838], device='cuda:0', dtype=torch.int32)
max_seqlen_in_batch 1023
"""
