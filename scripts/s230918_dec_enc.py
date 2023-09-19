import torch
import sentencepiece as spm

dec = '/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model'
enc = '/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model'

dec_sp = spm.SentencePieceProcessor()
dec_sp.load(dec)
enc_sp = spm.SentencePieceProcessor()
enc_sp.load(enc)

enc_data = []
dec_data = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-102400.pkl')
dec_data = dec_data['102400']

max_len = 48 * 1024

for data in dec_data:
    data = data['input_ids']
    data = dec_sp.decode_ids(data)
    data = enc_sp.encode(data)
    if len(data) >= max_len:
        data = {'input_ids': data[:max_len], 'labels': data[:max_len], }
        enc_data.append(data)

torch.save(enc_data, f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-V7-{max_len}.pkl')