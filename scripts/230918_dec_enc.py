import torch
import sentencepiece as spm

dec = '/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model'
enc = '/mnt/petrelfs/share_data/llm_data/mistral-7b-hf/tokenizer.model'  # '/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model'

dec_len = 100 * 1024
enc_len = 64 * 1024

dec_sp = spm.SentencePieceProcessor()
dec_sp.load(dec)
enc_sp = spm.SentencePieceProcessor()
enc_sp.load(enc)

enc_data = []
dec_data = torch.load(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-llama-{dec_len}.pkl')
dec_data = dec_data[f'{dec_len}']

for data in dec_data:
    data = data['input_ids']
    data = dec_sp.decode_ids(data)
    data = enc_sp.encode(data)
    if len(data) >= enc_len:
        data = {'input_ids': data[:enc_len], 'labels': data[:enc_len], }
        enc_data.append(data)

print(len(enc_data))
torch.save(enc_data, f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-test-mistral-{enc_len}.pkl')