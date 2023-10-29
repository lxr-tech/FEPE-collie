import json
import torch
import sentencepiece as spm

dec = '/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model'

dec_sp = spm.SentencePieceProcessor()
dec_sp.load(dec)

tot_data = torch.load('/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/books3-gen-llama-102656.pkl')

for length in [2304, 4352, 16640, 33024, ]:  # 49408, 65792, 82176, 102656, 
    enc_data = {}
    dec_data = tot_data[str(length)]
    print(length)
    for i, data in enumerate(dec_data):
        bgn_data = dec_sp.decode_ids(data['input_ids'][:256])
        end_data = dec_sp.decode_ids(data['input_ids'][-256:])
        enc_data[str(i)] = {'begin': bgn_data, 'end': end_data}
    with open(f'/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/caches/llama2_7B_gen_100k-{length}.json', 'w') as fp:
        json.dump(enc_data, fp, indent=4)

