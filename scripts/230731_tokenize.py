import sentencepiece as spm

model_paths = [
    '/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/llama-ar.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/v10.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/v6_tokenizer/spiece.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/V7_sft.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/v6_standard.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/V7.model',
    # '/mnt/petrelfs/share_data/yanhang/tokenizes/v9.model'
]

for model_path in model_paths:
    
    sp = spm.SentencePieceProcessor()
    print(type(sp), model_path)
    
    sp.load(model_path)

    # # encode: text => id
    # print(sp.encode_as_pieces('This is a test'))
    # print(sp.encode_as_ids('This is a test'))

    # decode: id => text
    print(sp.decode_ids([1, 259, 13, 2277, 2866, 1237, 13, 13, 1068, 29907, 957, 1068, 13, 13, 1068, 28173, 278, 6726, 1068, 13, 13, 1068, 28173, 278, 13361, 1068, 13, 13, 1068, 7030, 9305, 1068, 13, 13, 1068, 29928, 7486, 362, 1068, 13, 13, 1068, 1184, 1188, 434, 1068, 13, 13, 1068, 29896, 1068, 29871, 29896, 29929, 29929, 29929, 13, 13, 1068, 29906, 1068, 1724, 887, 2823, 338, 1724, 887, 3617, 13, 13, 1068, 29941, 1068, 17166, 263, 15197, 515, 278, 7437, 310, 263, 1522, 261, 5345, 13, 13, 1068, 29946, 1068, 2803, 29915, 29879, 3617, 11661, 936, 13, 13, 1068, 29945, 1068]))

