srun -p llm_o --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 --quotatype=reserved \
 --kill-on-bad-exit=1 -w HOST-10-140-66-195 \
 deepspeed --master_addr HOST-10-140-66-195 --master_port 32000 --include localhost:0,1,2,3,4,5,6,7 \
 test_mistral_hf.py --task_a='finetune' --task_b='testing' \
 --model_size='llama2-7B' --max_length=4096 --dataset='pajama' --ext_length='10k' \
 --dim='2d' --exp='rope' --imp='inv' --ln='raw' --log_base=4096 --exp_base=4096 \
 --base=10000 --pi_lambda=1 --ntk_option='none' --ntk_alpha=1  \
 --tag='pretrained' --path='pretrained' --group='test_mistral-books3_10k'
