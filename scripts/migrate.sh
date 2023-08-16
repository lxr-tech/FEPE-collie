#!/bin/bash
#
## access key and secret key
## if you are using cluster-S, please use the ak,sk of cluster-S and 
## cluster-S
## export AWS_ACCESS_KEY_ID=NGGJ11M5DVGWDZYHIA0B
## export AWS_SECRET_ACCESS_KEY=SJZ6LD6wCQxhvEtt6IUksbOoiYDc6QuEjvGIBgtm
## cluster-P
## export AWS_ACCESS_KEY_ID=AH2O1UKXBTPLJWW7UK2I
## export AWS_SECRET_ACCESS_KEY=vGnnSeXhByGRHF1JtnSvNCVwHpGEM6yOP79pwKGf

proxy_off

s_ak="NGGJ11M5DVGWDZYHIA0B"
s_sk="SJZ6LD6wCQxhvEtt6IUksbOoiYDc6QuEjvGIBgtm"
p_ak="AH2O1UKXBTPLJWW7UK2I"
p_sk="vGnnSeXhByGRHF1JtnSvNCVwHpGEM6yOP79pwKGf" 

# checkpoints/ caches/ csv_logs/

# src="/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/csv_logs/"
# dst="s3://${p_ak}:${p_sk}@checkpoints_ssd_02.10.135.3.249:80/liuxiaoran/FEPE-collie/csv_logs/"

src="s3://${p_ak}:${p_sk}@checkpoints_ssd_02.10.135.3.249:80/liuxiaoran/FEPE-collie/checkpoints/"
dst="/mnt/petrelfs/liuxiaoran/projects/FEPE-collie/checkpoints/"

# src="s3://${s_ak}:${s_sk}@model_weights.10.140.14.254:80/0331/5177_pj_v4_labelsm_v2_cn_qa_mixv6/4999/"
# dst="s3://${p_ak}:${p_sk}@checkpoints_ssd_02.10.135.3.249:80/0331/5177_pj_v4_labelsm_v2_cn_qa_mixv6/4999/"

# Execute the rclone sync command with the given options
/mnt/petrelfs/share_data/zhangshuo/sensesync --listers=50 --threads=50 cp "$src" "$dst"
echo "Done with $dst"
