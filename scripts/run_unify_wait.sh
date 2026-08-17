#!/bin/bash
# 等 bench40 完成后跑统一基线
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

for i in $(seq 1 240); do
    [ -f logs/b40_done.txt ] && break
    sleep 30
done
echo "[$(date)] bench40 done, starting unify baseline"
bash scripts/run_unify_baseline.sh
