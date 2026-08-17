#!/bin/bash
# 未见配置 Gen/Comp/Ana 评估: v4 no-RAG vs RAG k3
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

for i in $(seq 1 240); do
    [ -f logs/t60_done.txt ] && break
    sleep 30
done
echo "[$(date)] t60 done, starting unseen multitask eval"

venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --no-rag --data data/processed/unseen_multitask.json --tag unseen_v4_norag > logs/unseen_v4_norag.log 2>&1

venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --data data/processed/unseen_multitask.json --retrieval-file data/rag/unseen_multitask_retrieval.json --tag unseen_v4_ragk3 > logs/unseen_v4_ragk3.log 2>&1

echo "[$(date)] UNSEEN DONE" > logs/unseen_done.txt