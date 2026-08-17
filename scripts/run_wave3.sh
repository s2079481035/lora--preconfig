#!/bin/bash
# Wave3: 等 unify 基线完成后, 跑 40 对基准的 BM25 / Rerank 检索消融
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

for i in $(seq 1 360); do
    [ -f logs/unify_done.txt ] && break
    sleep 30
done
echo "[$(date)] unify done, starting bm25/rerank ablation"

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --bench40 --retrieval bm25 --tag b40_v4_ragk3_bm25 > logs/b40_v4_ragk3_bm25.log 2>&1

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --bench40 --retrieval rerank --tag b40_v4_ragk3_rerank > logs/b40_v4_ragk3_rerank.log 2>&1

echo "[$(date)] WAVE3 DONE" > logs/wave3_done.txt