#!/bin/bash
# Wave 2: 等待 rerank 第一波完成后, 依次跑
#   1) BM25 评估: base/v4 × k=3/k=5 (4组)
#   2) Rerank 补全: base/v4 × k=1/k=10 (4组)
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# 等待第一波完成 (最大 2 小时)
for i in $(seq 1 120); do
    if [ -f logs/rag_rerank_done.txt ]; then
        echo "[$(date)] wave1 done, starting wave2"
        break
    fi
    sleep 60
done

# BM25 评估 4 组
echo "[$(date)] start: bm25 base_k3"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --retrieval bm25 --tag base_k3 > logs/rag_bm25_base_k3.log 2>&1
echo "[$(date)] start: bm25 base_k5"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 5 --retrieval bm25 --tag base_k5 > logs/rag_bm25_base_k5.log 2>&1
echo "[$(date)] start: bm25 v4_k3"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --retrieval bm25 --tag v4_k3 > logs/rag_bm25_v4_k3.log 2>&1
echo "[$(date)] start: bm25 v4_k5"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 5 --retrieval bm25 --tag v4_k5 > logs/rag_bm25_v4_k5.log 2>&1

# Rerank 补全 k=1/k=10
echo "[$(date)] start: rerank base_k1"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 1 --retrieval rerank --tag base_k1 > logs/rag_rerank_base_k1.log 2>&1
echo "[$(date)] start: rerank base_k10"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 10 --retrieval rerank --tag base_k10 > logs/rag_rerank_base_k10.log 2>&1
echo "[$(date)] start: rerank v4_k1"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 1 --retrieval rerank --tag v4_k1 > logs/rag_rerank_v4_k1.log 2>&1
echo "[$(date)] start: rerank v4_k10"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 10 --retrieval rerank --tag v4_k10 > logs/rag_rerank_v4_k10.log 2>&1

echo "[$(date)] WAVE2 ALL DONE" > logs/rag_wave2_done.txt