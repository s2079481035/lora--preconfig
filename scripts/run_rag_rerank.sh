#!/bin/bash
# Reranker 评估（串行 4 组），完成写 done 标记
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] start: base_k3_rerank"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --retrieval rerank --tag base_k3 > logs/rag_rerank_base_k3.log 2>&1

echo "[$(date)] start: base_k5_rerank"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 5 --retrieval rerank --tag base_k5 > logs/rag_rerank_base_k5.log 2>&1

echo "[$(date)] start: v4_k3_rerank"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --retrieval rerank --tag v4_k3 > logs/rag_rerank_v4_k3.log 2>&1

echo "[$(date)] start: v4_k5_rerank"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 5 --retrieval rerank --tag v4_k5 > logs/rag_rerank_v4_k5.log 2>&1

echo "[$(date)] RERANK ALL DONE" > logs/rag_rerank_done.txt