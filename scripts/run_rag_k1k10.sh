#!/bin/bash
# RAG k=1/k=10 补跑（串行 4 组），全部完成后写 done 标记
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] start: base_k1"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 1 --tag base_k1 > logs/rag_base_k1.log 2>&1

echo "[$(date)] start: base_k10"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 10 --tag base_k10 > logs/rag_base_k10.log 2>&1

echo "[$(date)] start: v4_k1"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 1 --tag v4_k1 > logs/rag_v4_k1.log 2>&1

echo "[$(date)] start: v4_k10"
venv/bin/python scripts/18_evaluate_rag.py --model-path models/qwen-lora-multitask-v4 --k 10 --tag v4_k10 > logs/rag_v4_k10.log 2>&1

echo "[$(date)] ALL DONE" > logs/rag_k1k10_done.txt