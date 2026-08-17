#!/bin/bash
# 8对基准统一基线: 用 512 tokens 重跑无 RAG, 与旧 400-token 基线对比
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] start: v4 no-rag 8"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --no-rag --tag unify_v4_norag > logs/unify_v4_norag.log 2>&1

echo "[$(date)] start: base no-rag 8"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --no-rag --tag unify_base_norag > logs/unify_base_norag.log 2>&1

echo "[$(date)] UNIFY DONE" > logs/unify_done.txt