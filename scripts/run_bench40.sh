#!/bin/bash
# 40对基准评估: 统一 max_new_tokens=512 设置
# 1) v4 no-RAG (40对)
# 2) v4 + RAG k=3 (40对)
# 3) v4 + RAG k=3 + sanitize (40对)
# 4) base no-RAG (40对)
# 5) base + RAG k=3 (40对)
# 6) base + RAG k=3 + sanitize (40对)
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] start: v4 no-rag 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --no-rag --bench40 --tag b40_v4_norag > logs/b40_v4_norag.log 2>&1

echo "[$(date)] start: v4 rag k3 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --bench40 --tag b40_v4_ragk3 > logs/b40_v4_ragk3.log 2>&1

echo "[$(date)] start: v4 rag k3 san 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --sanitize --bench40 --tag b40_v4_ragk3san > logs/b40_v4_ragk3san.log 2>&1

echo "[$(date)] start: base no-rag 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --no-rag --bench40 --tag b40_base_norag > logs/b40_base_norag.log 2>&1

echo "[$(date)] start: base rag k3 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --bench40 --tag b40_base_ragk3 > logs/b40_base_ragk3.log 2>&1

echo "[$(date)] start: base rag k3 san 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --sanitize --bench40 --tag b40_base_ragk3san > logs/b40_base_ragk3san.log 2>&1

echo "[$(date)] BENCH40 ALL DONE" > logs/b40_done.txt