#!/bin/bash
# 60 对扩展翻译测试集评估: v4/base × no-RAG/RAG k3
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
P="data/external/translation_benchmark/translate_pairs_60.json"
R="data/rag/translate60_retrieval.json"

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --no-rag --pairs-file "$P" --tag t60_v4_norag > logs/t60_v4_norag.log 2>&1

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --pairs-file "$P" --retrieval-file "$R" --tag t60_v4_ragk3 > logs/t60_v4_ragk3.log 2>&1

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --no-rag --pairs-file "$P" --tag t60_base_norag > logs/t60_base_norag.log 2>&1

venv/bin/python scripts/22_eval_benchmark_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --pairs-file "$P" --retrieval-file "$R" --tag t60_base_ragk3 > logs/t60_base_ragk3.log 2>&1

echo "[$(date)] T60 DONE" > logs/t60_done.txt