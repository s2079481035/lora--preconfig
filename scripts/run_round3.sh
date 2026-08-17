#!/bin/bash
# Round 3: 未见基准 k 消融 (v4 k=1/5/10) + task-aware (40对) + base 未见 Gen/Comp
set -e
cd /home/sunjb/preconfig/network-config/preconfig-reproduce

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

echo "[$(date)] start: v4 k1 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 1 --bench40 --tag b40_v4_ragk1 > logs/b40_v4_ragk1.log 2>&1

echo "[$(date)] start: v4 k5 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 5 --bench40 --tag b40_v4_ragk5 > logs/b40_v4_ragk5.log 2>&1

echo "[$(date)] start: v4 k10 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 10 --bench40 --tag b40_v4_ragk10 > logs/b40_v4_ragk10.log 2>&1

echo "[$(date)] start: taskaware retrieve 40"
HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/33_benchmark40_retrieve_taskaware.py > logs/b40_taskaware_retrieve.log 2>&1

echo "[$(date)] start: v4 taskaware k3 40"
venv/bin/python scripts/22_eval_benchmark_rag.py --model-path models/qwen-lora-multitask-v4 --k 3 --bench40 --retrieval-file data/rag/benchmark40_retrieval_taskaware.json --tag b40_v4_ragk3_taskaware > logs/b40_v4_ragk3_taskaware.log 2>&1

echo "[$(date)] start: base unseen Gen/Comp/Ana no-rag"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --no-rag --data data/processed/unseen_multitask.json --tag unseen_base_norag > logs/unseen_base_norag.log 2>&1

echo "[$(date)] start: base unseen Gen/Comp/Ana rag k3"
venv/bin/python scripts/18_evaluate_rag.py --model-path Qwen/Qwen2.5-Coder-1.5B-Instruct --is-base --k 3 --data data/processed/unseen_multitask.json --retrieval-file data/rag/unseen_multitask_retrieval.json --tag unseen_base_ragk3 > logs/unseen_base_ragk3.log 2>&1

echo "[$(date)] ROUND3 DONE" > logs/round3_done.txt