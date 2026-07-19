#!/bin/bash
TRAIN_PID=103699
LOGDIR="/home/sunjb/preconfig/network-config/preconfig-reproduce/logs"
VENV_PYTHON="/home/sunjb/preconfig/network-config/preconfig-reproduce/venv/bin/python"

echo "[$(date)] Waiting for training (PID $TRAIN_PID) to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Training finished. Starting evaluation..."

env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $VENV_PYTHON scripts/07_evaluate.py \
    --model-path models/qwen-lora-multitask-v2 \
    --data data/processed/test_data_multitask.json \
    > "$LOGDIR/eval_v2.log" 2>&1

echo "[$(date)] Evaluation finished. Results in $LOGDIR/eval_v2.log"
tail -20 "$LOGDIR/eval_v2.log"
