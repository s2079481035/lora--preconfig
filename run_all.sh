#!/bin/bash
# ═══════════════════════════════════════════════════════════
# PreConfig 复现 - 完整实验流程
# 使用方式: bash run_all.sh
# ═══════════════════════════════════════════════════════════

set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "=========================================="
echo "PreConfig 完整实验流程"
echo "开始时间: $(date)"
echo "=========================================="

# ── Step 0: 下载数据集（HuggingFace / fallback）──
echo ""
echo "[Step 0/7] 下载/准备数据集..."
python3 scripts/00_download_datasets.py --fallback 2>&1 | tee $LOG_DIR/datasets_${TIMESTAMP}.log
echo "数据集已保存到 data/raw/"

# ── Step 1: 生成论文图表 ──
echo ""
echo "[Step 1/7] 生成论文图表..."
python3 scripts/06_reproduce_figures.py 2>&1 | tee $LOG_DIR/figures_${TIMESTAMP}.log
echo "图表已保存到 figures/"

# ── Step 2: 测试 ConfigBLEU ──
echo ""
echo "[Step 2/7] 测试 ConfigBLEU 指标..."
python3 scripts/04_configbleu.py --demo 2>&1 | tee $LOG_DIR/configbleu_${TIMESTAMP}.log

# ── Step 3: 数据爬取（可选，如有网络）──
echo ""
echo "[Step 3/7] 尝试数据爬取（可能因网络限制失败）..."
python3 scripts/01_data_crawler.py --all 2>&1 | tee $LOG_DIR/crawler_${TIMESTAMP}.log || true

# ── Step 4: 数据挖掘 + 构建训练集 ──
echo ""
echo "[Step 4/7] 数据挖掘 + 构建训练集..."
python3 scripts/02_data_mining.py --method all 2>&1 | tee $LOG_DIR/mining_${TIMESTAMP}.log

# ── Step 5: 数据增强（需要 OPENAI_API_KEY）──
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[Step 5/7] 跳过数据增强（未设置 OPENAI_API_KEY）"
    echo "  设置方式: export OPENAI_API_KEY='your-key'"
else
    echo "[Step 5/7] 数据增强..."
    python3 scripts/03_data_augmentation.py \
        --data data/processed/forum_configs_extracted.json \
        --save-prompts 2>&1 | tee $LOG_DIR/augmentation_${TIMESTAMP}.log
fi

# ── Step 6: 模型微调 ──
echo ""
echo "[Step 6/7] 模型微调（保存格式化数据）..."
python3 scripts/05_finetune.py --save-only 2>&1 | tee $LOG_DIR/finetune_${TIMESTAMP}.log

# ── Step 7: 显示结果统计 ──
echo ""
echo "[Step 7/7] 结果统计..."
echo "=========================================="
echo "data/raw/ 中的文件:"
ls -la data/raw/ 2>/dev/null || echo "  (空)"
echo ""
echo "data/processed/ 中的文件:"
ls -la data/processed/ 2>/dev/null || echo "  (空)"
echo ""
echo "figures/ 中的文件:"
ls -la figures/ 2>/dev/null || echo "  (空)"
echo "=========================================="

echo ""
echo "=========================================="
echo "实验流程完成！"
echo "结束时间: $(date)"
echo ""
echo "生成的文件："
echo "  figures/          - 论文图表"
echo "  data/raw/         - 原始数据"
echo "  data/processed/   - 处理后的数据"
echo "  data/augmented/   - 增强后的数据"
echo "  models/           - 格式化的训练数据"
echo "  logs/             - 运行日志"
echo "=========================================="
