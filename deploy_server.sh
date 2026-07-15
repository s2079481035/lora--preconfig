#!/bin/bash
# ═══════════════════════════════════════════════════════════
# PreConfig 复现 - 服务器部署脚本 (RTX 4090, 20GB VRAM)
# ═══════════════════════════════════════════════════════════

set -e

echo "=========================================="
echo "PreConfig 服务器部署 (4090 + 20GB)"
echo "=========================================="

# ── 1. 检查环境 ──
echo "[1/6] 检查环境..."
python3 --version || { echo "ERROR: need Python 3.10+"; exit 1; }
nvidia-smi || echo "WARNING: no NVIDIA driver found"

# ── 2. 创建虚拟环境 ──
echo "[2/6] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip

# ── 3. 安装 PyTorch (CUDA 12.1 for 4090) ──
echo "[3/6] 安装 PyTorch + CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 4. 安装其他依赖 ──
echo "[4/6] 安装项目依赖..."
pip install -r requirements.txt

# 单独安装 datasets（用于 HuggingFace 数据集下载）
pip install datasets>=2.16.0

# 可选: 安装 flash-attn 以加速训练（4090 支持）
pip install flash-attn --no-build-isolation || echo "flash-attn 安装失败（可选）"

# ── 5. 创建目录 ──
echo "[5/6] 创建目录..."
mkdir -p data/{raw,processed,augmented}
mkdir -p models/preconfig-finetuned
mkdir -p figures
mkdir -p prompts
mkdir -p logs

# ── 6. 验证 GPU ──
echo "[6/6] 验证 GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f}GB VRAM')
"

# ── 测试 ConfigBLEU ──
echo ""
echo "运行 ConfigBLEU 测试..."
python3 scripts/04_configbleu.py --demo || echo "ConfigBLEU 测试失败（可忽略）"

echo ""
echo "=========================================="
echo "部署完成！"
echo ""
echo "快速测试命令:"
echo "  bash run_all.sh                  # 运行全流程"
echo ""
echo "分步命令:"
echo "  python scripts/00_download_datasets.py --fallback    # 准备数据"
echo "  python scripts/04_configbleu.py --demo               # 测试评估指标"
echo "  python scripts/06_reproduce_figures.py               # 生成图表"
echo "  python scripts/05_finetune.py                        # 模型微调"
echo ""
echo "训练命令（4090 推荐配置）:"
echo "  python scripts/05_finetune.py \\"
echo "    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\"
echo "    --epochs 3 --batch-size 4 --lr 5e-5"
echo "=========================================="
