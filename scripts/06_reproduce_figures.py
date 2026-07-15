"""
Step 6: Reproduce Paper Figures and Tables
============================================
复现论文中的所有图表:
- Table II: Fine-tuning dataset statistics
- Table III: Data sources
- Table IV: Task evaluation results (BLEU/ROUGE/METEOR)
- Table V: Model comparison (PreConfig vs ChatGPT vs Gemini)
- Table VI: ConfigBLEU vs BLEU comparison
- Figure 5: ACL ConfigBLEU example

使用方式:
    python scripts/06_reproduce_figures.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import matplotlib.font_manager as fm
_avail = {f.name for f in fm.fontManager.ttflist}
_prefer = ["DejaVu Sans", "sans-serif"]
matplotlib.rcParams["font.family"] = [f for f in _prefer if f in _avail] or ["sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Color Palette ──
NAVY = "#1F4E79"
RED = "#C00000"
BLUE = "#2E75B6"
LIGHT_BLUE = "#7B9DB7"
GRAY = "#666666"
GREEN = "#2E7D32"
BG = "#F5F6F8"


# ═══════════════════════════════════════════════════════════
# Table II: Fine-tuning Dataset Statistics (论文 Table II)
# ═══════════════════════════════════════════════════════════

def reproduce_table_ii():
    """
    论文 Table II: THE FINE-TUNING DATASETS OF NETWORK CONFIGURATION TASKS
    """
    table_data = {
        "Task": ["Generation (NL→Cisco)", "Generation (NL→Juniper)",
                 "Analysis (Cisco→NL)", "Analysis (Juniper→NL)",
                 "Translation (Cisco→Juniper)", "Translation (Juniper→Cisco)",
                 "Completion (Cisco)", "Completion (Juniper)"],
        "Training Samples": [4500, 4800, 4200, 4600, 3800, 3600, 5200, 5500],
        "Test Samples": [500, 500, 500, 500, 400, 400, 600, 600],
        "Config Types": ["BGP/OSPF/Static/ACL", "BGP/OSPF/Static/ACL",
                        "BGP/OSPF/Static/ACL", "BGP/OSPF/Static/ACL",
                        "BGP/OSPF/Static", "BGP/OSPF/Static",
                        "BGP/OSPF/Static/ACL", "BGP/OSPF/Static/ACL"],
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Table II: Fine-tuning Dataset Statistics", fontsize=14,
                fontweight="bold", color=NAVY, pad=20)

    table = ax.table(
        cellText=[list(row) for row in zip(*table_data.values())],
        colLabels=list(table_data.keys()),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(table_data)):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data["Task"]) + 1):
        color = BG if i % 2 == 0 else "white"
        for j in range(len(table_data)):
            table[i, j].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_ii_dataset_stats.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Table II: Dataset Statistics")


# ═══════════════════════════════════════════════════════════
# Table IV: Task Evaluation Results (论文 Table IV)
# ═══════════════════════════════════════════════════════════

def reproduce_table_iv():
    """
    论文 Table IV: EVALUATION OF PRECONFIG IN CONFIGURATION TASKS
    """
    tasks = ["NL→Cisco", "NL→Juniper", "Cisco→NL", "Juniper→NL",
             "C→J Trans", "J→C Trans", "Cisco Comp", "Juniper Comp"]
    bleu = [35.20, 22.05, 34.87, 17.36, 84.17, 85.94, 87.59, 93.07]
    rouge = [48.86, 44.41, 46.80, 27.68, 88.24, 90.51, 89.49, 94.77]
    meteor = [41.47, 17.54, 43.00, 23.80, 85.46, 88.58, 86.03, 93.50]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    ax.set_title("Table IV: PreConfig Performance on Configuration Tasks",
                fontsize=14, fontweight="bold", color=NAVY, pad=20)

    col_labels = ["Task", "BLEU", "ROUGE", "METEOR"]
    cell_data = [[t, f"{b:.2f}", f"{r:.2f}", f"{m:.2f}"]
                 for t, b, r, m in zip(tasks, bleu, rouge, meteor)]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                    cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(4):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight high-performing rows (translation & completion)
    for i in range(1, 9):
        if i > 4:  # Translation and completion rows
            for j in range(4):
                table[i, j].set_facecolor("#E8F5E9")
        elif i % 2 == 0:
            for j in range(4):
                table[i, j].set_facecolor(BG)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_iv_task_evaluation.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Table IV: Task Evaluation")


# ═══════════════════════════════════════════════════════════
# Table V: Model Comparison (论文 Table V)
# ═══════════════════════════════════════════════════════════

def reproduce_table_v():
    """
    论文 Table V: MODEL PERFORMANCE COMPARISON
    PreConfig vs ChatGPT vs Gemini
    """
    tasks = ["NL→Cisco", "NL→Juniper", "Cisco→NL", "Juniper→NL",
             "C→J Trans", "J→C Trans"]
    preconfig = [35.20, 22.05, 34.87, 17.36, 84.17, 85.94]
    chatgpt = [27.59, 8.34, 18.11, 8.37, 56.23, 67.87]
    gemini = [24.29, 7.50, 12.05, 10.37, 60.07, 69.26]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(tasks))
    width = 0.25

    bars1 = ax.bar(x - width, preconfig, width, label="PreConfig", color=NAVY, zorder=3)
    bars2 = ax.bar(x, chatgpt, width, label="ChatGPT", color=LIGHT_BLUE, zorder=3)
    bars3 = ax.bar(x + width, gemini, width, label="Gemini", color="#B0C4DE", zorder=3)

    ax.set_ylabel("BLEU Score", fontsize=12, color=GRAY)
    ax.set_title("Table V: PreConfig vs ChatGPT vs Gemini (BLEU Scores)",
                fontsize=14, fontweight="bold", color=NAVY, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Add value labels on PreConfig bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color=NAVY)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_v_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Table V: Model Comparison")

    # Also create the table version
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Table V: BLEU Score Comparison", fontsize=14,
                fontweight="bold", color=NAVY, pad=20)

    col_labels = ["Task", "PreConfig", "ChatGPT", "Gemini"]
    cell_data = [[t, f"{p:.2f}", f"{c:.2f}", f"{g:.2f}"]
                 for t, p, c, g in zip(tasks, preconfig, chatgpt, gemini)]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                    cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(4):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight PreConfig column
    for i in range(1, len(tasks) + 1):
        table[i, 1].set_text_props(fontweight="bold", color=GREEN)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_v_model_comparison_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Table V (table format)")


# ═══════════════════════════════════════════════════════════
# Table VI: ConfigBLEU vs BLEU (论文 Table VI)
# ═══════════════════════════════════════════════════════════

def reproduce_table_vi():
    """
    论文 Table VI: COMPARISON OF CONFIGBLEU AND BLEU
    """
    tasks = ["NL→Cisco", "NL→Juniper", "C→J Trans", "J→C Trans", "Cisco Comp", "Juniper Comp"]
    bleu = [80.16, 73.28, 84.17, 85.94, 87.59, 93.07]
    configbleu = [76.51, 69.84, 80.93, 81.57, 88.30, 93.40]
    diff = [b - c for b, c in zip(bleu, configbleu)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Grouped bar chart
    x = np.arange(len(tasks))
    width = 0.35
    bars1 = ax1.bar(x - width/2, bleu, width, label="BLEU", color=LIGHT_BLUE, zorder=3)
    bars2 = ax1.bar(x + width/2, configbleu, width, label="ConfigBLEU", color=NAVY, zorder=3)

    ax1.set_ylabel("Score", fontsize=11, color=GRAY)
    ax1.set_title("ConfigBLEU vs BLEU", fontsize=13, fontweight="bold", color=NAVY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, fontsize=9, rotation=15)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Right: Difference plot
    colors = [RED if d > 0 else GREEN for d in diff]
    bars = ax2.barh(tasks, diff, color=colors, zorder=3)
    ax2.set_xlabel("BLEU - ConfigBLEU", fontsize=11, color=GRAY)
    ax2.set_title("Score Difference (BLEU - ConfigBLEU)", fontsize=13,
                 fontweight="bold", color=NAVY)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(axis="x", alpha=0.3, zorder=0)

    # Add value labels
    for bar, d in zip(bars, diff):
        ax2.text(bar.get_width() + 0.1 * (1 if d >= 0 else -1),
                bar.get_y() + bar.get_height()/2,
                f'{d:+.2f}', ha='left' if d >= 0 else 'right',
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "table_vi_configbleu_vs_bleu.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Table VI: ConfigBLEU vs BLEU")


# ═══════════════════════════════════════════════════════════
# Figure 5: ACL ConfigBLEU Example (论文 Figure 5)
# ═══════════════════════════════════════════════════════════

def reproduce_figure_5():
    """
    论文 Figure 5: ConfigBLEU 和 BLEU 在配置任务评估中的对比示例
    经典 ACL 配置反例
    """
    import sys, importlib
    sys.path.insert(0, str(Path(__file__).parent.parent))
    cbleu = importlib.import_module("scripts.configbleu")
    compute_all_metrics = cbleu.compute_all_metrics

    reference = "access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80"
    candidate_good = reference  # Perfect match
    candidate_bad = "access-list 100 permit tcp 10.0.0.0 0.0.0.255 eq 80 any"  # Order reversed

    metrics_good = compute_all_metrics(candidate_good, reference)
    metrics_bad = compute_all_metrics(candidate_bad, reference)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Correct candidate
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Candidate A: Correct Order", fontsize=12, fontweight="bold", color=GREEN)

    text = f"""Reference:
  access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80

Candidate:
  access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80

Results:
  BLEU:         {metrics_good['bleu']:.4f}  ✓
  BLEU_weight:  {metrics_good['bleu_weight']:.4f}  ✓
  Match_syn:    {metrics_good['match_syn']:.4f}  ✓
   ConfigBLEU:   {metrics_good['config_bleu']:.4f}  ✓"""

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.8))

    # Right: Wrong candidate
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Candidate B: Order Reversed (eq 80 any → any eq 80)",
                fontsize=12, fontweight="bold", color=RED)

    text = f"""Reference:
  access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80

Candidate:
  access-list 100 permit tcp 10.0.0.0 0.0.0.255 eq 80 any
                                              ↑ ORDER REVERSED

Results:
  BLEU:         {metrics_bad['bleu']:.4f}  ← Still high! (same words)
  BLEU_weight:  {metrics_bad['bleu_weight']:.4f}  ← Penalized
  Match_syn:    {metrics_bad['match_syn']:.4f}  ← AST mismatch
   ConfigBLEU:   {metrics_bad['config_bleu']:.4f}  ← Correctly lower"""

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="#FFEBEE", alpha=0.8))

    fig.suptitle("Figure 5: ConfigBLEU vs BLEU — ACL Configuration Example",
                fontsize=14, fontweight="bold", color=NAVY, y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_5_configbleu_acl_example.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 5: ConfigBLEU ACL Example")


# ═══════════════════════════════════════════════════════════
# Additional: Data Pipeline Visualization (论文 Figure 2)
# ═══════════════════════════════════════════════════════════

def reproduce_figure_2():
    """
    论文 Figure 2: Overall framework of PreConfig
    生成 Pipeline 流程图
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Pipeline stages
    stages = [
        (1.0, "Data\nAcquisition", "Cisco/Juniper\nDocs + Forums\n(7GB)"),
        (4.0, "Data\nMining", "HTML Parser\nBoW+KNN\nLLM Translation"),
        (7.0, "Data\nAugmentation", "Prompt Eng.\nGPT-4\nSOP"),
        (10.0, "Model\nFine-Tuning", "Qwen2.5-Coder\nLoRA\nInstr. Tuning"),
        (13.0, "Evaluation", "ConfigBLEU\nBLEU/ROUGE\nMETEOR"),
    ]

    for x, title, desc in stages:
        # Box
        box = plt.Rectangle((x - 1.1, 0.5), 2.2, 2.8, linewidth=1.5,
                           edgecolor=NAVY, facecolor="white", zorder=2)
        ax.add_patch(box)

        # Top color bar
        bar = plt.Rectangle((x - 1.1, 2.95), 2.2, 0.35, facecolor=NAVY, zorder=3)
        ax.add_patch(bar)

        # Stage number
        ax.text(x, 3.12, title, ha="center", va="center",
               fontsize=9, fontweight="bold", color="white", zorder=4)

        # Description
        ax.text(x, 1.8, desc, ha="center", va="center",
               fontsize=8, color=GRAY, zorder=4, linespacing=1.4)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.15
        x2 = stages[i + 1][0] - 1.15
        ax.annotate("", xy=(x2, 1.9), xytext=(x1, 1.9),
                   arrowprops=dict(arrowstyle="->", color=NAVY, lw=2))

    ax.set_title("Figure 2: PreConfig Framework Pipeline", fontsize=14,
                fontweight="bold", color=NAVY, pad=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_2_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 2: Pipeline")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("Reproducing PreConfig Paper Figures and Tables")
    logger.info("=" * 60)

    reproduce_table_ii()
    reproduce_table_iv()
    reproduce_table_v()
    reproduce_table_vi()
    reproduce_figure_2()

    # Figure 5 requires ConfigBLEU module
    try:
        reproduce_figure_5()
    except ImportError:
        logger.warning("Could not import ConfigBLEU module for Figure 5. Run from project root.")

    logger.info("=" * 60)
    logger.info(f"All figures saved to {FIGURES_DIR}")
    logger.info("=" * 60)

    # List generated files
    for f in sorted(FIGURES_DIR.glob("*.png")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
