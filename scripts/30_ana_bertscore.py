"""
Step 30: BERTScore Re-evaluation of Analysis Task
=================================================
用 BERTScore (bge-large-en-v1.5 特征) 重评 Ana 116 条样本,
对比 no-RAG vs RAG 预测相对参考的 F1, 检查 ROUGE 结论是否稳健。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/30_ana_bertscore.py
"""

import json, logging
from pathlib import Path

import torch
from bert_score import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"
MODEL_TYPE = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    norag = json.load(open(LOGS / "rag_eval_v4_norag.json"))["analysis"]
    rag = json.load(open(LOGS / "rag_eval_v4_k3.json"))["analysis"]
    # 保持样本对齐 (同 sample 顺序)
    refs = [x["reference"] for x in norag]
    preds_norag = [x["prediction"] for x in norag]
    preds_rag = [x["prediction"] for x in rag]

    logger.info(f"Scoring {len(refs)} samples with BERTScore ({MODEL_TYPE}, {DEVICE})")
    p_n, r_n, f_n = score(preds_norag, refs, model_type=MODEL_TYPE,
                          device=DEVICE, lang="en", rescale_with_baseline=True)
    p_r, r_r, f_r = score(preds_rag, refs, model_type=MODEL_TYPE,
                          device=DEVICE, lang="en", rescale_with_baseline=True)

    f_n = f_n.tolist()
    f_r = f_r.tolist()
    print(f"\n=== Ana BERTScore-F1 (n={len(refs)}) ===")
    print(f"no-RAG: {sum(f_n)/len(f_n):.4f}")
    print(f"RAG:    {sum(f_r)/len(f_r):.4f}")
    print(f"RAG 更高样本: {sum(1 for a, b in zip(f_n, f_r) if b > a)}/{len(refs)}")

    # 逐样本保存
    out = []
    for x, fn, fr in zip(norag, f_n, f_r):
        out.append({"sample": x["sample"], "bert_f1_norag": fn, "bert_f1_rag": fr,
                    "rouge_norag": x["rouge_l"]})
    with open("/tmp/ana_bertscore_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Saved → /tmp/ana_bertscore_results.json")


if __name__ == "__main__":
    main()