"""
Step 30b: Token-level F1 via bge-large-en-v1.5 (lightweight BERTScore)
====================================================================
手动实现 BERTScore 核心: token 嵌入余弦相似度矩阵 → max-row/col →
precision/recall/F1。用 bge-large-en-v1.5 (缓存已有)。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/30b_ana_bertscore_bge.py
"""

import json, logging
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def main():
    norag = json.load(open(LOGS / "rag_eval_v4_norag.json"))["analysis"]
    rag = json.load(open(LOGS / "rag_eval_v4_k3.json"))["analysis"]
    refs = [x["reference"] for x in norag]
    preds_norag = [x["prediction"] for x in norag]
    preds_rag = [x["prediction"] for x in rag]

    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Loaded {len(refs)} samples, model {MODEL_NAME}")

    def token_f1(pred, ref):
        pe = model.encode(pred, output_value="token_embeddings")
        re = model.encode(ref, output_value="token_embeddings")
        pe = torch.nn.functional.normalize(pe, dim=-1)
        re = torch.nn.functional.normalize(re, dim=-1)
        if pe.dim() == 1:
            pe = pe.unsqueeze(0)
        if re.dim() == 1:
            re = re.unsqueeze(0)
        sim = pe @ re.T
        prec = sim.max(dim=1).values.mean().item()
        rec = sim.max(dim=0).values.mean().item()
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return f1

    fn_vals, fr_vals = [], []
    for i, (pn, pr, ref) in enumerate(zip(preds_norag, preds_rag, refs)):
        fn_vals.append(token_f1(pn, ref))
        fr_vals.append(token_f1(pr, ref))
        if (i + 1) % 25 == 0:
            logger.info(f"Scored {i + 1}/{len(refs)}")

    print(f"\n=== Ana Token-F1 (bge-large-en-v1.5, n={len(refs)}) ===")
    print(f"no-RAG: {np.mean(fn_vals):.4f}")
    print(f"RAG:    {np.mean(fr_vals):.4f}")
    print(f"RAG 更高样本: {sum(1 for a, b in zip(fn_vals, fr_vals) if b > a)}/{len(refs)}")

    out = []
    for x, fn, fr in zip(norag, fn_vals, fr_vals):
        out.append({"sample": x["sample"], "token_f1_norag": float(fn),
                    "token_f1_rag": float(fr), "rouge_norag": x["rouge_l"]})
    with open("/tmp/ana_bertscore_bge_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Saved → /tmp/ana_bertscore_bge_results.json")


if __name__ == "__main__":
    main()