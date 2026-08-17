"""
Step 27: Semantic Similarity Re-evaluation of Analysis Samples
==============================================================
用 bge-large-en-v1.5 的余弦相似度 (语义) 重新评估 Ana 任务争议样本,
验证 "Ana 下降是 ROUGE 对简洁参考的风格惩罚" 假说。

比较: 无RAG 预测 vs RAG 预测 谁与参考语义更接近。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/27_ana_semantic.py
"""

import json, logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASES = Path("/tmp/ana_cases.json")
MODEL_NAME = "BAAI/bge-large-en-v1.5"


def main():
    cases = json.load(open(CASES, encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Loaded {len(cases)} disputed samples")

    texts = []
    for c in cases:
        texts.extend([c["ref"], c["pred_norag"], c["pred_rag"]])
    embs = model.encode(texts, normalize_embeddings=True, batch_size=16)
    n = len(cases)

    sim_norag, sim_rag = [], []
    for i in range(n):
        ref = embs[i * 3]
        pn = embs[i * 3 + 1]
        pr = embs[i * 3 + 2]
        sn = float(np.dot(ref, pn))
        sr = float(np.dot(ref, pr))
        sim_norag.append(sn)
        sim_rag.append(sr)
        if i < 4:
            c = cases[i]
            print(f"sample {c['sample']}: 无RAG 语义={sn:.3f}  RAG 语义={sr:.3f}  "
                  f"({'RAG更优' if sr > sn else '无RAG更优'})")

    avg_n, avg_r = np.mean(sim_norag), np.mean(sim_rag)
    win_rag = sum(1 for a, b in zip(sim_norag, sim_rag) if b > a)
    print(f"\n=== 结果 (n={n}) ===")
    print(f"无RAG 平均语义相似度: {avg_n:.4f}")
    print(f"RAG   平均语义相似度: {avg_r:.4f}")
    print(f"RAG 语义更优的样本: {win_rag}/{n} ({win_rag/n:.0%})")

    with open("/tmp/ana_semantic_results.json", "w") as f:
        json.dump({"cases": cases, "sim_norag": sim_norag, "sim_rag": sim_rag,
                   "avg_norag": float(avg_n), "avg_rag": float(avg_r),
                   "win_rag": win_rag, "n": n}, f, ensure_ascii=False, indent=2)
    logger.info("Saved → /tmp/ana_semantic_results.json")


if __name__ == "__main__":
    main()