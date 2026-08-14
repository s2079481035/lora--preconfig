"""
Step 20: BM25 Retrieval Cache
==============================
BM25 关键词检索 baseline: 对 583 测试样本在 v4 训练集上检索 top-10。
纯 CPU, rank-bm25 库 (系统 Python 3.12 已装)。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/20_bm25_retrieve.py
"""

import json, logging, re
from pathlib import Path

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v4.json"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"
OUTPUT = PROJECT_ROOT / "data" / "rag" / "test_retrieval_bm25.json"
TOP_K = 10

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:{};-]+")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1]


def main():
    train = json.load(open(TRAIN_DATA, encoding="utf-8"))
    test = json.load(open(TEST_DATA, encoding="utf-8"))
    logger.info(f"train={len(train)} test={len(test)}")

    docs_text = [f"{s.get('instruction', '')}\n{s.get('input', '')}" for s in train]
    corpus = [tokenize(t) for t in docs_text]
    bm25 = BM25Okapi(corpus)
    logger.info("BM25 index built")

    results = []
    for i, s in enumerate(test):
        q = f"{s.get('instruction', '')}\n{s.get('input', '')}"
        scores = bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda j: -scores[j])[:TOP_K]
        hits = [{
            "score": float(scores[j]),
            "task": train[j].get("task", ""),
            "doc_text": train[j].get("output", ""),
        } for j in order]
        results.append({"sample": i, "query": q, "hits": hits})
        if (i + 1) % 100 == 0:
            logger.info(f"Retrieved {i+1}/{len(test)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()