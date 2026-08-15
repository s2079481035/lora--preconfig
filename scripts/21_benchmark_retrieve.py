"""
Step 21: Retrieve for NetworkConfigPro unseen benchmark pairs
==============================================================
对 8 对全新配置 (不在知识库) 做向量检索, 生成 RAG 评估用缓存。
方向: c2j 查询=cisco配置, j2c 查询=juniper配置。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/21_benchmark_retrieve.py
"""

import json, logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "rag"
INDEX_PATH = RAG_DIR / "index.faiss"
DOCS_PATH = RAG_DIR / "docs.jsonl"
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "pairs.json"
OUTPUT = RAG_DIR / "benchmark_retrieval.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 5


def main():
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    docs = [json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")]
    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Loaded {len(pairs)} pairs, {index.ntotal} index docs")

    results = []
    for p in pairs:
        sample = {"sample": p["sample"], "scenario": p["scenario"], "hits": {}}
        for direction, src_key in [("c2j", "cisco"), ("j2c", "juniper")]:
            query = p[src_key]
            emb = model.encode([query], normalize_embeddings=True)
            scores, idxs = index.search(np.array(emb, dtype="float32"), k=TOP_K)
            hits = []
            for sc, di in zip(scores[0], idxs[0]):
                d = docs[di]
                hits.append({"score": float(sc), "task": d["task"], "doc_text": d["doc_text"]})
            sample["hits"][direction] = hits
        results.append(sample)
        logger.info(f"Retrieved for sample {sample['sample']} ({sample['scenario']})")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()