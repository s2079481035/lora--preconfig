"""
Step 25: Retrieve for expanded 40-pair unseen benchmark
=======================================================
对 40 对全新配置做向量检索 (top-5), 生成 RAG 评估用缓存。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/25_benchmark40_retrieve.py
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
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json"
OUTPUT = RAG_DIR / "benchmark40_retrieval.json"

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
        if (p["sample"] + 1) % 10 == 0:
            logger.info(f"Retrieved {p['sample'] + 1}/{len(pairs)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()