"""
Step 17: Retrieve top-k documents for test samples (cache)
===========================================================
对 583 条测试样本, 用 bge-large-en-v1.5 + FAISS 检索 top-10 相似训练配置,
缓存到 data/rag/test_retrieval.json, 供 RAG 评估脚本按 k 截取。

运行环境: 系统 Python 3.12 (有 faiss + sentence-transformers)
  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/17_rag_retrieve.py
"""

import json, logging, sys
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
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"
OUTPUT = RAG_DIR / "test_retrieval.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 10


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="自定义样本文件 (替代 test_data_multitask.json)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else TEST_DATA
    output_path = Path(args.output) if args.output else OUTPUT
    test_data = json.load(open(data_path, encoding="utf-8"))
    logger.info(f"Loaded {len(test_data)} test samples from {data_path}")

    docs = [json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")]
    index = faiss.read_index(str(INDEX_PATH))
    logger.info(f"Index: {index.ntotal} docs")

    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded")

    results = []
    batch_size = 32
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        queries = [f"{s.get('instruction', '')}\n{s.get('input', '')}" for s in batch]
        embs = model.encode(queries, normalize_embeddings=True)
        scores, idxs = index.search(np.array(embs, dtype="float32"), k=TOP_K)
        for j, (score_row, idx_row) in enumerate(zip(scores, idxs)):
            sample = batch[j]
            hits = []
            for sc, di in zip(score_row, idx_row):
                d = docs[di]
                hits.append({
                    "score": float(sc),
                    "task": d["task"],
                    "doc_text": d["doc_text"],
                })
            results.append({
                "sample": i + j,
                "query": queries[j],
                "hits": hits,
            })
        logger.info(f"Retrieved {i + len(batch)}/{len(test_data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {output_path} ({len(results)} samples, top-{TOP_K} each)")


if __name__ == "__main__":
    main()
