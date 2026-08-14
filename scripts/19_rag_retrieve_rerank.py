"""
Step 19: Hybrid Retrieval with Reranker
=======================================
两阶段检索: 向量粗筛 top-20 (bge-large-en-v1.5 + FAISS)
→ bge-reranker-base 精排 → 保存 top-10 排序结果缓存。

运行环境: 系统 Python 3.12
  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/19_rag_retrieve_rerank.py
"""

import json, logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "rag"
INDEX_PATH = RAG_DIR / "index.faiss"
DOCS_PATH = RAG_DIR / "docs.jsonl"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"
OUTPUT = RAG_DIR / "test_retrieval_rerank.json"

QUERY_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
COARSE_K = 20
TOP_K = 10


def main():
    test_data = json.load(open(TEST_DATA, encoding="utf-8"))
    logger.info(f"Loaded {len(test_data)} test samples")

    docs = [json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")]
    index = faiss.read_index(str(INDEX_PATH))

    query_model = SentenceTransformer(QUERY_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    logger.info("Models loaded")

    results = []
    batch_size = 16
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        queries = [f"{s.get('instruction', '')}\n{s.get('input', '')}" for s in batch]
        embs = query_model.encode(queries, normalize_embeddings=True)
        _, idxs = index.search(np.array(embs, dtype="float32"), k=COARSE_K)

        for j, idx_row in enumerate(idxs):
            cand_docs = [docs[di] for di in idx_row]
            pairs = [(queries[j], d["doc_text"][:512]) for d in cand_docs]
            scores = reranker.predict(pairs)
            order = np.argsort(-np.array(scores))
            hits = []
            for rank in order[:TOP_K]:
                d = cand_docs[rank]
                hits.append({
                    "score": float(scores[rank]),
                    "task": d["task"],
                    "doc_text": d["doc_text"],
                })
            results.append({"sample": i + j, "query": queries[j], "hits": hits})
        logger.info(f"Reranked {i + len(batch)}/{len(test_data)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT} ({len(results)} samples, top-{TOP_K} after rerank)")


if __name__ == "__main__":
    main()