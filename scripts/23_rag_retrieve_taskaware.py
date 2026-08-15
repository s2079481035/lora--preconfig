"""
Step 23: Task-aware Retrieval Cache
===================================
按任务过滤的检索: 每个测试样本只在其同任务文档子集内检索 top-10。
- config_analysis   → 仅检索 task=config_analysis 文档 (865条, 输出是NL分析)
- config_generation → 仅检索 task=config_generation (1422)
- config_completion → 仅检索 task=config_completion (3441)
- translation_c2j   → 仅检索 task=config_translation_c2j (283)
- translation_j2c   → 仅检索 task=config_translation_j2c (283)

做法: 全量 FAISS 检索 top-N, 过滤保留同任务命中直到凑够 top-K。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/23_rag_retrieve_taskaware.py
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
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"
OUTPUT = RAG_DIR / "test_retrieval_taskaware.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COARSE_K = 200
TOP_K = 10


def main():
    test_data = json.load(open(TEST_DATA, encoding="utf-8"))
    docs = [json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")]
    index = faiss.read_index(str(INDEX_PATH))
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Loaded {len(test_data)} test samples, {index.ntotal} index docs")

    results = []
    batch_size = 32
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        queries = [f"{s.get('instruction', '')}\n{s.get('input', '')}" for s in batch]
        embs = model.encode(queries, normalize_embeddings=True)
        _, idxs = index.search(np.array(embs, dtype="float32"), k=COARSE_K)

        for j, idx_row in enumerate(idxs):
            sample = batch[j]
            task = sample.get("task", "config_generation")
            same_task = [di for di in idx_row if docs[di]["task"] == task]
            selected = same_task[:TOP_K] if len(same_task) >= TOP_K else same_task + idx_row[:TOP_K]
            hits = []
            for di in selected:
                d = docs[di]
                hits.append({"task": d["task"], "doc_text": d["doc_text"]})
            results.append({"sample": i + j, "query": queries[j], "task": task, "hits": hits})
        logger.info(f"Retrieved {i + len(batch)}/{len(test_data)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT}")

    from collections import Counter
    coverage = Counter()
    for r in results:
        coverage[r["task"]] += 1
    logger.info(f"Task coverage: {dict(coverage)}")


if __name__ == "__main__":
    main()