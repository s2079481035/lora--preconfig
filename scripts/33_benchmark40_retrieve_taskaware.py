"""
Step 33: Task-aware retrieval for the 40-pair benchmark
=======================================================
对 40 对基准做 task-aware 向量检索: c2j 查询只保留翻译 c2j 类文档,
j2c 查询只保留翻译 j2c 类文档。输出兼容 22 脚本格式。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/33_benchmark40_retrieve_taskaware.py
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
OUTPUT = RAG_DIR / "benchmark40_retrieval_taskaware.json"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COARSE_K = 20
TOP_K = 5

TASK_BY_DIR = {"c2j": "config_translation_c2j", "j2c": "config_translation_j2c"}


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
            _, idxs = index.search(np.array(emb, dtype="float32"), k=COARSE_K)
            want_task = TASK_BY_DIR[direction]
            same_task = [di for di in idxs[0] if docs[di]["task"] == want_task]
            selected = same_task[:TOP_K] if len(same_task) >= TOP_K else same_task + list(idxs[0][:TOP_K])
            hits = []
            for di in selected:
                d = docs[di]
                hits.append({"score": None, "task": d["task"], "doc_text": d["doc_text"]})
            sample["hits"][direction] = hits
        results.append(sample)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {OUTPUT}")

    from collections import Counter
    task_hits = Counter()
    for r in results:
        for direction in ["c2j", "j2c"]:
            for h in r["hits"][direction]:
                task_hits[h["task"]] += 1
    logger.info(f"Task distribution of hits: {dict(task_hits)}")


if __name__ == "__main__":
    main()