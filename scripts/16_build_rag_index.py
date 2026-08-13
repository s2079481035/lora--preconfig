"""
Step 16: Build RAG Knowledge Base Index
=======================================
用 bge-large-en-v1.5 对 v4 训练集的检索文本编码, 建 FAISS 索引。

知识库文档 = v4 训练集每条样本:
  - doc_text: 配置 output (作为可注入参考)
  - query_text: instruction + input (作为检索匹配文本)

检索流程: 测试样本 instruction+input → 编码 → FAISS top-k → 返回 doc_text 列表

运行环境: 系统 Python 3.12 (有 sentence-transformers + faiss-gpu)
  /usr/bin/python3.12 scripts/16_build_rag_index.py

输出: data/rag/index.faiss, data/rag/docs.jsonl
"""

import json, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v4.json"
RAG_DIR = PROJECT_ROOT / "data" / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = RAG_DIR / "index.faiss"
DOCS_PATH = RAG_DIR / "docs.jsonl"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024


def load_docs():
    data = json.load(open(TRAIN_DATA, encoding="utf-8"))
    docs = []
    for i, s in enumerate(data):
        inst = s.get("instruction", "")
        inp = s.get("input", "")
        out = s.get("output", "")
        task = s.get("task", "")
        docs.append({
            "id": i,
            "query_text": f"{inst}\n{inp}",
            "doc_text": out,
            "task": task,
        })
    logger.info(f"Loaded {len(docs)} docs from {TRAIN_DATA}")
    return docs


def encode(queries, model, batch_size=32):
    embs = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.extend(emb.tolist())
        if (i + batch_size) % (batch_size * 4) == 0:
            logger.info(f"  encoded {i + len(batch)}/{len(queries)}")
    return embs


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded")

    docs = load_docs()
    queries = [d["query_text"] for d in docs]

    logger.info(f"Encoding {len(queries)} query texts...")
    t0 = time.time()
    embs = encode(queries, model)
    logger.info(f"Encoded in {time.time() - t0:.1f}s")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    import numpy as np
    index.add(np.array(embs, dtype="float32"))
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"Index saved → {INDEX_PATH} ({index.ntotal} vectors)")

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info(f"Docs saved → {DOCS_PATH} ({len(docs)} docs)")


if __name__ == "__main__":
    main()
