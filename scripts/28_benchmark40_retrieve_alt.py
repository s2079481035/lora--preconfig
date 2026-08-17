"""
Step 28: BM25 + Reranker retrieval for 40-pair benchmark
========================================================
对 40 对基准生成 BM25 和 rerank 检索缓存 (CPU), 用于消融。

  HF_HUB_OFFLINE=1 /usr/bin/python3.12 scripts/28_benchmark40_retrieve_alt.py [--method bm25|rerank]
"""

import json, logging, re, argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "rag"
INDEX_PATH = RAG_DIR / "index.faiss"
DOCS_PATH = RAG_DIR / "docs.jsonl"
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json"

QUERY_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
COARSE_K = 20
TOP_K = 5
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:{};-]+")


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text) if len(t) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "rerank"], required=True)
    args = parser.parse_args()

    pairs = json.load(open(PAIRS, encoding="utf-8"))
    docs = [json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")]
    train = json.load(open(PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v4.json"))
    logger.info(f"Loaded {len(pairs)} pairs, {len(docs)} docs")

    results = []
    if args.method == "bm25":
        corpus = [tokenize(f"{s.get('instruction','')}\n{s.get('input','')}") for s in train]
        bm25 = BM25Okapi(corpus)
        for p in pairs:
            sample = {"sample": p["sample"], "scenario": p["scenario"], "hits": {}}
            for direction, src_key in [("c2j", "cisco"), ("j2c", "juniper")]:
                q = p[src_key]
                scores = bm25.get_scores(tokenize(q))
                order = sorted(range(len(scores)), key=lambda j: -scores[j])[:TOP_K]
                sample["hits"][direction] = [{
                    "score": float(scores[j]), "task": train[j].get("task", ""),
                    "doc_text": train[j].get("output", "")} for j in order]
            results.append(sample)
        logger.info("BM25 retrieval done")
    else:
        index = faiss.read_index(str(INDEX_PATH))
        query_model = SentenceTransformer(QUERY_MODEL)
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        for p in pairs:
            sample = {"sample": p["sample"], "scenario": p["scenario"], "hits": {}}
            for direction, src_key in [("c2j", "cisco"), ("j2c", "juniper")]:
                query = p[src_key]
                emb = query_model.encode([query], normalize_embeddings=True)
                _, idxs = index.search(np.array(emb, dtype="float32"), k=COARSE_K)
                cands = [docs[di] for di in idxs[0]]
                scores = reranker.predict([(query, d["doc_text"][:512]) for d in cands])
                order = np.argsort(-np.array(scores))[:TOP_K]
                sample["hits"][direction] = [{
                    "score": float(scores[di]), "task": cands[di]["task"],
                    "doc_text": cands[di]["doc_text"]} for di in order]
            results.append(sample)
        logger.info("Rerank retrieval done")

    out = RAG_DIR / f"benchmark40_retrieval_{args.method}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()