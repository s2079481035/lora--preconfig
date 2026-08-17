"""
Step 34: Retrieval Quality (Parameter-Hit Recall@k)
===================================================
直接评测检索器: 对每个查询, 计算 top-k 检索文档与目标配置的
参数 Jaccard 重合度, 报告 Recall@k (k 内至少命中 1 个正确参数)
与平均最大参数 F1。

  venv/bin/python scripts/34_retrieval_quality.py
"""

import json, logging, importlib.util
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAG_DIR = PROJECT_ROOT / "data" / "rag"
LOGS = PROJECT_ROOT / "logs"

spec = importlib.util.spec_from_file_location(
    "ana32", PROJECT_ROOT / "scripts" / "32_ana_param_accuracy.py")
ana32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ana32)
extract_params = ana32.extract_params

DATA_SETS = [
    ("main-testset", PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json",
     RAG_DIR / "test_retrieval.json", "output"),
    ("benchmark40", PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json",
     RAG_DIR / "benchmark40_retrieval.json", "target", "pairs"),
    ("benchmark40-bm25", PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json",
     RAG_DIR / "benchmark40_retrieval_bm25.json", "target", "pairs"),
    ("benchmark40-rerank", PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json",
     RAG_DIR / "benchmark40_retrieval_rerank.json", "target", "pairs"),
]


def param_f1(pred_params, target_params):
    if not pred_params or not target_params:
        return 0.0
    inter = pred_params & target_params
    rec = len(inter) / len(target_params)
    prec = len(inter) / len(pred_params)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def evaluate_samples(samples, retrieval, target_key, fmt="multitask"):
    """samples: multitask 格式或 pairs 格式; retrieval: 22/17 格式缓存。"""
    results = []
    for i, s in enumerate(samples):
        if fmt == "pairs":
            # pairs 格式: 每个 pair 两个方向, 检索缓存 sample 0-39 两方向
            r = next((x for x in retrieval if x["sample"] == s["sample"]), None)
            if not r:
                continue
            for direction, src_key, tgt_key in [("c2j", "cisco", "juniper"),
                                                ("j2c", "juniper", "cisco")]:
                tgt = extract_params(s[tgt_key])
                hits = r["hits"].get(direction, [])
                top_f1 = [param_f1(extract_params(h["doc_text"]), tgt) for h in hits[:10]]
                results.append({"sample": s["sample"], "direction": direction,
                                "target_params": len(tgt), "top1_f1": top_f1[0] if top_f1 else 0.0,
                                "top5_max": max(top_f1[:5]) if top_f1 else 0.0,
                                "top10_max": max(top_f1) if top_f1 else 0.0,
                                "recall@5": float(any(f > 0 for f in top_f1[:5])),
                                "recall@10": float(any(f > 0 for f in top_f1))})
        else:
            tgt = extract_params(s.get("output", ""))
            r = retrieval[i]
            top_f1 = [param_f1(extract_params(h["doc_text"]), tgt) for h in r["hits"][:10]]
            results.append({"sample": i, "task": s.get("task"),
                            "target_params": len(tgt), "top1_f1": top_f1[0] if top_f1 else 0.0,
                            "top5_max": max(top_f1[:5]) if top_f1 else 0.0,
                            "top10_max": max(top_f1) if top_f1 else 0.0,
                            "recall@5": float(any(f > 0 for f in top_f1[:5])),
                            "recall@10": float(any(f > 0 for f in top_f1))})
    return results


def main():
    for name, data_path, ret_path, target_key, *rest in DATA_SETS:
        fmt = rest[0] if rest else "multitask"
        samples = json.load(open(data_path, encoding="utf-8"))
        retrieval = json.load(open(ret_path, encoding="utf-8"))
        res = evaluate_samples(samples, retrieval, target_key, fmt)
        n = len(res)
        if not n:
            logger.info(f"{name}: no samples")
            continue
        print(f"\n=== {name} (n={n}) ===")
        for key, label in [("top1_f1", "top-1 参数F1"), ("top5_max", "top-5 最大F1"),
                           ("top10_max", "top-10 最大F1"), ("recall@5", "参数命中率@5"),
                           ("recall@10", "参数命中率@10")]:
            vals = [r[key] for r in res]
            print(f"  {label:<14}: {np.mean(vals):.4f}")

    print("\n注意: 主测试集与知识库同模板族, 命中率是上界; 未见基准反映真实检索质量.")


if __name__ == "__main__":
    main()