"""
Step 26: Bootstrap Confidence Intervals for RAG Results
=======================================================
对 rag_eval_*.json 的每任务分数做 bootstrap 95% CI。

用法:
  venv/bin/python scripts/26_bootstrap_ci.py --tag v4_k3
  venv/bin/python scripts/26_bootstrap_ci.py --all
"""

import json, argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"

TASKS = [
    ("generation", "config_bleu"),
    ("translation_c2j", "config_bleu"),
    ("translation_j2c", "config_bleu"),
    ("completion", "config_bleu"),
    ("analysis", "rouge_l"),
]

RNG = np.random.default_rng(42)


def bootstrap_ci(scores, n_iter=2000, ci=0.95):
    if not scores:
        return None, None, None
    arr = np.array(scores, dtype=float)
    means = np.array([RNG.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iter)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return arr.mean(), lo, hi


def analyze(tag):
    path = LOGS / f"rag_eval_{tag}.json"
    if not path.exists():
        print(f"  [skip] {tag}: file not found")
        return None
    data = json.load(open(path))
    print(f"\n=== {tag} ===")
    for bucket, key in TASKS:
        scores = [x[key] for x in data.get(bucket, [])]
        mean, lo, hi = bootstrap_ci(scores)
        if mean is not None:
            print(f"  {bucket:<20} mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  n={len(scores)}")
    return mean is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for path in sorted(LOGS.glob("rag_eval_*.json")):
            if "smoke" in path.name:
                continue
            analyze(path.stem.replace("rag_eval_", ""))
    elif args.tag:
        analyze(args.tag)
    else:
        parser.error("需要 --tag 或 --all")


if __name__ == "__main__":
    main()