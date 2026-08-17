"""
Step 29: Paired Significance Tests (RAG vs no-RAG)
===================================================
对 40 对未见基准做配对 bootstrap 与 permutation test,
回答 "RAG 增益是否显著"。

  venv/bin/python scripts/29_paired_significance.py
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"

RNG = np.random.default_rng(123)
N_ITER = 10000


def load(fn):
    d = json.load(open(LOGS / fn))
    return {"c2j": [x["config_bleu"] for x in d["c2j"]],
            "j2c": [x["config_bleu"] for x in d["j2c"]]}


def paired_bootstrap(a, b):
    """a vs b 配对 bootstrap: 返回差异均值的 95% CI 与 a>b 的比例"""
    n = len(a)
    arr_a, arr_b = np.array(a), np.array(b)
    diffs = []
    for _ in range(N_ITER):
        idx = RNG.integers(0, n, n)
        diffs.append(arr_a[idx].mean() - arr_b[idx].mean())
    diffs = np.array(diffs)
    ci = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), ci, float((diffs > 0).mean())


def permutation(a, b, n_perm=10000):
    """置换检验: H0 两分布相同, 返回观测差异的 p 值 (双侧)"""
    arr_a, arr_b = np.array(a), np.array(b)
    obs = arr_a.mean() - arr_b.mean()
    combined = np.concatenate([arr_a, arr_b])
    n = len(arr_a)
    cnt = 0
    for _ in range(n_perm):
        idx = RNG.permutation(len(combined))
        m1, m2 = combined[idx[:n]].mean(), combined[idx[n:]].mean()
        if abs(m1 - m2) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def report(name, a, b):
    print(f"\n=== {name} (n={len(a)}) ===")
    for d, (ra, rb) in {"C→J": ("c2j", "c2j"), "J→C": ("j2c", "j2c")}.items():
        # 注意: a/b 是 dict, 用 ra/rb 键
        pass


def main():
    v4_norag = load("b40_v4_norag.json")
    v4_rag = load("b40_v4_ragk3.json")
    base_norag = load("b40_base_norag.json")
    base_rag = load("b40_base_ragk3.json")

    for model, norag, rag in [("LoRA v4", v4_norag, v4_rag),
                              ("Base", base_norag, base_rag)]:
        print(f"\n######## {model}: RAG k=3 vs no-RAG (n=40) ########")
        for d in ["c2j", "j2c"]:
            mean, ci, pgt = paired_bootstrap(rag[d], norag[d])
            obs, pval = permutation(rag[d], norag[d])
            # 配对 t 检验作为参考
            from scipy import stats
            t, tp = stats.ttest_rel(rag[d], norag[d])
            print(f"  {d:<4} 均值差={mean:+.4f}  95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                  f"RAG优占比={pgt:.3f}  permutation p={pval:.4f}  paired-t p={tp:.4f}")


if __name__ == "__main__":
    main()