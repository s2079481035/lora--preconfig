# Network Configuration Generation: LoRA Fine-tuning × RAG

## Overview

Building network configuration generation for Cisco IOS and Juniper Junos with a
Qwen2.5-Coder-1.5B backbone. Two enhancement tracks are explored and compared:

1. **LoRA fine-tuning** on task-mixed data (generation, translation C2J/J2C, completion, analysis)
2. **Retrieval-Augmented Generation (RAG)** with a FAISS index over the training configs

Evaluation is done on a held-out test set of **583 samples**
(Generation=107, Translation C->J=14, J->C=14, Completion=332, Analysis=116).

**Metrics**: ConfigBLEU (structure-aware config similarity) for generation/translation/completion;
ROUGE-L for analysis (natural language output).

## Data Evolution

| Version | Train samples | Additions |
|---|---|---|
| v2 | 5694 | internal synthetic + crawled data |
| v3 | 5994 | + NIT (473 gen), jvd (75 gen, 75 completion) |
| v4 | 6294 | + NetworkConfigPro translation pairs (300: 150 C2J + 150 J2C) |

Knowledge base for RAG: v4 training set (6294 configs), embedded with
`BAAI/bge-large-en-v1.5`, indexed with FAISS (inner product, normalized).
No overlap between knowledge base and test set.

## Model Evolution (without RAG)

| Model | Generation | C->J | J->C | Completion | Analysis |
|---|---|---|---|---|---|
| Base (zero-shot) | 0.0806 | 0.1951 | 0.3348 | - | - |
| LoRA v2 | 0.8880 | 0.9960 | 0.8590 | 0.9690 | 0.6560 |
| LoRA v3 (+NIT/jvd) | 0.8850 | 0.9125 | **0.8821** | 0.9686 | 0.6537 |
| LoRA v4 (+translation) | **0.8895** | 0.9708 | 0.8333 | 0.9642 | **0.6583** |

## RAG × LoRA 2×2 Results

Retrieval: query = `instruction + input`, embedded with bge-large-en-v1.5,
top-k docs injected into the prompt as reference configurations.

| Config | Generation | C->J | J->C | Completion | Analysis |
|---|---|---|---|---|---|
| Base (no-RAG @512, unified) | 0.0813 | 0.0727 | 0.1905 | 0.3324 | 0.0921 |
| Base + RAG k=1 | 0.3201 | 0.1728 | 0.6462 | 0.5273 | 0.1214 |
| Base + RAG k=3 | 0.4244 | 0.1783 | 0.3375 | 0.5948 | 0.1198 |
| Base + RAG k=5 | 0.4509 | 0.1801 | 0.3242 | 0.5732 | 0.1167 |
| Base + RAG k=10 | **0.5737** | 0.1621 | 0.3321 | 0.5862 | 0.1173 |
| LoRA v4 | 0.8895 | 0.9708 | 0.8333 | 0.9642 | 0.6583 |
| LoRA v4 + RAG k=1 | 0.8228 | 0.9708 | 0.9225 | 0.7761 | 0.3855 |
| LoRA v4 + RAG k=3 | 0.8172 | **1.0000** | **0.9395** | 0.9088 | 0.5430 |
| LoRA v4 + RAG k=5 | 0.8102 | **1.0000** | 0.9406 | 0.9398 | 0.5537 |
| LoRA v4 + RAG k=10 | 0.7995 | 0.9189 | 0.9406 | 0.8967 | 0.5282 |

Note: the previous "Base no-RAG" row used 0.1951/0.3348 from an older eval script
(`07_evaluate_base.py`) with a different prompt; re-run with the unified RAG-eval
script at @512 gives 0.0727/0.1905. With consistent prompts, **RAG lifts the base
model substantially on every task** (+77% J->C, +145% C->J, +422% generation,
+79% completion).

## Key Findings

1. **Fine-tuning is the dominant factor**: Generation 0.08 → 0.89 (11x) from LoRA on
   task-mixed data. RAG alone (on base model) reaches 0.57 but still well below LoRA.

2. **RAG helps the base model substantially**: Generation 0.0806 → 0.5737 (k=10, ~7x).
   In-context examples teach the untrained model config syntax and structure.

3. **RAG is a double-edged sword for fine-tuned models**:
   - **Fixes weak tasks**: J->C translation 0.8333 → 0.9406 (retrieved parallel configs
     give the model the exact structure to fill in); C->J reaches 1.0000 at k=3/5.
   - **Harms internalized tasks**: Generation/Completion/Analysis drop — the model already
     memorized training templates, and injected references add noise/distract attention.
     Analysis drops most (0.6583 → 0.3855~0.5537) since retrieved "configs" are poor
     evidence for NL-output analysis tasks.

4. **Opposite k-trends**: base model improves with more references (k=1→10: 0.32→0.57),
   fine-tuned model degrades (k=1→10: 0.82→0.80). Selecting k by model type matters.

5. **Caveat**: knowledge base shares the same template generator as the test set, so
   retrieved neighbors are near-identical templates (only parameters differ). RAG scores
   here represent an optimistic upper bound for this benchmark.

## Retriever Comparison: Vector vs Reranker vs BM25

Three retrievers over the same 6294-doc knowledge base:

| Retriever | Base Gen | v4 Gen | v4 J->C | v4 Comp |
|---|---|---|---|---|
| No RAG | 0.0806 | 0.8895 | 0.8333 | 0.9642 |
| Vector k=1 | 0.3201 | 0.8228 | 0.9225 | 0.7761 |
| Vector k=3 | 0.4244 | 0.8172 | 0.9395 | 0.9088 |
| Vector k=5 | 0.4509 | 0.8102 | 0.9406 | 0.9398 |
| Vector k=10 | **0.5737** | 0.7995 | 0.9406 | 0.8967 |
| Reranker k=1 | 0.3541 | 0.8208 | 0.9029 | 0.7799 |
| Reranker k=3 | 0.3822 | **0.8223** | 0.9066 | 0.9144 |
| Reranker k=5 | 0.4125 | 0.8150 | 0.8804 | **0.9409** |
| Reranker k=10 | 0.5264 | 0.8088 | 0.9395 | 0.9035 |
| BM25 k=3 | 0.3427 | 0.7682 | 0.8059 | 0.8969 |
| BM25 k=5 | 0.3880 | 0.7623 | 0.8035 | 0.9096 |

**Findings**:
1. **Reranker gives marginal gains over pure vector search for v4** (Gen k=3:
   0.8172 → 0.8223; Comp k=5: 0.9398 → 0.9409) but **hurts J->C** (0.9395 → 0.9066)
   — reranking emphasizes lexical/semantic relevance and can break the near-duplicate
   template ordering that helps translation.
2. **Reranker does not help the base model**: k=3 Gen 0.4244 (vector) vs 0.3822 (rerank);
   k=10 0.5737 vs 0.5264. With a coarse pool of near-duplicate templates, the extra
   ranking stage adds noise without new evidence.
3. **BM25 is a solid lightweight baseline** (base Gen 0.34~0.39, no GPU/embedding needed)
   but consistently below vector retrieval; for v4 it also underperforms vector search
   on all tasks (Gen 0.76~0.77 vs 0.80~0.82).
4. Overall: **vector retrieval + LoRA v4 remains the best practical configuration**;
   reranker and BM25 are useful ablations showing the retriever's robustness.

## Unseen-Config Validation (NetworkConfigPro Benchmark)

The 583-sample test set shares a template generator with the knowledge base, so RAG
gains there are an optimistic upper bound. To test **real generalization**, we evaluate
RAG on 8 brand-new configurations from NetworkConfigPro (absent from the knowledge base;
retrieved neighbors are structurally similar but parameter-different, top-1 cosine ~0.81-0.86):

| Config | C->J | J->C |
|---|---|---|
| Base (no RAG) | 0.1951 | 0.3348 |
| Base + RAG k=3 | **0.2853** (+46%) | **0.3435** (+3%) |
| LoRA v4 (no RAG) | 0.3181 | 0.6681 |
| LoRA v4 + RAG k=3 | **0.3997** (+26%) | 0.4500 (-33%) |

**Findings** (n=8; superseded by the 40-pair benchmark below, which is more reliable):
1. RAG improves C->J translation on truly unseen configs for both models
   (base +46%, v4 +26%).
2. RAG hurts v4 J->C on unseen configs (-33%): parameter copy from the retrieved
   reference (e.g. hostname `rt-045` from the retrieved doc instead of `rt-098` from
   the source Junos config). This was later shown to be a small-sample artifact — see
   the 40-pair benchmark, where J->C no longer degrades.
3. The direction asymmetry observation survives but is much smaller at n=40.

### Expanded benchmark: 40 fresh pairs (reliable estimate)

`scripts/24_gen_benchmark_40.py` generated 40 fresh parallel Cisco/Junos configs
(seed 2024, zero overlap with the 8-pair set and the knowledge base; 0 render fails).
Full matrix evaluated with `scripts/22_eval_benchmark_rag.py --bench40`:

| Config | C->J | J->C |
|---|---|---|
| Base (no RAG) | 0.1632 | 0.2463 |
| Base + RAG k=3 | **0.2945** (+80%) | **0.3906** (+59%) |
| LoRA v4 (no RAG) | 0.2373 | 0.6257 |
| LoRA v4 + RAG k=3 | **0.3570** (+51%) | **0.6360** (+2%) |
| Base + RAG k=3 + sanitize | 0.2734 (+68%) | 0.2646 (+7%) |
| LoRA v4 + RAG k=3 + sanitize | 0.2742 (+16%) | 0.5956 (-5%) |

Retriever ablation at k=3 (v4): vector C->J 0.3570 / J->C 0.6360; BM25 0.3577 /
0.6554; reranker 0.3374 / 0.5381. Same ordering as the main test set (rerank worse,
BM25 ≈ vector).

**Key results (n=40, statistically robust):**
1. **RAG improves C->J translation on truly unseen configs for both models
   (base +80%, v4 +51%)** — the strongest and most reliable RAG effect.
2. **v4 J->C no longer degrades with RAG** (+2%): the -33% at n=8 was a
   small-sample artifact. Base J->C still improves strongly (+59%).
3. RAG helps the weaker model (base) more across the board — consistent with
   reference injection providing structure that the base model lacks.
4. **Sanitize flips from helpful (n=8) to neutral/harmful (n=40)**: with fresh
   random configs, retrieved references do not lure the model into parameter
   plagiarism (they are clearly unrelated), so removing parameters only destroys
   information (e.g. hostname `rt-003` → `<HOSTNAME>` mismatch). Sanitize only
   helps when reference and source share a template family (the n=8 case).

### Unified no-RAG baselines (@max_new_tokens=512)
Re-run no-RAG on the 8-pair benchmark with the same script/tokens as RAG runs:
v4 C->J 0.3181 / J->C 0.6681; base C->J 0.1970 / J->C 0.3348. Confirms the
earlier @512 values; the old 0.3012 figure was the @400 setting.

### Second fresh benchmark: 60 pairs (seed 7777, replication)
`translate_pairs_60.json` — another 60 fresh pairs (seed 7777, zero overlap) as a
replication set. Full matrix: v4 no-RAG 0.2260/0.6655 → v4 RAG k=3 0.3123/0.5722
(C->J +38%, J->C -14%); base no-RAG 0.1408/0.2292 → base RAG k=3 0.2729/0.3662
(C->J +94%, J->C +60%). C->J gains replicate strongly; v4 J->C again swings
(noise + parameter plagiarism, e.g. sample 29 copies hostname rt-039 from the
retrieved reference instead of rt-029), consistent with the n=40 significance
result that v4 J->C RAG effect is not significant. Combined n=100 (40+60):
v4 C->J +~44%, J->C ~±8% (non-significant); base C->J +~87%, J->C +~59%.

### Unseen multi-task evaluation (Gen/Comp/Ana on unseen configs, n=80 each)
`scripts/31_build_unseen_multitask.py` builds generation/completion/analysis
samples from the 40 unseen pairs (query→config, truncation→config, config→
description), `scripts/17` retrieves, `scripts/18` evaluates:

| Task | v4 no-RAG | v4 RAG k=3 | base no-RAG | base RAG k=3 |
|---|---|---|---|---|
| Generation (ConfigBLEU) | 0.0586 | **0.1002** (+71%) | 0.0737 | **0.1083** (+47%) |
| Completion (ConfigBLEU) | 0.4135 | **0.5412** (+31%) | 0.2572 | **0.4710** (+83%) |
| Analysis (param-F1) | 0.169 | **0.355** (+110%) | 0.495 | **0.644** (+30%) |

RAG helps generation, completion and analysis on truly unseen configs for both
models — the first evidence outside the template-family test set. (Analysis rows
use parameter-accuracy F1, not ROUGE — see Case 2.)

### 40-pair benchmark: k-scan and task-aware (v4)
k=1/3/5/10 on the 40-pair benchmark: C->J 0.2888/0.3570/0.3549/0.3549,
J->C 0.6120/0.6360/0.6532/0.6532 — gains saturate at k=3. Task-aware retrieval
at k=3 (only same-direction translation docs): C->J 0.3570 (= vector, since the
KB translation docs dominate anyway), J->C 0.6457 (≈ vector) — no gain, matching
the main-testset task-aware finding.

## Sanitize: Reference Parameter Cleaning

When RAG references contain concrete parameters (hostname/IP/AS/port/password), a
fine-tuned model can *plagiarize* them — copying e.g. the retrieved hostname into the
output instead of the source's hostname. We add a `--sanitize` mode that replaces such
values with placeholders (`<HOSTNAME>`, `<IP>`, `<AS>`, `<PORT>`, `<PASSWORD>`), keeping
only structural syntax. (Script: `scripts/sanitize_ref.py`)

### Unseen benchmark (8 pairs, k=3)

| Config | C->J | J->C |
|---|---|---|
| Base + RAG | 0.2853 | 0.3435 |
| Base + RAG + sanitize | 0.2844 | 0.1801 (--0.16) |
| v4 + RAG | 0.3997 | 0.4500 |
| v4 + RAG + sanitize | 0.3113 | **0.6503** (+0.20) |

### Main testset (583, k=3)

| Config | Gen | C->J | J->C | Comp | Ana |
|---|---|---|---|---|---|
| v4 + RAG | 0.8172 | 1.0000 | 0.9395 | 0.9088 | 0.5430 |
| v4 + RAG + sanitize | **0.8384** | 0.9685 | 0.9024 | **0.9471** | 0.5365 |
| Base + RAG | 0.4244 | 0.1783 | 0.3375 | 0.5948 | 0.1198 |
| Base + RAG + sanitize | 0.4003 | 0.1922 | **0.6006** | 0.5121 | 0.1183 |

**Findings**:
1. **Sanitize fixes v4 param plagiarism**: unseen J->C 0.4500 → 0.6503; main testset
   Gen/Comp also improve. The fine-tuned model has internal parameter-extraction skill;
   structure-only references let it apply that skill without reference-param interference.
2. **Sanitize dramatically helps base J->C on the main testset**: 0.3375 → 0.6006,
   13/14 samples improve. For same-family (template) references, the base model uses the
   cleaned reference as a pure structure template and fills in source parameters.
3. **Sanitize hurts base J->C on unseen benchmark** (0.3435 → 0.1801): retrieved
   references there are cross-syntax (similarity ~0.07), so after cleaning they carry
   almost no usable evidence — the base model loses the param hints it relied on.
4. **Interpretation**: sanitization effectiveness depends on *reference-source similarity*.
   High similarity → cleaning is a win (structure template); low similarity → cleaning
   destroys the only information the reference carried.

## Task-Aware Retrieval

Retrieve only within the same task's document subset (analysis retrieves only
`config_analysis` NL docs, etc.). On the main testset this changed almost nothing
(v4 k=3: Gen 0.8172→0.8164, Ana 0.5430→0.5430): vector search already retrieves
same-task docs because queries and docs within a task share vocabulary. Task filtering
is thus a cheap no-op here, but would matter for heterogeneous corpora.

## Error Case Analysis

### Case 1: Reference-parameter plagiarism (translation, unseen benchmark)
v4 J->C, sample 0: source Junos has `host-name gw-01`, but RAG output says
`hostname rt-074` — the hostname from the retrieved Cisco reference (rt-046 family).
ConfigBLEU drops 0.7591 → 0.1119. `--sanitize` restores it to 0.5793 (hostname `gw-01`).
This motivated the sanitize experiment.

### Case 2: Analysis "degradation" — degenerate references, not real regression
v4 analysis samples (6, 8, 21, 31, ...): without RAG the model emits a terse summary
("Configure BGP on Juniper with AS 65162") that exactly matches the short reference
(ROUGE=1.0, bge semantic sim=1.0 — string-identical). The reference is itself the
**training template sentence** (its AS number is unrelated to the source config,
e.g. src AS 65408 vs ref AS 65362), so the high no-RAG score is memorization, not ability.
With RAG, the model produces verbose analyses that (a) are more complete but (b) **copy
parameters from the retrieved reference** (e.g. AS 65362 and neighbor 192.168.82.1
instead of the source's 65408/192.168.128.1) — parameter plagiarism extends to the
analysis task. ROUGE=0.2~0.3. 18 samples show this pattern.
Automatic metrics (ROUGE 0.6583→0.5430, sentence-cosine 1.0000→0.7657, token-F1
0.9139→0.8832) all "agree" with no-RAG because the gold is a memorized template —
every metric measures string-match against a degenerate reference.
**Definitive resolution — parameter-accuracy evaluation (`scripts/32_ana_param_accuracy.py`):**
extract (type, value) parameter tuples from the source config and from each prediction;
RAG achieves recall 0.4098 vs no-RAG 0.1505, F1 0.355 vs 0.169, RAG better in 42/116
(paired-t p=5.4e-06). On the 18 disputed samples: RAG F1 0.528 vs no-RAG 0.111.
**Conclusion: RAG genuinely improves analysis quality (parameter correctness);
the apparent degradation was entirely a degenerate-reference evaluation artifact.**
The analysis-task ROUGE metric is unreliable; parameter accuracy is the reported metric.

### Case 3: Template-family plagiarism (generation, v3)
v3 generation samples contaminated by jvd `policy-statement export-direct` /
`route-filter` templates (ConfigBLEU dropped to ~0.59); fixed in v4 by training-set
cleanup. Shows model-side memorization risk, distinct from retrieval-side plagiarism.

## Statistical-Rigor Fixes (in progress)

### 1. Expanded unseen benchmark: 8 → 40 pairs
`scripts/24_gen_benchmark_40.py` regenerates 40 fresh parallel Cisco/Junos configs
with NetworkConfigPro (seed 2024, zero overlap with the original 8 or the train set;
0 render failures). Retrieval caches: `data/rag/benchmark40_retrieval.json`
(vector), `_bm25.json`, `_rerank.json` (`scripts/25`, `scripts/28`). Evaluation:
`scripts/22_eval_benchmark_rag.py --bench40 [--retrieval bm25|rerank]`.
Full 6-config matrix + retriever ablation evaluated — see "Expanded benchmark"
section above. **Bottom line: J->C degradation at n=8 was a small-sample artifact
(+2% at n=40); sanitize's benefit does not transfer to fresh configs.**

### 2. Unified no-RAG baselines
Re-running no-RAG baselines (base + v4) on both benchmarks with the same script and
`max_new_tokens=512` to fix the earlier 400/512 token inconsistency. Done — confirms
@512 values (v4 8-pair C->J 0.3181 / J->C 0.6681; base 0.1970 / 0.3348).

### 3. Bootstrap confidence intervals
`scripts/26_bootstrap_ci.py` — per-task 2000-sample bootstrap 95% CIs on every
`rag_eval_*.json`. Example (`v4_k3`): generation 0.8172 [0.7532, 0.8778],
translation_c2j 1.0000 [1.0000, 1.0000] (n=14), translation_j2c 0.9395
[0.8621, 0.9955] (n=14), completion 0.9088 [0.8913, 0.9248], analysis 0.5430
[0.4829, 0.5999]. The J→C CI spanning ±0.07 (n=14) confirms the small-translation-
testset fragility identified earlier.

### 3b. Paired significance tests (40-pair benchmark, n=40)
`scripts/29_paired_significance.py` — paired bootstrap (10k) + permutation
(10k) + paired-t on RAG k=3 vs no-RAG, per direction:

| Model | Direction | Δmean | 95% CI | perm p | paired-t p |
|---|---|---|---|---|---|
| LoRA v4 | C→J | +0.120 | [+0.076, +0.166] | <0.0001 | <0.0001 |
| LoRA v4 | J→C | +0.010 | [-0.054, +0.075] | 0.79 | 0.76 |
| Base | C→J | +0.131 | [+0.113, +0.148] | <0.0001 | <0.0001 |
| Base | J→C | +0.144 | [+0.084, +0.206] | <0.0001 | <0.0001 |

C→J RAG gains are strongly significant for both models; v4 J→C shows **no
significant effect** (CI straddles 0, p=0.79) — confirming the n=8 -33% was noise;
base J→C gain is significant.

### 4. Analysis-task evaluation: replaced with parameter accuracy
The analysis gold references are degenerate (instruction restatements with
template-random parameters), so ROUGE/cosine/token-F1 all measure string-match
against them (see Case 2). `scripts/32_ana_param_accuracy.py` re-evaluates the
analysis task by **parameter accuracy**: (type, value) tuples extracted from the
source config vs from each prediction. Result: RAG recall 0.4098 vs no-RAG 0.1505,
F1 0.355 vs 0.169, paired-t p=5.4e-06 — **RAG significantly improves analysis
quality; the apparent degradation was an artifact.** Analysis results are reported
with this metric.

### 5. Retrieval quality (parameter-hit recall@k)
`scripts/34_retrieval_quality.py` — for each query, parameter F1 between retrieved
docs and the target config:

| Dataset | top-1 F1 | top-5 max F1 | hit-rate@5 | hit-rate@10 |
|---|---|---|---|---|
| main testset (n=583, same-family upper bound) | 0.570 | 0.637 | 0.732 | 0.762 |
| benchmark40 vector (n=80) | 0.301 | 0.452 | 1.000 | 1.000 |
| benchmark40 BM25 | 0.390 | 0.523 | 1.000 | 1.000 |
| benchmark40 reranker | 0.276 | 0.436 | 1.000 | 1.000 |

Hit-rate@5=1.0 on unseen benchmarks because the KB shares the generator's common
parameter space (e.g. remote-as 64512); top-1 F1 is the informative column: vector
0.30, BM25 0.39, rerank 0.28. BM25's better retrieval quality does not translate to
better translation (BM25 ≈ vector end-to-end); rerank is worst on both axes.

## Files

- `scripts/16_build_rag_index.py` — embed train set, build FAISS index
- `scripts/17_rag_retrieve.py` — top-10 vector retrieval cache for test samples
- `scripts/18_evaluate_rag.py` — RAG evaluation (`--k`, `--is-base`, `--no-rag`, `--retrieval vector|rerank|bm25`)
- `scripts/19_rag_retrieve_rerank.py` — two-stage retrieval (vector top-20 → bge-reranker-base top-10)
- `scripts/20_bm25_retrieve.py` — BM25 keyword retrieval cache (CPU-only)
- `scripts/21_benchmark_retrieve.py` — vector retrieval for NetworkConfigPro unseen benchmark
- `scripts/22_eval_benchmark_rag.py` — RAG eval on unseen benchmark (`--is-base`, `--k`, `--no-rag`, `--bench40`, `--retrieval`)
- `scripts/24_gen_benchmark_40.py` — generate 40 fresh benchmark pairs
- `scripts/25_benchmark40_retrieve.py` — vector retrieval for the 40-pair benchmark
- `scripts/26_bootstrap_ci.py` — bootstrap CIs per task
- `scripts/27_ana_semantic.py` — semantic similarity check for analysis samples
- `scripts/28_benchmark40_retrieve_alt.py` — BM25 / reranker retrieval for the 40-pair benchmark
- `logs/rag_eval_*.json` — per-sample results per config
- `data/rag/` — FAISS index, docs, retrieval caches (not in git, ~30MB)
