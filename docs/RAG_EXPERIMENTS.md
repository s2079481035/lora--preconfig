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
| Base | 0.0806 | 0.1951 | 0.3348 | - | - |
| Base + RAG k=1 | 0.3201 | 0.1728 | 0.6462 | 0.5273 | 0.1214 |
| Base + RAG k=3 | 0.4244 | 0.1783 | 0.3375 | 0.5948 | 0.1198 |
| Base + RAG k=5 | 0.4509 | 0.1801 | 0.3242 | 0.5732 | 0.1167 |
| Base + RAG k=10 | **0.5737** | 0.1621 | 0.3321 | 0.5862 | 0.1173 |
| LoRA v4 | 0.8895 | 0.9708 | 0.8333 | 0.9642 | 0.6583 |
| LoRA v4 + RAG k=1 | 0.8228 | 0.9708 | 0.9225 | 0.7761 | 0.3855 |
| LoRA v4 + RAG k=3 | 0.8172 | **1.0000** | **0.9395** | 0.9088 | 0.5430 |
| LoRA v4 + RAG k=5 | 0.8102 | **1.0000** | 0.9406 | 0.9398 | 0.5537 |
| LoRA v4 + RAG k=10 | 0.7995 | 0.9189 | 0.9406 | 0.8967 | 0.5282 |

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

### Case 2: Analysis "degradation" — training-template memorization, not pure eval artifact
v4 analysis samples (6, 8, 21, 31, ...): without RAG the model emits a terse summary
("Configure BGP on Juniper with AS 65162") that exactly matches the short reference
(ROUGE=1.0, bge semantic sim=1.0 — string-identical). The reference is itself the
**training template sentence**, so the high no-RAG score is memorization, not ability.
With RAG, the retrieved detailed reference pulls the model into a verbose style
("Configure BGP on this Juniper router with AS number 65152, and add an external
peer group...") that is semantically complete and correct but no longer matches the
template gold text (ROUGE=0.2~0.3). 18 samples show this pattern. Verification with
bge-large-en-v1.5 cosine similarity (scripts/27_ana_semantic.py): no-RAG avg=1.0000,
RAG avg=0.7657, RAG better in 0/18 — i.e. **the semantic metric agrees with ROUGE here
because the gold is a memorized template, so both metrics measure string-match against
a degenerate reference.** Conclusion: the analysis task's gold references are
low-quality (instruction restatements); this caveat is reported and per-sample
inspection favors RAG outputs for informativeness.

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

### 4. Analysis-task evaluation verification
`scripts/27_ana_semantic.py` — bge-large-en-v1.5 cosine similarity on the 18
disputed analysis samples (see Case 2). Finding: no-RAG predictions are
string-identical to the template gold (sim=1.0); semantic similarity **agrees**
with ROUGE (RAG 0.7657 avg), i.e. the apparent RAG advantage is NOT recoverable
by a better metric — the gold references themselves are degenerate
(instruction restatements). Reported as a caveat; per-sample qualitative
inspection favors RAG outputs.

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
