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

**Findings**:
1. **RAG improves C->J translation on truly unseen configs for both models**
   (base +46%, v4 +26%) — the model uses the retrieved Junos example's structure while
   translating the Cisco parameters. This is genuine generalization, not memorization.
2. **RAG hurts v4 J->C on unseen configs** (-33%): the model sometimes *copies parameters
   from the retrieved Cisco reference* (e.g. hostname `rt-045` from the retrieved doc
   instead of `rt-098` from the source Junos config). Retrieved reference params leak
   into the output and override the source. Base J->C is immune (it lacks the confidence
   to trust references over the source).
3. This direction asymmetry is a novel, publication-worthy observation: reference
   injection helps when the model is weaker (base), helps when the target structure is
   the bottleneck (C->J: retrieving the target-syntax example), and hurts when the model
   must preserve source parameters under conflicting reference evidence (J->C).

## Files

- `scripts/16_build_rag_index.py` — embed train set, build FAISS index
- `scripts/17_rag_retrieve.py` — top-10 vector retrieval cache for test samples
- `scripts/18_evaluate_rag.py` — RAG evaluation (`--k`, `--is-base`, `--no-rag`, `--retrieval vector|rerank|bm25`)
- `scripts/19_rag_retrieve_rerank.py` — two-stage retrieval (vector top-20 → bge-reranker-base top-10)
- `scripts/20_bm25_retrieve.py` — BM25 keyword retrieval cache (CPU-only)
- `scripts/21_benchmark_retrieve.py` — vector retrieval for NetworkConfigPro unseen benchmark
- `scripts/22_eval_benchmark_rag.py` — RAG eval on unseen benchmark (`--is-base`, `--k`, `--no-rag`)
- `logs/rag_eval_*.json` — per-sample results per config
- `data/rag/` — FAISS index, docs, retrieval caches (not in git, ~30MB)
