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

## Files

- `scripts/16_build_rag_index.py` — embed train set, build FAISS index
- `scripts/17_rag_retrieve.py` — top-10 retrieval cache for test samples
- `scripts/18_evaluate_rag.py` — RAG evaluation (reference injection, `--k`, `--is-base`, `--no-rag`)
- `logs/rag_eval_*.json` — per-sample results per config
- `data/rag/` — FAISS index, docs, retrieval cache (not in git, ~30MB)
