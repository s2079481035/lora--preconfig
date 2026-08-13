"""
Step 18: RAG Evaluation
=======================
在 583 条测试集上评估 RAG 增强的模型。
读取预计算的检索结果 (data/rag/test_retrieval.json), 按 k 截取 top-k
配置注入 prompt, 复用 ConfigBLEU 指标。

用法 (venv):
  env HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/18_evaluate_rag.py \
      --model-path models/qwen-lora-multitask-v4 --k 3 \
      --tag v4_k3 [--is-base]
"""

import os, sys, json, logging, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.configbleu import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"
RETRIEVAL = PROJECT_ROOT / "data" / "rag" / "test_retrieval.json"

TASK_CONFIG_OUTPUT = {"config_generation", "config_translation_c2j", "config_translation_j2c", "config_completion"}
TASK_NL_OUTPUT = {"config_analysis"}


def build_rag_prompt(sample, hits, k):
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    refs = "\n\n".join(h["doc_text"] for h in hits[:k])
    return (
        "<|im_start|>system\n"
        "You are a network configuration expert. Generate accurate and syntactically "
        "correct network configurations. Use the reference configurations below as "
        "guidance for structure and syntax when they are relevant.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{inp}\n\n"
        "Reference configurations (from a network configuration knowledge base, "
        "retrieved as similar examples):\n"
        f"{refs}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_plain_prompt(sample):
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    return (
        "<|im_start|>system\n"
        "You are a network configuration expert. Generate accurate and syntactically "
        "correct network configurations.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\n{inp}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_model(model_path, is_base):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_base:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--k", type=int, default=3, help="top-k 检索结果注入")
    parser.add_argument("--tag", default=None, help="输出文件名标记")
    parser.add_argument("--is-base", action="store_true", help="基座模型")
    parser.add_argument("--no-rag", action="store_true", help="跑无 RAG baseline")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    test_data = json.load(open(TEST_DATA, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL, encoding="utf-8"))
    logger.info(f"Loaded {len(test_data)} test samples, {len(retrieval)} retrieval results")

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        retrieval = retrieval[:args.max_samples]

    model, tokenizer = load_model(args.model_path, args.is_base)

    TASK_BUCKET = {
        "config_generation": "generation",
        "config_translation_c2j": "translation_c2j",
        "config_translation_j2c": "translation_j2c",
        "config_completion": "completion",
    }
    results = {"generation": [], "analysis": [], "translation_c2j": [], "translation_j2c": [], "completion": []}

    for i, sample in enumerate(test_data):
        task = sample.get("task", "config_generation")
        reference = sample.get("output", "")
        hits = retrieval[i]["hits"] if i < len(retrieval) else []

        if args.no_rag:
            prompt = build_plain_prompt(sample)
        else:
            prompt = build_rag_prompt(sample, hits, args.k)

        prediction = generate(model, tokenizer, prompt)

        if task in TASK_CONFIG_OUTPUT:
            metrics = compute_all_metrics(prediction, reference)
            bucket = TASK_BUCKET.get(task, "generation")
            results[bucket].append({"sample": i, "reference": reference, "prediction": prediction, **metrics})
            if (i + 1) % 20 == 0:
                logger.info(f"[{i+1}/{len(test_data)}] {task} ConfigBLEU={metrics['config_bleu']:.4f}")
        else:
            from scripts.configbleu import compute_rouge_l, compute_meteor
            rouge = compute_rouge_l(prediction, reference)
            meteor = compute_meteor(prediction, reference)
            results["analysis"].append({"sample": i, "reference": reference, "prediction": prediction,
                                        "rouge_l": rouge, "meteor": meteor})
            if (i + 1) % 20 == 0:
                logger.info(f"[{i+1}/{len(test_data)}] {task} ROUGE-L={rouge:.4f}")

    model_name = Path(args.model_path).name if Path(args.model_path).exists() else args.model_path.replace("/", "_")
    tag = args.tag or f"{model_name}_k{args.k}"
    out_path = PROJECT_ROOT / "logs" / f"rag_eval_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {out_path}")

    for bucket_name, label in [("generation", "Config Generation"), ("translation_c2j", "C->J"),
                               ("translation_j2c", "J->C"), ("completion", "Completion")]:
        bucket = results[bucket_name]
        if bucket:
            cb = sum(r["config_bleu"] for r in bucket) / len(bucket)
            logger.info(f"{label}: ConfigBLEU={cb:.4f} (n={len(bucket)})")
    bucket = results["analysis"]
    if bucket:
        r = sum(x["rouge_l"] for x in bucket) / len(bucket)
        logger.info(f"Analysis: ROUGE-L={r:.4f} (n={len(bucket)})")


if __name__ == "__main__":
    main()
