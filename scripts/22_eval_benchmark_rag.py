"""
Step 22: Evaluate RAG on NetworkConfigPro unseen benchmark
===========================================================
对 8 对全新配置 (不在知识库) 评估 RAG 注入效果。
对比: 无 RAG vs RAG k=3, 模型: 基座 / LoRA v4。
方向: c2j (cisco->juniper), j2c (juniper->cisco)。

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 venv/bin/python scripts/22_eval_benchmark_rag.py \
      --model-path models/qwen-lora-multitask-v4 [--is-base] [--k 3] [--no-rag]
"""

import json, logging, argparse, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.configbleu import compute_all_metrics
from scripts.sanitize_ref import sanitize_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "pairs.json"
RETRIEVAL = PROJECT_ROOT / "data" / "rag" / "benchmark_retrieval.json"

SYSTEM_PROMPT = ("You are a network configuration expert. Generate accurate and "
                 "syntactically correct network configurations. Use the reference "
                 "configurations below as guidance for structure and syntax when they "
                 "are relevant.")


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


TRANSLATE_PROMPTS = {
    "c2j": """<|im_start|>system
You are a senior network engineer with 20+ years of experience in both Cisco IOS and Juniper Junos syntax. Translate the Cisco IOS configuration to Juniper Junos while preserving functional equivalence.<|im_end|>
<|im_start|>user
Translate the following Cisco IOS configuration to Juniper Junos format:

{config}

{refs}Requirements:
- Output MUST be valid Juniper Junos hierarchical format (curly braces)
- Preserve all routing policies, access control rules, and interface settings
- Use functionally equivalent Juniper constructs
- Output ONLY the translated configuration. Do NOT include any explanations.<|im_end|>
<|im_start|>assistant
""",
    "j2c": """<|im_start|>system
You are a senior network engineer with 20+ years of experience in both Cisco IOS and Juniper Junos syntax. Translate the Juniper Junos configuration to Cisco IOS while preserving functional equivalence.<|im_end|>
<|im_start|>user
Translate the following Juniper Junos configuration to Cisco IOS format:

{config}

{refs}Requirements:
- Output MUST be valid Cisco IOS flat command format
- Preserve all routing policies, access control rules, and interface settings
- Use functionally equivalent Cisco constructs
- Output ONLY the translated configuration. Do NOT include any explanations.<|im_end|>
<|im_start|>assistant
""",
}


def build_prompt(direction, src_cfg, refs, k, sanitize=False):
    ref_block = ""
    if refs:
        refs_ = [sanitize_config(r["doc_text"]) if sanitize else r["doc_text"] for r in refs[:k]]
        ref_block = ("Reference configurations (retrieved as similar examples "
                     "from the knowledge base):\n"
                     + "\n\n".join(refs_) + "\n\n")
    return TRANSLATE_PROMPTS[direction].format(config=src_cfg, refs=ref_block)


def generate(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--is-base", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--sanitize", action="store_true", help="参考配置参数清洗(占位符化)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    pairs = json.load(open(PAIRS, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL, encoding="utf-8"))
    model, tokenizer = load_model(args.model_path, args.is_base)

    results = {"c2j": [], "j2c": []}
    for p in pairs:
        ret = next(r for r in retrieval if r["sample"] == p["sample"])
        for direction, src_key, ref_key in [
            ("c2j", "cisco", "juniper"),
            ("j2c", "juniper", "cisco"),
        ]:
            src_cfg = p[src_key]
            ref_cfg = p[ref_key]
            refs = ret["hits"][direction] if not args.no_rag else []
            prompt = build_prompt(direction, src_cfg, refs, args.k, sanitize=args.sanitize)
            pred = generate(model, tokenizer, prompt)
            m = compute_all_metrics(pred, ref_cfg)
            results[direction].append({
                "sample": p["sample"], "scenario": p["scenario"],
                "prediction": pred, "reference": ref_cfg, **m,
            })
            logger.info(f"[{p['sample']}] {direction} ConfigBLEU={m['config_bleu']:.4f}")

    model_name = Path(args.model_path).name if Path(args.model_path).exists() else args.model_path.replace("/", "_")
    mode = "norag" if args.no_rag else f"rag_k{args.k}"
    tag = args.tag or f"benchmark_{model_name}_{mode}"
    out = PROJECT_ROOT / "logs" / f"{tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved → {out}")

    for direction in ["c2j", "j2c"]:
        avg = sum(r["config_bleu"] for r in results[direction]) / len(results[direction])
        logger.info(f"{direction}: avg ConfigBLEU={avg:.4f} (n={len(results[direction])})")


if __name__ == "__main__":
    main()