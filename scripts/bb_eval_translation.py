"""
Translation benchmark evaluation on NetworkConfigPro pairs.
在 parallel pairs 上评估模型的双向翻译 (Cisco <-> Juniper)。
"""

import json, logging, sys, torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.configbleu import compute_all_metrics


TRANSLATE_PROMPTS = {
    "c2j": """<|im_start|>system
You are a senior network engineer with 20+ years of experience in both Cisco IOS and Juniper Junos syntax. Translate the Cisco IOS configuration to Juniper Junos while preserving functional equivalence.<|im_end|>
<|im_start|>user
Translate the following Cisco IOS configuration to Juniper Junos format:

{config}

Requirements:
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

Requirements:
- Output MUST be valid Cisco IOS flat command format
- Preserve all routing policies, access control rules, and interface settings
- Use functionally equivalent Cisco constructs
- Output ONLY the translated configuration. Do NOT include any explanations.<|im_end|>
<|im_start|>assistant
""",
}


def evaluate_translation(model_path: str, pairs: list, max_new_tokens: int = 400, is_base: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_base:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.eval()

    name = Path(model_path).name
    results = {"c2j": [], "j2c": []}

    for p in pairs:
        for direction, (key, src_cfg, ref_cfg) in [
            ("c2j", ("c2j", p["cisco"], p["juniper"])),
            ("j2c", ("j2c", p["juniper"], p["cisco"])),
        ]:
            prompt = TRANSLATE_PROMPTS[direction].format(config=src_cfg)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            metrics = compute_all_metrics(pred, ref_cfg)
            results[key].append({
                "sample": p["sample"],
                "hostname": p["hostname"],
                "reference": ref_cfg,
                "prediction": pred,
                **{k: metrics[k] for k in ["config_bleu", "bleu", "bleu_weight", "match_syn", "rouge_l", "meteor"]},
            })
            logger.info(f"[{direction}] {p['hostname']}: ConfigBLEU={metrics['config_bleu']:.4f}")

    # summary
    out = PROJECT_ROOT / "logs" / f"translation_benchmark_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for key in ["c2j", "j2c"]:
        items = results[key]
        n = len(items)
        avg = {k: sum(i[k] for i in items) / n for k in ["config_bleu", "bleu", "rouge_l", "meteor", "match_syn"]}
        logger.info(f"\n=== {key} ({n} samples) ===")
        for k, v in avg.items():
            logger.info(f"  {k}: {v:.4f}")
    logger.info(f"Saved → {out}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", default=str(PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "pairs.json"))
    parser.add_argument("--is-base", action="store_true", help="基座模型 (非 PEFT), 用模型名加载")
    args = parser.parse_args()
    pairs = json.load(open(args.pairs))
    evaluate_translation(args.model, pairs, is_base=args.is_base)