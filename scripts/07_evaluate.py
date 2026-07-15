"""
Step 7: Model Evaluation (Hold-out + ConfigBLEU)
==================================================
评估微调后的模型在测试集上的 ConfigBLEU 指标。
"""

import os, sys, json, logging, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.configbleu import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "preconfig-finetuned"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.json"


TASK_CONFIG_OUTPUT = {"config_generation", "config_translation_c2j", "config_translation_j2c", "config_completion"}
TASK_NL_OUTPUT = {"config_analysis"}


def build_prompt(sample):
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    return f"<|im_start|>system\nYou are a network configuration expert. Generate accurate and syntactically correct network configurations.<|im_end|>\n<|im_start|>user\n{instruction}\n\n{inp}<|im_end|>\n<|im_start|>assistant\n"


def load_model_and_tokenizer(model_path):
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(MODEL_DIR))
    parser.add_argument("--data", type=str, default=str(TEST_DATA_PATH))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test samples from {args.data}")

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    TASK_BUCKET = {
        "config_generation": "generation",
        "config_translation_c2j": "translation_c2j",
        "config_translation_j2c": "translation_j2c",
        "config_completion": "completion",
    }
    results = {"generation": [], "analysis": [], "translation_c2j": [], "translation_j2c": [], "completion": []}
    for i, sample in enumerate(test_data):
        task = sample.get("task", "config_generation")
        prompt = build_prompt(sample)
        reference = sample.get("output", "")

        logger.info(f"[{i+1}/{len(test_data)}] Evaluating {task}...")
        prediction = generate(model, tokenizer, prompt)

        if i < 3:
            logger.info(f"  [DEBUG] Input[:80]: {sample.get('input', '')[:80]}")
            logger.info(f"  [DEBUG] Reference[:80]: {reference[:80]}")
            logger.info(f"  [DEBUG] Prediction[:80]: {prediction[:80]}")

        if task in TASK_CONFIG_OUTPUT:
            metrics = compute_all_metrics(prediction, reference)
            bucket = TASK_BUCKET.get(task, "generation")
            results[bucket].append({"sample": i, "reference": reference, "prediction": prediction, **metrics})
            logger.info(f"  ConfigBLEU={metrics['config_bleu']:.4f}, BLEU={metrics['bleu']:.4f}")
        else:
            from scripts.configbleu import compute_rouge_l, compute_meteor
            rouge = compute_rouge_l(prediction, reference)
            meteor = compute_meteor(prediction, reference)
            results["analysis"].append({"sample": i, "reference": reference, "prediction": prediction, "rouge_l": rouge, "meteor": meteor})
            logger.info(f"  ROUGE-L={rouge:.4f}, METEOR={meteor:.4f}")

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(test_data)}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    for bucket_name, bucket_label in [
        ("generation", "Config Generation"),
        ("translation_c2j", "C->J Translation"),
        ("translation_j2c", "J->C Translation"),
        ("completion", "Config Completion"),
        ("analysis", "Config Analysis"),
    ]:
        bucket = results[bucket_name]
        if bucket:
            if bucket_name == "analysis":
                metrics = {k: [r[k] for r in bucket] for k in ["rouge_l", "meteor"]}
            else:
                metrics = {k: [r[k] for r in bucket] for k in ["config_bleu", "bleu", "rouge_l", "meteor"]}
            logger.info(f"\n{bucket_label} ({len(bucket)} samples):")
            for k, v in metrics.items():
                logger.info(f"  Average {k}: {sum(v)/len(v):.4f}")

    output_path = PROJECT_ROOT / "logs" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    evaluate()
