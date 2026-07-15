"""
Step 10: Multi-Task Data Generation (Translation + Completion)
==============================================================
对应论文 Table II / Figure 1 中的四个任务：
  1. config_generation (NL -> Config)
  2. config_analysis (Config -> NL)
  3. config_translation (Cisco <-> Juniper)
  4. config_completion (Partial -> Complete)

用法:
    # 只创建 completion 数据（免费）
    python scripts/10_create_multitask_data.py --no-translation

    # Completion + DeepSeek translation（需 DEEPSEEK_API_KEY）
    python scripts/10_create_multitask_data.py --translation-provider deepseek

    # 全量：原始数据 + 增强数据 + 翻译 + 补全 → merge 到一个训练集
    python scripts/10_create_multitask_data.py --merge-enhanced
"""

import os, re, json, logging, argparse, random, sys
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "all_crawled.json"
AUGMENTED_RAW_PATH = PROJECT_ROOT / "data" / "augmented" / "augmented_raw.json"
ENHANCED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_data_augmented.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "train_data_multitask.json"

# ═══════════════════════════════════════════════════════════
# 1. Completion Data (synthetic, no API needed)
# ═══════════════════════════════════════════════════════════

def create_completion_data(raw_data: List[Dict], samples_per_config: int = 2) -> List[Dict]:
    """从完整配置中截断生成补全任务数据。"""
    results = []
    min_lines = 6  # 至少需要 6 行才能截断

    for item in raw_data:
        cfg = item.get("config_text", "")
        lines = [l.rstrip() for l in cfg.split("\n") if l.strip()]
        if len(lines) < min_lines:
            continue

        for _ in range(samples_per_config):
            # 保留 30%-70% 的行
            split = random.randint(max(2, len(lines) // 3), len(lines) * 7 // 10)
            prefix = "\n".join(lines[:split])
            full = "\n".join(lines)

            results.append({
                "instruction": "Complete the following incomplete network configuration with the missing lines:",
                "input": prefix,
                "output": full,
                "task": "config_completion",
            })

    logger.info(f"Completion: {len(results)} samples")
    return results


# ═══════════════════════════════════════════════════════════
# 2. Translation Data (via DeepSeek/OpenAI)
# ═══════════════════════════════════════════════════════════

DETECT_VENDOR_RE = re.compile(r'(router\s+bgp|router\s+ospf|access-list|ip\s+|interface\s)', re.I)
DETECT_JUNIPER_RE = re.compile(r'(set\s+|protocols\s+\{|routing-options\s+\{|policy-options\s+)', re.I)


def detect_vendor(config: str) -> str:
    if DETECT_JUNIPER_RE.search(config):
        return "Juniper"
    if DETECT_VENDOR_RE.search(config) or "!" in config or "hostname" in config:
        return "Cisco"
    return "Cisco"


TRANSLATION_SYSTEM_PROMPT = (
    "You are a senior network engineer with 20+ years of experience in both Cisco IOS and Juniper Junos syntax. "
    "You can accurately translate configurations between the two vendor formats while preserving functional equivalence."
)

TRANSLATION_PROMPTS = {
    "cisco_to_juniper": """Translate the following Cisco IOS configuration to Juniper Junos format.

Cisco Configuration:
```
{config}
```

Requirements:
- Output MUST be valid Juniper Junos hierarchical format (curly braces)
- Preserve all routing policies, access control rules, and interface settings
- Use functionally equivalent Juniper constructs

CRITICAL: Output ONLY the translated configuration. Do NOT include any explanations, notes, or natural language.""",

    "juniper_to_cisco": """Translate the following Juniper Junos configuration to Cisco IOS format.

Juniper Configuration:
```
{config}
```

Requirements:
- Output MUST be valid Cisco IOS flat command format
- Preserve all routing policies, access control rules, and interface settings
- Use functionally equivalent Cisco constructs

CRITICAL: Output ONLY the translated configuration. Do NOT include any explanations, notes, or natural language.""",
}


def _create_llm_client(provider: str = "deepseek"):
    configs = {
        "openai": {"base_url": None, "key_var": "OPENAI_API_KEY", "model": "gpt-4"},
        "deepseek": {"base_url": "https://api.deepseek.com", "key_var": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    }
    cfg = configs.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.environ.get(cfg["key_var"])
    if not api_key:
        logger.warning(f"No {cfg['key_var']} found, falling back to mock translation")
        return None, None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, falling back to mock translation")
        return None, None

    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg["model"]


def _llm_translate(client, model: str, direction: str, config: str) -> Optional[str]:
    prompt = TRANSLATION_PROMPTS[direction].format(config=config)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content.strip()
        # 移除可能的 markdown 围栏
        text = re.sub(r'^```[\w]*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


def create_translation_data(raw_data: List[Dict], provider: str = "deepseek") -> List[Dict]:
    """通过 LLM 生成翻译数据。"""
    client, model = _create_llm_client(provider)
    has_llm = client is not None

    results = []
    cisco_configs = [item for item in raw_data if detect_vendor(item.get("config_text", "")) == "Cisco"]
    juniper_configs = [item for item in raw_data if detect_vendor(item.get("config_text", "")) == "Juniper"]

    logger.info(f"Cisco configs: {len(cisco_configs)}, Juniper configs: {len(juniper_configs)}")

    # Cisco → Juniper
    for i, item in enumerate(cisco_configs):
        cfg = item.get("config_text", "")
        if len(cfg) < 10:
            continue
        if has_llm:
            translated = _llm_translate(client, model, "cisco_to_juniper", cfg)
            if not translated:
                continue
        else:
            continue  # 不生成 mock 数据

        results.append({
            "instruction": "Translate the following Cisco IOS configuration to Juniper Junos format:",
            "input": cfg,
            "output": translated,
            "task": "config_translation_c2j",
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  C2J progress: {i+1}/{len(cisco_configs)}")

    # Juniper → Cisco
    for i, item in enumerate(juniper_configs):
        cfg = item.get("config_text", "")
        if len(cfg) < 10:
            continue
        if has_llm:
            translated = _llm_translate(client, model, "juniper_to_cisco", cfg)
            if not translated:
                continue
        else:
            continue

        results.append({
            "instruction": "Translate the following Juniper Junos configuration to Cisco IOS format:",
            "input": cfg,
            "output": translated,
            "task": "config_translation_j2c",
        })
        if (i + 1) % 10 == 0:
            logger.info(f"  J2C progress: {i+1}/{len(juniper_configs)}")

    logger.info(f"Translation: {len(results)} samples")
    return results


# ═══════════════════════════════════════════════════════════
# 3. Merge
# ═══════════════════════════════════════════════════════════

def load_json(path: Path) -> List[Dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_existing_instruction_data() -> List[Dict]:
    """Load existing generation + analysis instruction data."""
    samples = load_json(ENHANCED_DATA_PATH)
    if samples:
        logger.info(f"Loaded {len(samples)} existing instruction samples")
        return samples
    return []


def load_raw_data() -> List[Dict]:
    return load_json(RAW_DATA_PATH)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-task training data")
    parser.add_argument("--translation-provider", type=str, default=None,
                        choices=["openai", "deepseek", None],
                        help="LLM provider for translation (omit to skip translation)")
    parser.add_argument("--no-translation", action="store_true",
                        help="Skip translation data generation")
    parser.add_argument("--completion-samples", type=int, default=2,
                        help="Completion samples per config (default: 2)")
    parser.add_argument("--merge-enhanced", action="store_true",
                        help="Merge with existing enhanced instruction data")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--test-output", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"))
    args = parser.parse_args()

    all_samples = []

    # 1. Load existing generation + analysis data
    if args.merge_enhanced:
        existing = load_existing_instruction_data()
        all_samples.extend(existing)
        logger.info(f"Added {len(existing)} existing generation/analysis samples")

    # 2. Load raw data
    raw_data = load_raw_data()
    if not raw_data and not all_samples:
        logger.error(f"No data found. Check {RAW_DATA_PATH}")
        return

    # 3. Create completion data (always, free)
    completion = create_completion_data(raw_data, args.completion_samples)
    all_samples.extend(completion)
    logger.info(f"Added {len(completion)} completion samples")

    # 4. Create translation data (if requested)
    if not args.no_translation and args.translation_provider:
        translation = create_translation_data(raw_data, args.translation_provider)
        all_samples.extend(translation)
        logger.info(f"Added {len(translation)} translation samples")

    # 5. Shuffle and split test set
    random.shuffle(all_samples)

    if args.test_ratio > 0 and all_samples:
        split_idx = int(len(all_samples) * (1 - args.test_ratio))
        train = all_samples[:split_idx]
        test = all_samples[split_idx:]

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        logger.info(f"Train: {len(train)} samples → {out_path}")

        test_path = Path(args.test_output)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        logger.info(f"Test: {len(test)} samples → {test_path}")

        # Print task distribution
        from collections import Counter
        train_task_dist = Counter(s["task"] for s in train)
        test_task_dist = Counter(s["task"] for s in test)
        logger.info(f"Train task distribution: {dict(train_task_dist)}")
        logger.info(f"Test task distribution: {dict(test_task_dist)}")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(all_samples)} samples → {out_path}")


if __name__ == "__main__":
    main()
