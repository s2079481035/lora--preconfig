"""
Step 8: Data Augmentation for Instruction Data
===============================================
对应论文 Section III-C: LLM-Based Data Augmentation
Prompt 设计四要素 (论文 Table I):
  1. Model Role (角色设定)
  2. Vendor Name (厂商名称)
  3. Configuration Attributes (配置属性)
  4. SOP (标准操作流程)

使用方式:
    # 本地模板增强（无需 API）
    python scripts/08_augment_data.py --method local --num-variants 3

    # DeepSeek 增强（需设置 DEEPSEEK_API_KEY）
    python scripts/08_augment_data.py --method llm --provider deepseek --num-variants 3

    # GPT-4 增强（需设置 OPENAI_API_KEY）
    python scripts/08_augment_data.py --method llm --provider openai --num-variants 3
"""

import os, re, json, logging, argparse, random
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "all_crawled.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "augmented" / "augmented_raw.json"

random.seed(42)


# ═══════════════════════════════════════════════════════════
# GPT-4 Prompt Templates (论文 Table I / II)
# ═══════════════════════════════════════════════════════════

GPT_PROMPTS = {
    "nl_paraphrase": {
        "system": (
            "You are a senior network engineer with 20+ years of experience in Cisco and Juniper "
            "network configuration. You excel at explaining network configurations in clear, varied natural language."
        ),
        "user": """Task: Generate {num_variants} different natural language descriptions that express the SAME network configuration intent as the example below.

Vendor: {vendor}
Protocol: {config_type}

Original description:
"{original_nl}"

Original configuration:
```
{original_config}
```

Requirements (SOP):
Step 1: Understand the configuration intent — what network behavior does this achieve?
Step 2: Write {num_variants} different paraphrases. Each must:
  - Express the exact same configuration intent
  - Use different sentence structure and word choice
  - Sound like a real network operator describing the task
  - Include necessary protocol details (AS numbers, IPs, interface names, etc.)
Step 3: Self-check — ensure each paraphrase would lead to the same configuration.

Output format: one paraphrase per line, starting with "V1:", "V2:", etc.""",
    },
    "config_variant": {
        "system": (
            "You are a senior network engineer with expertise in Cisco IOS and Juniper Junos syntax. "
            "You can write equivalent configurations using different valid command structures."
        ),
        "user": """Task: Generate {num_variants} syntactically different but semantically equivalent variants of the following network configuration.

Vendor: {vendor}
Protocol: {config_type}

Original configuration:
```
{original_config}
```

Original description:
"{original_nl}"

Requirements (SOP):
Step 1: Identify the core intent — what routing policy, access control, or protocol behavior is being configured?
Step 2: Generate {num_variants} variants. You may:
  - Reorder commands where the syntax permits
  - Use equivalent constructs (e.g., named vs numbered ACLs, different route-map organization)
  - Add/remove default parameters
  - Use different but equivalent parameter values
  - For Juniper: use "set" style vs hierarchical style where applicable
Step 3: Self-check each variant for:
  - Syntactic correctness (valid {vendor} commands)
  - Semantic equivalence (same network behavior as original)

CRITICAL: Output ONLY raw configuration syntax. Do NOT include any explanations, notes, descriptions, or natural language before, after, or within the configuration. No markdown formatting.

Output format: each variant clearly separated by "=== Variant N ===".""",
    },
}


# ═══════════════════════════════════════════════════════════
# Local Template-Based Augmentation
# ═══════════════════════════════════════════════════════════

NL_PARAPHRASE_TEMPLATES = {
    "bgp": [
        "Set up BGP on the device with AS {asn}, establishing a session with peer at {peer}.",
        "Configure BGP routing with local AS number {asn} and peer {peer}.",
        "Enable BGP process {asn} and peer with neighbor {peer}.",
        "Create a BGP configuration using AS {asn}, peering with {peer}.",
        "Establish BGP peering between AS {asn} and neighbor {peer}.",
        "Configure the router for BGP operation with AS {asn}, adding {peer} as a BGP neighbor.",
        "Implement BGP routing with autonomous system {asn} and peer address {peer}.",
    ],
    "ospf": [
        "Configure OSPF process {proc} and advertise the {network} network.",
        "Set up OSPF routing with process ID {proc}, enabling it on network {network}.",
        "Enable OSPF on the router with process {proc}, advertising {network}.",
        "Configure OSPF {proc} to distribute routes for the {network} subnet.",
        "Activate OSPF process {proc} and include network {network} in area 0.",
    ],
    "acl": [
        "Create an access list to permit HTTP traffic from the {subnet} subnet.",
        "Configure ACL {num} to allow HTTP access from {subnet} while denying everything else.",
        "Set up an extended ACL permitting TCP port 80 from source {subnet} to any destination.",
        "Implement access control allowing web traffic from {subnet} and blocking all other traffic.",
        "Define ACL rules to permit HTTP (port 80) originating from {subnet} to any destination.",
    ],
    "juniper_bgp": [
        "Set up BGP on the Juniper device with autonomous system {asn}.",
        "Configure Juniper BGP routing with AS {asn} and an external peer group.",
        "Enable BGP on the Juniper router using AS number {asn}.",
        "Establish BGP on JunOS with local AS {asn} and configure the external neighbor.",
    ],
}


def detect_config_type(config_text: str) -> str:
    cl = config_text.lower()
    if "router bgp" in cl or "protocols bgp" in cl:
        return "bgp"
    if "router ospf" in cl or "protocols ospf" in cl:
        return "ospf"
    if "access-list" in cl:
        return "acl"
    if "set protocols bgp" in cl:
        return "juniper_bgp"
    return "bgp"


def detect_vendor(config_text: str) -> str:
    if "set " in config_text or "protocols {" in config_text or "routing-options {" in config_text:
        return "Juniper"
    return "Cisco"


def _extract_params(nl_text: str, config_text: str) -> dict:
    params = {}
    m = re.search(r'AS\s*(\d+)', nl_text)
    if m:
        params["asn"] = m.group(1)
    m = re.search(r'(\d+\.\d+\.\d+\.\d+)', nl_text)
    if m:
        params["peer"] = m.group(1)
    m = re.search(r'router (bgp|ospf)\s*(\d+)', config_text)
    if m:
        params["asn_or_proc"] = m.group(2)
    m = re.search(r'process\s*(\d+)', nl_text, re.I)
    if m:
        params["proc"] = m.group(1)
    m = re.search(r'10\.(\d+)\.0\.0', nl_text)
    if m:
        params["proc"] = m.group(1)
        params["network"] = f"10.{m.group(1)}.0.0/24"
    m = re.search(r'ACL\s*(\d+)', nl_text, re.I)
    if m:
        params["num"] = m.group(1)
    m = re.search(r'10\.(\d+)\.0\.0', nl_text)
    if m:
        params["subnet"] = f"10.{m.group(1)}.0.0/24"
    return params


def _paraphrase_nl_local(sample: Dict, num_variants: int = 3) -> List[str]:
    nl = sample.get("input", "")
    cfg = sample.get("output", "")
    task = sample.get("task", "")
    cfg_type = detect_config_type(cfg)
    variants = []

    templates = NL_PARAPHRASE_TEMPLATES.get(cfg_type, NL_PARAPHRASE_TEMPLATES["bgp"])
    params = _extract_params(nl, cfg)

    random.shuffle(templates)
    for t in templates[:num_variants]:
        try:
            variant = t.format(**params)
            if variant != nl:
                variants.append(variant)
        except KeyError:
            continue

    while len(variants) < num_variants:
        variants.append(nl)

    return variants[:num_variants]


def augment_local(raw_data: List[Dict], num_variants: int = 3) -> List[Dict]:
    augmented = []
    for item in raw_data:
        nl = item.get("nl_text", "")
        cfg = item.get("config_text", "")
        source = item.get("source", "unknown")
        cfg_type = item.get("config_type", "bgp")
        vendor = detect_vendor(cfg)

        augmented.append(item)

        for v in range(num_variants):
            p = _extract_params(nl, cfg)
            templates = NL_PARAPHRASE_TEMPLATES.get(cfg_type, NL_PARAPHRASE_TEMPLATES["bgp"])
            random.shuffle(templates)
            if templates:
                try:
                    new_nl = templates[0].format(**p)
                except KeyError:
                    new_nl = nl
            else:
                new_nl = nl

            if new_nl and new_nl != nl:
                augmented.append({
                    "source": vendor.lower(),
                    "doc_type": "augmented_local",
                    "url": "augmented",
                    "nl_text": new_nl,
                    "config_text": cfg,
                    "config_type": cfg_type,
                    "metadata": {"original_index": len(augmented), "variant": v + 1},
                })

    logger.info(f"Local augmentation: {len(raw_data)} → {len(augmented)} samples")
    return augmented


def _create_llm_client(provider: str = "openai"):
    configs = {
        "openai": {"base_url": None, "key_var": "OPENAI_API_KEY", "model": "gpt-4"},
        "deepseek": {"base_url": "https://api.deepseek.com", "key_var": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    }
    cfg = configs.get(provider)
    if not cfg:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.environ.get(cfg["key_var"])
    if not api_key:
        logger.warning(f"No {cfg['key_var']} found, falling back to local augmentation")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, falling back to local augmentation")
        return None

    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg["model"]


def augment_llm(raw_data: List[Dict], num_variants: int = 3, provider: str = "openai") -> List[Dict]:
    result = _create_llm_client(provider)
    if result is None:
        return augment_local(raw_data, num_variants)
    client, model_name = result

    augmented = []
    augmented.extend(raw_data)

    for i, item in enumerate(raw_data):
        nl = item.get("nl_text", "")
        cfg = item.get("config_text", "")
        vendor = detect_vendor(cfg)
        cfg_type = item.get("config_type", "bgp")

        logger.info(f"[{i+1}/{len(raw_data)}] Augmenting with {provider} ({model_name})...")

        new_nls = _llm_generate(client, model_name, "nl_paraphrase",
                                vendor=vendor, config_type=cfg_type,
                                original_nl=nl, original_config=cfg,
                                num_variants=num_variants)

        new_cfgs = _llm_generate(client, model_name, "config_variant",
                                 vendor=vendor, config_type=cfg_type,
                                 original_nl=nl, original_config=cfg,
                                 num_variants=num_variants)

        for v in range(max(len(new_nls), len(new_cfgs))):
            aug_nl = new_nls[v] if v < len(new_nls) else nl
            aug_cfg = new_cfgs[v] if v < len(new_cfgs) else cfg
            augmented.append({
                "source": vendor.lower(),
                "doc_type": f"augmented_{provider}",
                "url": "augmented",
                "nl_text": aug_nl,
                "config_text": aug_cfg,
                "config_type": cfg_type,
                "metadata": {"original_index": i, "variant": v + 1},
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(raw_data)}")

    logger.info(f"{provider} augmentation: {len(raw_data)} \u2192 {len(augmented)} samples")
    return augmented


def _llm_generate(client, model_name: str, prompt_key: str, **kwargs) -> List[str]:
    template = GPT_PROMPTS[prompt_key]
    user_prompt = template["user"].format(**kwargs)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content

        # Parse V1:/V2: format (nl_paraphrase)
        results = re.findall(r'^V\d+:\s*(.+)$', text, re.MULTILINE)

        # Parse === Variant N === format (config_variant)
        if not results:
            sections = re.split(r'^={3,}\s*Variant\s*\d+\s*={3,}', text, flags=re.MULTILINE)
            if len(sections) > 1:
                results = [s.strip() for s in sections[1:] if s.strip()]
            else:
                # No sections found, try splitting on ===
                sections = re.split(r'^={3,}', text, flags=re.MULTILINE)
                results = [s.strip() for s in sections if s.strip()]

        # Last resort: use the entire text as a single result
        if not results:
            results = [text.strip()]

        return [r.strip() for r in results[:kwargs.get("num_variants", 3)]]
    except Exception as e:
        logger.error(f"LLM generation failed ({model_name}): {e}")
        return []


def recreate_instruction_data(augmented_raw: List[Dict]):
    """Recreate instruction-format data from augmented raw data."""
    samples = []
    for item in augmented_raw:
        nl = item.get("nl_text", "")
        cfg = item.get("config_text", "")
        if len(nl) < 5 or len(cfg) < 5:
            continue
        samples.append({
            "instruction": "Generate the network configuration for the following requirement:",
            "input": nl,
            "output": cfg,
            "task": "config_generation",
        })
        samples.append({
            "instruction": "Analyze the following network configuration and describe its functionality in natural language:",
            "input": cfg,
            "output": nl,
            "task": "config_analysis",
        })

    random.shuffle(samples)
    out_path = PROJECT_ROOT / "data" / "processed" / "train_data_augmented.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Created {len(samples)} instruction samples → {out_path}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["local", "llm"], default="local")
    parser.add_argument("--provider", choices=["openai", "deepseek"], default="deepseek",
                        help="LLM provider (used when --method=llm)")
    parser.add_argument("--num-variants", type=int, default=3, help="Variants per sample")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    raw_path = Path(args.input)
    if not raw_path.exists():
        logger.error(f"Input not found: {raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} raw samples")

    if args.method == "local":
        augmented = augment_local(raw_data, args.num_variants)
    else:
        augmented = augment_llm(raw_data, args.num_variants, args.provider)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved augmented data → {args.output}")

    recreate_instruction_data(augmented)


if __name__ == "__main__":
    main()
