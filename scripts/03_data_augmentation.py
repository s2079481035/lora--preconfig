"""
Step 3: LLM-Based Data Augmentation
=====================================
对应论文 Section III-C: LLM-Based Data Augmentation
使用 Prompt Engineering 引导 GPT-4 对已有数据做同义改写。

Prompt 设计四要素 (论文 Table I):
  1. Vendor Name (厂商名称)
  2. Configuration Attributes (配置属性)
  3. Model Role (角色设定)
  4. SOP (标准操作流程)

使用方式:
    python scripts/03_data_augmentation.py --data data/processed/forum_configs_extracted.json
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "augmented"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# Prompt Templates (论文 Table I)
# ═══════════════════════════════════════════════════════════

AUGMENTATION_PROMPTS = {
    "config_generation": {
        "system": """You are a senior network engineer with 20+ years of experience in configuring Cisco and Juniper routers. You are an expert in BGP, OSPF, static routes, ACLs, and route policies.""",
        "user_template": """Task: Generate {num_variants} semantically equivalent but syntactically different variants of the following network configuration.

Vendor: {vendor}
Protocol/Feature: {config_type}

Original Configuration:
```
{original_config}
```

Requirements (SOP):
Step 1: Analyze the configuration intent - understand what routing policy or behavior this configuration implements.
Step 2: Generate variants that achieve the same functionality but use different:
  - Command ordering (where syntax permits)
  - Equivalent parameter values (e.g., different but equivalent ACL numbering)
  - Equivalent syntax constructs (e.g., using named vs numbered ACLs)
Step 3: Self-check each variant for:
  - Syntactic correctness (valid {vendor} commands)
  - Semantic equivalence (same behavior as original)

Please provide {num_variants} variants, each clearly separated:""",
    },
    "config_analysis": {
        "system": """You are a network operations expert who excels at explaining complex configurations in plain English.""",
        "user_template": """Task: Generate {num_variants} different natural language descriptions of the following network configuration. Each description should explain the same functionality from a different perspective.

Vendor: {vendor}
Configuration:
```
{original_config}
```

Requirements:
- Description 1: Focus on the routing/forwarding behavior
- Description 2: Focus on security implications
- Description 3: Focus on the network design rationale

Provide {num_variants} descriptions:""",
    },
    "config_completion": {
        "system": """You are a network configuration expert who can infer missing configuration elements from context.""",
        "user_template": """Task: Given the following partial network configuration, generate {num_variants} different completions that are all valid and consistent with the existing context.

Vendor: {vendor}
Partial Configuration:
```
{partial_config}
```

Requirements:
- Each completion should add 2-5 lines that logically follow the existing configuration
- All completions must be syntactically valid {vendor} configuration
- Different completions should explore different valid possibilities

Provide {num_variants} completions:""",
    },
}


class DataAugmentor:
    """
    论文 Section III-C: 配置数据增强
    使用 Prompt Engineering 引导 LLM 生成高质量配置变体。
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def detect_vendor(self, config_text: str) -> str:
        """Detect if config is Cisco or Juniper."""
        juniper_indicators = ["set ", "protocols {", "routing-options {", "interfaces {"]
        cisco_indicators = ["router bgp", "router ospf", "interface GigabitEthernet",
                           "access-list", "route-map", "ip route"]

        config_lower = config_text.lower()
        juniper_score = sum(1 for ind in juniper_indicators if ind in config_lower)
        cisco_score = sum(1 for ind in cisco_indicators if ind in config_lower)

        return "Juniper" if juniper_score > cisco_score else "Cisco"

    def detect_config_type(self, config_text: str) -> str:
        """Detect the protocol/feature type."""
        config_lower = config_text.lower()
        if "router bgp" in config_lower or "protocols bgp" in config_lower:
            return "BGP"
        elif "router ospf" in config_lower or "protocols ospf" in config_lower:
            return "OSPF"
        elif "access-list" in config_lower or "firewall" in config_lower:
            return "ACL"
        elif "ip route" in config_lower or "routing-options static" in config_lower:
            return "Static Route"
        elif "route-map" in config_lower or "policy-options" in config_lower:
            return "Route Policy"
        return "General"

    def build_augmentation_prompt(
        self,
        config_text: str,
        task: str = "config_generation",
        num_variants: int = 5,
    ) -> Dict[str, str]:
        """
        构建论文 Table I 描述的 Prompt Template。
        包含四个要素: 角色设定 + 领域知识 + 任务指令 + SOP 约束
        """
        vendor = self.detect_vendor(config_text)
        config_type = self.detect_config_type(config_text)

        template = AUGMENTATION_PROMPTS.get(task, AUGMENTATION_PROMPTS["config_generation"])

        system_prompt = template["system"]
        user_prompt = template["user_template"].format(
            vendor=vendor,
            config_type=config_type,
            original_config=config_text,
            partial_config=config_text,
            num_variants=num_variants,
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
            "metadata": {
                "vendor": vendor,
                "config_type": config_type,
                "task": task,
                "num_variants": num_variants,
            },
        }

    def augment_single(self, config_text: str, task: str = "config_generation", num_variants: int = 5) -> List[str]:
        """Generate augmented variants for a single config."""
        prompt = self.build_augmentation_prompt(config_text, task, num_variants)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            raw_output = response.choices[0].message.content
            return self._parse_variants(raw_output)
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            return []

    def _parse_variants(self, raw_output: str) -> List[str]:
        """Parse LLM output into individual config variants."""
        # Try to split by common delimiters
        variants = re.split(r"\n*(?:Variant \d+|---+|```)\n*", raw_output)
        variants = [v.strip() for v in variants if v.strip() and len(v.strip()) > 20]

        # Remove code block markers
        cleaned = []
        for v in variants:
            v = v.replace("```", "").strip()
            if v and len(v) > 10:
                cleaned.append(v)
        return cleaned

    def augment_batch(
        self,
        configs: List[Dict],
        task: str = "config_generation",
        num_variants: int = 5,
        max_samples: int = 100,
    ) -> List[Dict]:
        """Augment a batch of configs."""
        augmented = []
        configs_to_process = configs[:max_samples]

        for i, item in enumerate(configs_to_process):
            config_text = item.get("config_text", "") or item.get("text", "")
            if not config_text:
                continue

            variants = self.augment_single(config_text, task, num_variants)
            for j, variant in enumerate(variants):
                augmented.append({
                    "original": config_text,
                    "augmented": variant,
                    "task": task,
                    "variant_index": j,
                    "source_index": i,
                })

            if (i + 1) % 10 == 0:
                logger.info(f"Augmented {i + 1}/{len(configs_to_process)} configs")

        return augmented

    def save_prompts_for_reference(self):
        """Save prompt templates for reference (论文 Table I)."""
        for task_name, template in AUGMENTATION_PROMPTS.items():
            prompt_doc = {
                "task": task_name,
                "system_prompt": template["system"],
                "user_template": template["user_template"],
                "design_principles": {
                    "role": "Senior network engineer / operations expert",
                    "domain_knowledge": "Vendor name, protocol type, config attributes",
                    "task_instruction": "Generate semantically equivalent variants",
                    "sop": "Analyze intent → Generate variants → Self-check syntax",
                },
            }
            output_path = PROMPTS_DIR / f"augmentation_prompt_{task_name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(prompt_doc, f, ensure_ascii=False, indent=2)

        logger.info(f"Prompt templates saved to {PROMPTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="PreConfig Data Augmentation")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data JSON")
    parser.add_argument("--task", type=str, default="config_generation",
                       choices=["config_generation", "config_analysis", "config_completion"])
    parser.add_argument("--num-variants", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--save-prompts", action="store_true", help="Save prompt templates")
    args = parser.parse_args()

    augmentor = DataAugmentor()

    # Save prompt templates
    if args.save_prompts:
        augmentor.save_prompts_for_reference()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Using sample data for demonstration...")

        # Sample data for testing
        sample_configs = [
            {
                "config_text": """router bgp 65000
 neighbor 192.168.1.1 remote-as 64512
 address-family ipv4 unicast
  network 10.0.0.0 mask 255.255.255.0
  neighbor 192.168.1.1 activate""",
            },
            {
                "config_text": """router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
 network 192.168.1.0 0.0.0.255 area 0
 default-information originate""",
            },
            {
                "config_text": """set protocols bgp group external-peers
 set protocols bgp group external-peers neighbor 192.168.1.1
 set protocols bgp group external-peers neighbor 192.168.1.1 peer-as 64512""",
            },
        ]
        configs = sample_configs
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            configs = json.load(f)

    # Run augmentation
    logger.info(f"Augmenting {min(len(configs), args.max_samples)} configs for task: {args.task}")
    augmented = augmentor.augment_batch(
        configs,
        task=args.task,
        num_variants=args.num_variants,
        max_samples=args.max_samples,
    )

    # Save results
    output_path = AUGMENTED_DIR / f"augmented_{args.task}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(augmented)} augmented samples to {output_path}")


if __name__ == "__main__":
    main()
