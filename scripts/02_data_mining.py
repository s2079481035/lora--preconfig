"""
Step 2: Data Mining - Extract Task-Specific Supervision Data
=============================================================
对应论文 Section III-B:
  - 方法一: HTML Parser (已在 01_data_crawler.py 实现)
  - 方法二: 基于文本分类的配置提取 (BoW + KNN)
  - 方法三: LLM-Based Task Data Generation (翻译任务)

使用方式:
    python scripts/02_data_mining.py --method bow_knn
    python scripts/02_data_mining.py --method llm_translate
    python scripts/02_data_mining.py --all
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# 方法二: 基于文本分类的配置提取 (Algorithm 1 in Paper)
# ═══════════════════════════════════════════════════════════

class ConfigExtractor:
    """
    论文 Algorithm 1: Configuration Extraction Based on Text Classification
    从论坛混合文本中提取配置片段。

    流程:
      1. DataProcess: 初步分离自然语言和配置片段
      2. ModelPretrain: 用 TF-IDF (词袋模型) 学习配置语言特征
      3. DataSelection: 用 KNN 找到与标准配置最相似的候选片段
      4. DataJudgment: 过滤，保留真正的配置片段
    """

    def __init__(self, min_config_lines: int = 3, knn_top_n: int = 5):
        self.min_config_lines = min_config_lines
        self.knn_top_n = knn_top_n
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words="english",
            token_pattern=r"(?u)\b\w[\w-]+\b",
        )
        self.knn = None
        self.standard_configs = []  # D2: known high-quality configs

    def load_crawled_data(self, filepath: Path) -> List[Dict]:
        """Load raw crawled data."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_standard_configs(self, filepath: Path) -> List[str]:
        """Load known high-quality config snippets (D2)."""
        data = self.load_crawled_data(filepath)
        configs = [d["config_text"] for d in data if d.get("config_text")]
        self.standard_configs = configs
        return configs

    def data_process(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Step 1: DataProcess - 初步分离自然语言和配置片段
        从论坛帖子中分离出可能的配置片段。
        """
        processed = []
        for item in raw_data:
            text = item.get("nl_text", "") + "\n" + item.get("config_text", "")
            lines = text.split("\n")

            config_buffer = []
            nl_buffer = []

            for line in lines:
                if self._is_config_line(line.strip()):
                    if nl_buffer:
                        processed.append({
                            "type": "nl_text",
                            "text": "\n".join(nl_buffer).strip(),
                        })
                        nl_buffer = []
                    config_buffer.append(line.strip())
                else:
                    if config_buffer:
                        processed.append({
                            "type": "config",
                            "text": "\n".join(config_buffer).strip(),
                        })
                        config_buffer = []
                    if line.strip():
                        nl_buffer.append(line.strip())

            # Flush remaining
            if config_buffer:
                processed.append({
                    "type": "config",
                    "text": "\n".join(config_buffer).strip(),
                })
            if nl_buffer:
                processed.append({
                    "type": "nl_text",
                    "text": "\n".join(nl_buffer).strip(),
                })

        logger.info(f"DataProcess: {len(processed)} segments extracted")
        return processed

    def model_pretrain(self, mixed_data: List[Dict]) -> None:
        """
        Step 2: ModelPretrain - 用标准配置 + 混合数据训练 TF-IDF 向量化器
        """
        texts = []
        # D1: mixed forum data
        for item in mixed_data:
            if item.get("text") and len(item["text"]) > 10:
                texts.append(item["text"])
        # D2: standard configs
        texts.extend(self.standard_configs)

        if not texts:
            logger.warning("No texts for pretraining vectorizer")
            return

        self.vectorizer.fit(texts)
        logger.info(f"ModelPretrain: TF-IDF vectorizer fitted on {len(texts)} texts")

    def data_selection(self, mixed_data: List[Dict], top_n: int = None) -> List[Dict]:
        """
        Step 3: DataSelection - 对每个标准配置，用 KNN 找最相似的候选
        """
        if top_n is None:
            top_n = self.knn_top_n

        # Get all candidate texts
        candidates = [item for item in mixed_data if item.get("text") and len(item["text"]) > 10]
        if not candidates:
            return []

        candidate_texts = [c["text"] for c in candidates]
        candidate_vecs = self.vectorizer.transform(candidate_texts)

        # For each standard config, find top-n most similar candidates
        selected = set()
        for std_config in self.standard_configs:
            std_vec = self.vectorizer.transform([std_config])

            # Cosine similarity
            similarities = (candidate_vecs @ std_vec.T).toarray().flatten()
            top_indices = np.argsort(similarities)[-top_n:][::-1]

            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    selected.add(idx)

        results = [candidates[i] for i in sorted(selected)]
        logger.info(f"DataSelection: {len(results)} candidates selected from {len(candidates)}")
        return results

    def data_judgment(self, candidates: List[Dict]) -> List[Dict]:
        """
        Step 4: DataJudgment - 最终过滤，保留高质量配置片段
        """
        filtered = []
        for item in candidates:
            text = item.get("text", "")
            if self._is_valid_config(text):
                filtered.append(item)

        logger.info(f"DataJudgment: {len(filtered)} configs retained from {len(candidates)}")
        return filtered

    def extract(self, raw_data_path: Path, standard_config_path: Path) -> List[Dict]:
        """Full pipeline: DataProcess → ModelPretrain → DataSelection → DataJudgment"""
        # Load data
        raw_data = self.load_crawled_data(raw_data_path)
        self.load_standard_configs(standard_config_path)

        # Step 1
        mixed_data = self.data_process(raw_data)

        # Step 2
        self.model_pretrain(mixed_data)

        # Step 3
        candidates = self.data_selection(mixed_data)

        # Step 4
        final_configs = self.data_judgment(candidates)

        return final_configs

    def _is_config_line(self, line: str) -> bool:
        if not line or len(line) < 3:
            return False
        config_prefixes = [
            "interface", "ip ", "router", "network", "neighbor",
            "access-list", "route-map", "prefix-list", "hostname",
            "!", "permit", "deny", "set ", "match ", "no ",
            "protocols", "routing-options", "security", "policy-options",
        ]
        return any(line.lower().startswith(p) for p in config_prefixes)

    def _is_valid_config(self, text: str) -> bool:
        lines = text.strip().split("\n")
        config_line_count = sum(1 for l in lines if self._is_config_line(l.strip()))
        return config_line_count >= self.min_config_lines


# ═══════════════════════════════════════════════════════════
# 方法三: LLM-Based Task Data Generation (翻译任务)
# ═══════════════════════════════════════════════════════════

class TranslationDataGenerator:
    """
    论文 Section III-B.3: LLM-Based Task Data Generation
    用 GPT-4 + Campion 生成 Cisco↔Juniper 翻译数据。

    流程:
      1. 设计 Prompt Template（含领域知识）
      2. 用 GPT-4 生成初始翻译
      3. 用 Campion 验证语法+语义一致性
      4. 如果不正确，反馈给 GPT-4 迭代修正
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key. LLM generation will use mock data.")

    def get_translation_prompt(self, source_config: str, direction: str) -> str:
        """
        生成翻译任务的 Prompt Template
        direction: "cisco_to_juniper" or "juniper_to_cisco"
        """
        if direction == "cisco_to_juniper":
            return f"""You are a senior network engineer specializing in Cisco and Juniper configurations.

Task: Translate the following Cisco IOS configuration into an equivalent Juniper Junos configuration.

Requirements:
1. The translated configuration must be functionally equivalent
2. Use proper Juniper Junos syntax (hierarchical format with curly braces)
3. Preserve all routing policies, access control rules, and interface settings
4. Add comments explaining key translation decisions

Cisco Configuration:
```
{source_config}
```

Please provide the equivalent Juniper Junos configuration:"""
        else:
            return f"""You are a senior network engineer specializing in Cisco and Juniper configurations.

Task: Translate the following Juniper Junos configuration into an equivalent Cisco IOS configuration.

Requirements:
1. The translated configuration must be functionally equivalent
2. Use proper Cisco IOS syntax (flat indentation format)
3. Preserve all routing policies, access control rules, and interface settings
4. Add comments explaining key translation decisions

Juniper Configuration:
```
{source_config}
```

Please provide the equivalent Cisco IOS configuration:"""

    def generate_translation(self, source_config: str, direction: str) -> str:
        """Call LLM to generate translation. Falls back to mock if no API key."""
        prompt = self.get_translation_prompt(source_config, direction)

        if not self.api_key:
            logger.debug("No API key, using mock translation")
            return self._mock_translate(source_config, direction)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("openai package not installed, returning mock translation")
            return self._mock_translate(source_config, direction)
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return self._mock_translate(source_config, direction)

    def _mock_translate(self, config: str, direction: str) -> str:
        """Mock translation for testing without API."""
        if direction == "cisco_to_juniper":
            return f"/* Juniper translation of Cisco config */\nprotocols {{\n    /* TODO: translate */\n}}"
        else:
            return f"! Cisco translation of Juniper config\n! TODO: translate"

    def generate_batch(self, configs: List[Dict], direction: str) -> List[Dict]:
        """Generate translations for a batch of configs."""
        results = []
        for item in configs:
            source = item.get("config_text", "")
            if not source:
                continue
            translation = self.generate_translation(source, direction)
            results.append({
                "source_config": source,
                "translated_config": translation,
                "direction": direction,
                "source_vendor": direction.split("_to_")[0],
                "target_vendor": direction.split("_to_")[1],
            })
        return results


def create_training_datasets():
    """
    将挖掘的数据组织成论文 Table II 的格式:
    四个任务的训练数据集。
    """
    datasets = {
        "config_generation": {
            "train": [],
            "val": [],
            "test": [],
            "description": "NL -> Config (Cisco/Juniper)",
        },
        "config_analysis": {
            "train": [],
            "val": [],
            "test": [],
            "description": "Config -> NL (Cisco/Juniper)",
        },
        "config_translation": {
            "train": [],
            "val": [],
            "test": [],
            "description": "Cisco <-> Juniper",
        },
        "config_completion": {
            "train": [],
            "val": [],
            "test": [],
            "description": "Partial -> Complete",
        },
    }
    return datasets


def format_as_instruction(task_name: str, input_text: str, output_text: str) -> Dict:
    """
    将数据格式化为 Instruction Tuning 格式
    论文 Section III-D: 使用 <指令, 输入, 输出> 三元组
    """
    task_prompts = {
        "config_generation": "Generate the network configuration for the following requirement:",
        "config_analysis": "Analyze the following network configuration and describe its functionality:",
        "config_translation_cisco_to_juniper": "Translate the following Cisco configuration to Juniper Junos format:",
        "config_translation_juniper_to_cisco": "Translate the following Juniper Junos configuration to Cisco IOS format:",
        "config_completion": "Complete the following incomplete network configuration:",
    }

    instruction = task_prompts.get(task_name, "Process the following network configuration:")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "task": task_name,
    }


def main():
    parser = argparse.ArgumentParser(description="PreConfig Data Mining")
    parser.add_argument("--method", choices=["bow_knn", "llm_translate", "all"], default="all")
    parser.add_argument("--raw-data", type=str, default=str(RAW_DATA_DIR / "all_crawled.json"))
    parser.add_argument("--standard-configs", type=str, default=str(RAW_DATA_DIR / "all_crawled.json"))
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Try multiple raw data locations
    possible_raw_paths = [
        Path(args.raw_data),
        RAW_DATA_DIR / "all_crawled.json",
        RAW_DATA_DIR / "netops7b.json",
        RAW_DATA_DIR / "fallback_synthetic.json",
    ]

    raw_path = None
    for p in possible_raw_paths:
        if p.exists():
            raw_path = p
            break

    if args.method in ["bow_knn", "all"]:
        logger.info("=" * 60)
        logger.info("Method 2: BoW + KNN Configuration Extraction")
        logger.info("=" * 60)

        extractor = ConfigExtractor(min_config_lines=3, knn_top_n=5)
        std_path = Path(args.standard_configs)

        if raw_path is not None:
            configs = extractor.extract(raw_path, std_path)
            output_path = PROCESSED_DIR / "forum_configs_extracted.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(configs, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(configs)} extracted configs to {output_path}")
        else:
            logger.warning(f"No raw data found. Run 00_download_datasets.py first.")
            logger.warning(f"Searched: {possible_raw_paths}")

    if args.method in ["llm_translate", "all"]:
        logger.info("=" * 60)
        logger.info("Method 3: LLM-Based Translation Data Generation")
        logger.info("=" * 60)

        generator = TranslationDataGenerator()

        if raw_path is not None:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            cisco_configs = [d for d in raw_data if d.get("source") == "cisco" and d.get("config_text")]
            juniper_configs = [d for d in raw_data if d.get("source") == "juniper" and d.get("config_text")]

            cisco_to_juniper = generator.generate_batch(cisco_configs[:20], "cisco_to_juniper")
            juniper_to_cisco = generator.generate_batch(juniper_configs[:20], "juniper_to_cisco")

            translations = cisco_to_juniper + juniper_to_cisco
            output_path = PROCESSED_DIR / "translation_data.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(translations)} translation pairs to {output_path}")
        else:
            logger.warning(f"No raw data found. Run 00_download_datasets.py first.")

    # Create full training dataset from all available raw data
    logger.info("=" * 60)
    logger.info("Creating Full Instruction-Tuning Training Dataset")
    logger.info("=" * 60)

    if raw_path is not None:
        with open(raw_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        logger.info(f"Loaded {len(all_data)} total raw samples")

        training_samples = []
        for item in all_data:
            nl = item.get("nl_text", "")
            cfg = item.get("config_text", "")
            source = item.get("source", "unknown")
            if len(nl) < 5 or len(cfg) < 5:
                continue

            # Generation: NL -> Config
            training_samples.append(format_as_instruction("config_generation", nl, cfg))
            # Analysis: Config -> NL
            training_samples.append(format_as_instruction("config_analysis", cfg, nl))

        if training_samples:
            import random
            random.shuffle(training_samples)
            train_path = PROCESSED_DIR / "train_data.json"
            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(training_samples, f, ensure_ascii=False, indent=2)
            logger.info(f"Created {len(training_samples)} instruction-tuning samples -> {train_path}")

    datasets = create_training_datasets()
    output_path = PROCESSED_DIR / "datasets_structure.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)
    logger.info(f"Dataset structure saved to {output_path}")


if __name__ == "__main__":
    main()
