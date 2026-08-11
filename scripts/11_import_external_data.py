"""
Step 11: Import External Datasets (NIT / NetConfEval / Juniper jvd)
=================================================================
把外部数据源统一成现有 pipeline 的 7 字段格式:
    {source, doc_type, url, nl_text, config_text, config_type, metadata}

数据源:
  1. NIT (Smarneh/NIT): 1000 条 NL intent -> Juniper EX3300 set 命令
  2. NetConfEval (NetConfEval/NetConfEval): NL -> 形式化规范 / 冲突检测 / FRR 配置
  3. Juniper jvd: 官方验证设计配置库 (901 个 .conf)

用法:
    python scripts/11_import_external_data.py
"""

import json, logging, re
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "converted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. NIT: NL intent -> Juniper config
# ═══════════════════════════════════════════════════════════

def convert_nit() -> List[Dict]:
    path = EXTERNAL_DIR / "NIT_dataset.json"
    if not path.exists():
        logger.warning("NIT dataset not found, skip")
        return []
    data = json.load(open(path, encoding="utf-8"))
    results = []
    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        context = item.get("context", "").strip()
        if not question or not answer:
            continue
        results.append({
            "source": "nit",
            "doc_type": "labeled",
            "url": "https://huggingface.co/datasets/Smarneh/NIT",
            "nl_text": question,
            "config_text": answer,
            "config_type": "junos",
            "metadata": {
                "source": "Smarneh/NIT",
                "context": context,
            },
        })
    logger.info(f"NIT: {len(results)} samples")
    return results


# ═══════════════════════════════════════════════════════════
# 2. NetConfEval: NL -> formal spec / conflict / FRR config
# ═══════════════════════════════════════════════════════════

def convert_netconfeval() -> Dict[str, List[Dict]]:
    nce_dir = EXTERNAL_DIR / "netconfeval"
    if not nce_dir.exists():
        logger.warning("NetConfEval dir not found, skip")
        return {}

    def load_jsonl(name: str) -> List[Dict]:
        path = nce_dir / name
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # 1a. NL -> formal spec translation
    spec = load_jsonl("step_1_spec_translation.jsonl")
    spec_out = []
    for item in spec:
        hl = item.get("human_language", "").strip()
        exp = item.get("expected", "").strip()
        if not hl or not exp:
            continue
        spec_out.append({
            "source": "netconfeval",
            "doc_type": "labeled",
            "url": "https://huggingface.co/datasets/NetConfEval/NetConfEval",
            "nl_text": hl,
            "config_text": exp,
            "config_type": "formal_spec",
            "metadata": {
                "source": "NetConfEval",
                "task": "spec_translation",
                "conflict_exist": item.get("conflict_exist"),
            },
        })
    logger.info(f"NetConfEval spec_translation: {len(spec_out)} samples")

    # 1b. Conflict detection (NL -> conflict decision)
    conflict = load_jsonl("step_1_spec_conflict.jsonl")
    conflict_out = []
    for item in conflict:
        hl = item.get("human_language", "").strip()
        exp = item.get("expected", "").strip()
        desc = item.get("description", "").strip()
        if not hl:
            continue
        conflict_out.append({
            "source": "netconfeval",
            "doc_type": "labeled",
            "url": "https://huggingface.co/datasets/NetConfEval/NetConfEval",
            "nl_text": desc + "\n\n" + hl,
            "config_text": exp,
            "config_type": "conflict",
            "metadata": {
                "source": "NetConfEval",
                "task": "conflict_detection",
                "conflict_exist": item.get("conflict_exist"),
            },
        })
    logger.info(f"NetConfEval conflict: {len(conflict_out)} samples")

    # 1c. Low-level config (FRR)
    low = load_jsonl("step_3_low_level.jsonl")
    low_out = []
    for item in low:
        prompt = item.get("prompt", "").strip()
        result = item.get("result", "").strip()
        if not result:
            continue
        # result 是 {"device": "config", ...} 的 dict
        try:
            result_dict = json.loads(result)
        except json.JSONDecodeError:
            result_dict = {"all": result}
        for dev, cfg in result_dict.items():
            low_out.append({
                "source": "netconfeval",
                "doc_type": "labeled",
                "url": "https://huggingface.co/datasets/NetConfEval/NetConfEval",
                "nl_text": prompt,
                "config_text": cfg.strip(),
                "config_type": "frr",
                "metadata": {
                    "source": "NetConfEval",
                    "task": "low_level",
                    "scenario": item.get("scenario_name"),
                    "device": dev,
                },
            })
    logger.info(f"NetConfEval low_level: {len(low_out)} samples")

    return {"spec_translation": spec_out, "conflict": conflict_out, "low_level": low_out}


# ═══════════════════════════════════════════════════════════
# 3. Juniper jvd: validated design configs
# ═══════════════════════════════════════════════════════════

CONF_IGNORE = {
    "configuration/conf",  # 目录级配置集合
}

def convert_jvd(jvd_root: Path) -> List[Dict]:
    if not jvd_root.exists():
        logger.warning("jvd dir not found, skip")
        return []
    results = []
    for conf in sorted(jvd_root.rglob("*.conf")):
        rel = conf.relative_to(jvd_root)
        text = conf.read_text(encoding="utf-8", errors="ignore").strip()
        if not text or len(text) < 20:
            continue
        # 跳过目录清单类配置（可能不是单个设备的完整配置）
        results.append({
            "source": "juniper",
            "doc_type": "jvd",
            "url": "https://github.com/Juniper/jvd",
            "nl_text": f"Juniper Validated Design configuration: {rel}",
            "config_text": text,
            "config_type": "junos",
            "metadata": {
                "source": "Juniper/jvd",
                "path": str(rel),
            },
        })
    logger.info(f"jvd: {len(results)} samples")
    return results


# ═══════════════════════════════════════════════════════════

def main():
    nit = convert_nit()
    nce = convert_netconfeval()
    jvd = convert_jvd(EXTERNAL_DIR / "jvd")

    all_samples = list(nit) + jvd
    for k, v in nce.items():
        all_samples.extend(v)

    with open(OUTPUT_DIR / "external_unified.json", "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Total unified: {len(all_samples)} samples → data/external/converted/external_unified.json")

    from collections import Counter
    dist = Counter(x["source"] for x in all_samples)
    logger.info(f"Source distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
