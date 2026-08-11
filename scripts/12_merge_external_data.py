"""
Step 12: Merge external data (NIT / jvd) into multitask training set
====================================================================
- NIT: NL intent -> Juniper set 命令, 转成花括号格式后作为 generation 数据
- jvd: 官方验证配置, 去注释后作为 completion / generation 素材
- 输出: data/processed/train_data_multitask_v3.json (含外部数据)
"""

import json, logging, re, sys, random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from set_to_curly import convert_set_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_CONVERTED = PROJECT_ROOT / "data" / "external" / "converted" / "external_unified.json"
TRAIN_V2 = PROJECT_ROOT / "data" / "processed" / "train_data_multitask.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v3.json"

random.seed(42)

VAR_RE = re.compile(r'(\$[A-Za-z_]+|<\S+>)')


def strip_jvd_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        ls = line.strip()
        if ls.startswith("/*") or ls.startswith("*") or ls == "*/" or ls.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


def build_nit_samples() -> List[Dict]:
    data = json.load(open(EXTERNAL_CONVERTED, encoding="utf-8"))
    nit = [x for x in data if x["source"] == "nit"]
    samples = []
    for x in nit:
        answer = x["config_text"]
        cmds = [l.strip() for l in answer.split("\n")
                if l.strip() and l.strip() != "commit"
                and not l.strip().startswith("Use the following")]
        # 只要纯 set 生成类 (排除 delete / show / 混合)
        if not cmds or any(c.startswith(("delete ", "show ", "request ", "clear ")) for c in cmds):
            continue
        if not all(c.startswith("set ") for c in cmds):
            continue
        curly = convert_set_config("\n".join(cmds))
        if not curly or "<" in curly:
            continue  # 跳过带占位符的
        samples.append({
            "instruction": "Generate the Juniper Junos configuration for the following requirement:",
            "input": x["nl_text"],
            "output": curly,
            "task": "config_generation",
            "metadata": {"source": "nit"},
        })
    logger.info(f"NIT generation samples: {len(samples)}")
    return samples


def build_jvd_samples() -> List[Dict]:
    data = json.load(open(EXTERNAL_CONVERTED, encoding="utf-8"))
    jvd = [x for x in data if x["source"] == "juniper"]
    generation, completion = [], []
    for x in jvd:
        clean = strip_jvd_comments(x["config_text"])
        if not clean or VAR_RE.search(clean):
            continue
        # 过滤太短/太长
        if len(clean) < 300 or len(clean) > 5000:
            continue
        path = x["metadata"]["path"]
        generation.append({
            "instruction": "Generate the Juniper Junos configuration for the following requirement:",
            "input": f"Configure the Juniper device for: {path.split('/')[-1].replace('.conf', '').replace('-', ' ')}",
            "output": clean,
            "task": "config_generation",
            "metadata": {"source": "jvd"},
        })
        # completion: 截断
        lines = [l.rstrip() for l in clean.split("\n") if l.strip()]
        if len(lines) >= 8:
            for _ in range(1):
                split = random.randint(len(lines) // 3, len(lines) * 7 // 10)
                completion.append({
                    "instruction": "Complete the following incomplete network configuration with the missing lines:",
                    "input": "\n".join(lines[:split]),
                    "output": clean,
                    "task": "config_completion",
                    "metadata": {"source": "jvd"},
                })
    logger.info(f"jvd generation: {len(generation)}, completion: {len(completion)}")
    return generation + completion


def main():
    external = build_nit_samples() + build_jvd_samples()
    logger.info(f"External samples total: {len(external)}")

    # 加载 v2 train, 合并
    v2_train = json.load(open(TRAIN_V2, encoding="utf-8"))
    logger.info(f"v2 train: {len(v2_train)}")

    # 按 output 去重
    existing_outputs = set(s["output"] for s in v2_train)
    added = [s for s in external if s["output"] not in existing_outputs]
    skipped = len(external) - len(added)
    logger.info(f"External added after dedup: {len(added)}, skipped dup: {skipped}")

    merged = v2_train + added
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(s["task"] for s in merged)
    logger.info(f"Merged total: {len(merged)}, task dist: {dict(dist)}")
    logger.info(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
