"""
Step 15: Build v4 training set (v3 + NetworkConfigPro translation pairs)
=======================================================================
- 读取 v3 训练集 (5994 条)
- 加入 NetworkConfigPro 生成的 150 对翻译 (c2j + j2c 共 300 条)
- 输出 train_data_multitask_v4.json
"""

import json, logging, sys
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
V3 = PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v3.json"
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "train_pairs.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "train_data_multitask_v4.json"

C2J_INSTR = "Translate the following Cisco IOS configuration to Juniper Junos format:"
J2C_INSTR = "Translate the following Juniper Junos configuration to Cisco IOS format:"


def main():
    v3 = json.load(open(V3, encoding="utf-8"))
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    logger.info(f"v3: {len(v3)}, pairs: {len(pairs)}")

    # 用 v3 中已有的 translation 数据 check 是否重复
    existing_outputs = set(s["output"] for s in v3)

    added = 0
    for p in pairs:
        for direction, (instr, src, ref) in [
            ("c2j", (C2J_INSTR, p["cisco"], p["juniper"])),
            ("j2c", (J2C_INSTR, p["juniper"], p["cisco"])),
        ]:
            if ref in existing_outputs:
                continue
            v3.append({
                "instruction": instr,
                "input": src,
                "output": ref,
                "task": f"config_translation_{direction}",
                "metadata": {"source": "netconfigpro"},
            })
            existing_outputs.add(ref)
            added += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(v3, f, ensure_ascii=False, indent=2)

    dist = Counter(s["task"] for s in v3)
    logger.info(f"Added {added} translation samples")
    logger.info(f"v4 total: {len(v3)}, dist: {dict(dist)}")
    logger.info(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()