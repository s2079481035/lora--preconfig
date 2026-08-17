"""
Step 31: Build Unseen Multi-Task Data from Benchmark Pairs
==========================================================
从 40 对未见配置构建 config_generation / config_completion / config_analysis
任务样本, 用于验证 RAG 在未见配置上的 Gen/Comp/Ana 增益。

用法:
  venv/bin/python scripts/31_build_unseen_multitask.py [--pairs .../benchmark_pairs_40.json]
"""

import json, logging, re, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PAIRS = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "unseen_multitask.json"

rng = random.Random(777)


def extract_features(cfg: str) -> dict:
    """从配置文本提取协议特征 (正则启发式)。"""
    low = cfg.lower()
    f = {}
    m = re.search(r'router bgp (\d+)', cfg)
    f["bgp_as"] = m.group(1) if m else None
    m = re.search(r'local-as (\d+)', cfg)
    if m:
        f["bgp_as"] = m.group(1)
    m = re.search(r'router ospf (\d+)', cfg)
    f["ospf_proc"] = m.group(1) if m else None
    f["ospf"] = "ospf" in low
    f["bgp"] = "bgp" in low
    f["static"] = bool(re.search(r'ip route|routing-options.*static|static', low, re.S))
    f["vlans"] = bool(re.search(r'vlan|access vlan', low))
    f["acls"] = len(re.findall(r'access-list|access-list-name|firewall', low, re.I))
    m = re.findall(r'neighbor ([\d.]+) remote-as (\d+)', cfg)
    if not m:
        m = re.findall(r'neighbor ([\d.]+) as (\d+)', cfg)
    f["neighbors"] = m
    m = re.findall(r'(?:network|networks|interface) (10\.\d+\.\d+\.\d+)', cfg)
    f["networks"] = [x for x in m if not x.startswith("10.255")][:2]
    return f


def build_generation_desc(f: dict, vendor: str) -> str:
    """从特征构造 Gen 任务的描述输入 (模板句风格, 与主测试集一致)。"""
    parts = []
    if f["bgp"]:
        parts.append(f"Configure BGP with AS {f['bgp_as']}")
        if f["neighbors"]:
            parts.append(f"peer with {f['neighbors'][0][0]} whose AS is {f['neighbors'][0][1]}")
    if f["ospf"]:
        nets = ", ".join(f"{n}/24" for n in f["networks"][:2]) or "10.0.0.0/24"
        parts.append(f"Configure OSPF process {f['ospf_proc'] or 64} on networks {nets}")
    if f["static"]:
        parts.append("Add a default static route via the next hop")
    if f["acls"]:
        parts.append(f"Apply {f['acls']} access list(s) to control traffic")
    if f["vlans"]:
        parts.append("Create VLANs and assign access ports")
    desc = "; ".join(parts)
    if not desc:
        desc = f"Configure a basic {vendor} router"
    return desc + f" ({vendor} config requested)"


def build_analysis_ref(f: dict, vendor: str) -> str:
    """构造 Ana 参考 (模板句, 与主测试集同款退化参考口径)。"""
    parts = []
    if f["bgp"]:
        parts.append(f"BGP is configured with AS {f['bgp_as']}")
    if f["ospf"]:
        parts.append(f"OSPF process {f['ospf_proc'] or 64} is running")
    if f["static"]:
        parts.append("a default route is set")
    if f["acls"]:
        parts.append(f"{f['acls']} ACL(s) filter traffic")
    return ", ".join(parts) + f" on {vendor}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    samples = []

    for p in pairs:
        for vendor, key in [("Cisco", "cisco"), ("Juniper", "juniper")]:
            cfg = p[key]
            f = extract_features(cfg)

            # Generation: 描述 → 配置
            desc = build_generation_desc(f, vendor)
            samples.append({
                "instruction": "Generate the network configuration described below:",
                "input": desc, "output": cfg, "task": "config_generation",
            })

            # Analysis: 配置 → 描述 (模板句参考)
            ref = build_analysis_ref(f, vendor)
            samples.append({
                "instruction": "Analyze the following configuration and describe what it does:",
                "input": cfg, "output": ref, "task": "config_analysis",
            })

            # Completion: 截断 → 完整
            lines = [l.rstrip() for l in cfg.split("\n") if l.strip()]
            if len(lines) >= 8:
                split = rng.randint(len(lines) // 3, len(lines) * 7 // 10)
                samples.append({
                    "instruction": "Complete the following incomplete network configuration with the missing lines:",
                    "input": "\n".join(lines[:split]), "output": cfg, "task": "config_completion",
                })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    from collections import Counter
    logger.info(f"Saved {len(samples)} samples → {args.output}")
    logger.info(f"Task distribution: {dict(Counter(s['task'] for s in samples))}")


if __name__ == "__main__":
    main()