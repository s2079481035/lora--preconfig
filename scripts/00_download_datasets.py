"""
Step 0: Download and adapt HuggingFace datasets for PreConfig
==============================================================
Downloads from:
  - NetOps-7B (cwccie/netops-7b): Cisco + Juniper config data
  - NetConfEval (NetConfEval/NetConfEval): OSPF/BGP config benchmark
  - NIT (Smarneh/NIT): Juniper intent-to-config pairs

Usage:
    python scripts/00_download_datasets.py [--hf-token YOUR_TOKEN]
"""

import sys, os, json, logging, argparse, itertools, random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_text(item) -> str:
    """Extract text string from various field types."""
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return "\n".join(_flatten_text(x) for x in item if x)
    if isinstance(item, dict):
        return "\n".join(str(v) for v in item.values() if v)
    return str(item) if item else ""


def download_netops7b() -> List[Dict]:
    """Download NetOps-7B dataset - multic-vendor config data."""
    logger.info("=" * 60)
    logger.info("Downloading NetOps-7B (HuggingFace: cwccie/netops-7b)")
    logger.info("=" * 60)
    try:
        from datasets import load_dataset
        ds = load_dataset("cwccie/netops-7b", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load NetOps-7B: {e}")
        logger.info("Using fallback: generating synthetic config data")
        return _generate_fallback_cisco_data()

    results = []
    for i, sample in enumerate(ds):
        text = _flatten_text(sample.get("text", ""))
        if len(text) < 20:
            continue
        # Classify by vendor keywords
        vendor = "unknown"
        if any(kw in text.lower() for kw in ["junos", "set ", "protocols {", "routing-options"]):
            vendor = "juniper"
        elif any(kw in text.lower() for kw in ["interface", "router bgp", "router ospf", "ip route", "access-list"]):
            vendor = "cisco"

        config_type = "other"
        for tp, kws in [("bgp", ["bgp"]), ("ospf", ["ospf"]), ("static", ["ip route", "routing-options"]),
                        ("acl", ["access-list", "firewall", "security"]), ("route_policy", ["route-map", "policy-options"])]:
            if any(kw in text.lower() for kw in kws):
                config_type = tp
                break

        results.append({
            "source": vendor,
            "doc_type": "hf_dataset",
            "url": "cwccie/netops-7b",
            "nl_text": text[:200],
            "config_text": text,
            "config_type": config_type,
            "metadata": {"hf_dataset": "netops-7b", "index": i},
        })
    logger.info(f"Extracted {len(results)} configs from NetOps-7B")
    return results


def download_netconfeval() -> List[Dict]:
    """Download NetConfEval benchmark dataset."""
    logger.info("=" * 60)
    logger.info("Downloading NetConfEval (HuggingFace: NetConfEval/NetConfEval)")
    logger.info("=" * 60)
    try:
        from datasets import load_dataset
        ds = load_dataset("NetConfEval/NetConfEval", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load NetConfEval: {e}")
        return []

    results = []
    for i, sample in enumerate(ds):
        prompt = _flatten_text(sample.get("prompt", ""))
        result = sample.get("result", {})
        if isinstance(result, dict):
            for device, config in result.items():
                config_text = _flatten_text(config)
                if len(config_text) > 20:
                    results.append({
                        "source": "frrouting",
                        "doc_type": "benchmark",
                        "url": "NetConfEval/NetConfEval",
                        "nl_text": prompt,
                        "config_text": config_text,
                        "config_type": "bgp_ospf",
                        "metadata": {"hf_dataset": "NetConfEval", "device": device, "index": i},
                    })
    logger.info(f"Extracted {len(results)} configs from NetConfEval")
    return results


def download_nit() -> List[Dict]:
    """Download NIT (Network Intent Translations) dataset."""
    logger.info("=" * 60)
    logger.info("Downloading NIT (HuggingFace: Smarneh/NIT)")
    logger.info("=" * 60)
    try:
        from datasets import load_dataset
        ds = load_dataset("Smarneh/NIT", split="train", trust_remote_code=True)
        logger.info(f"Loaded {len(ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load NIT: {e}")
        return []

    results = []
    for i, sample in enumerate(ds):
        question = _flatten_text(sample.get("question", ""))
        answer = _flatten_text(sample.get("answer", ""))
        if len(question) > 10 and len(answer) > 10:
            results.append({
                "source": "juniper",
                "doc_type": "intent_nl",
                "url": "Smarneh/NIT",
                "nl_text": question,
                "config_text": answer,
                "config_type": "juniper_config",
                "metadata": {"hf_dataset": "NIT", "index": i},
            })
    logger.info(f"Extracted {len(results)} pairs from NIT")
    return results


def _generate_fallback_cisco_data(n_samples: int = 200) -> List[Dict]:
    """Fallback: synthetic Cisco configs when HuggingFace is unavailable."""
    logger.warning(f"Generating {n_samples} fallback synthetic configs")
    templates = []
    # Cisco BGP
    for asn in range(65000, 65500):
        templates.append({
            "nl": f"Configure BGP with AS {asn}, peer with 192.168.{asn % 256}.1",
            "cfg": f"router bgp {asn}\n neighbor 192.168.{asn % 256}.1 remote-as 64512\n address-family ipv4 unicast\n  network 10.0.0.0 mask 255.255.255.0\n  neighbor 192.168.{asn % 256}.1 activate\n exit-address-family",
            "type": "bgp",
        })
    # Cisco OSPF
    for proc in range(1, 100):
        templates.append({
            "nl": f"Configure OSPF process {proc} on networks 10.{proc}.0.0/24",
            "cfg": f"router ospf {proc}\n network 10.{proc}.0.0 0.0.0.255 area 0\n default-information originate",
            "type": "ospf",
        })
    # Cisco ACL
    for acl_n in range(100, 150):
        templates.append({
            "nl": f"Create ACL {acl_n} to permit HTTP from 10.{acl_n % 100}.0.0/24",
            "cfg": f"access-list {acl_n} permit tcp 10.{acl_n % 100}.0.0 0.0.0.255 any eq 80\naccess-list {acl_n} deny ip any any",
            "type": "acl",
        })
    # Juniper BGP
    for asn in range(65000, 65500):
        templates.append({
            "nl": f"Configure BGP on Juniper with AS {asn}",
            "cfg": f"routing-options {{\n    autonomous-system {asn};\n}}\nprotocols {{\n    bgp {{\n        group external {{\n            type external;\n            peer-as 64512;\n            neighbor 192.168.{asn % 256}.1;\n        }}\n    }}\n}}",
            "type": "bgp",
        })
    random.shuffle(templates)
    results = []
    for t in templates[:n_samples]:
        results.append({
            "source": "cisco" if "router" in t["cfg"] else "juniper",
            "doc_type": "synthetic",
            "url": "fallback",
            "nl_text": t["nl"],
            "config_text": t["cfg"],
            "config_type": t["type"],
            "metadata": {"source": "fallback"},
        })
    return results


def save_results(results: List[Dict], name: str):
    path = RAW_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} items to {path}")


def merge_all(raw_data_paths: List[Path]) -> List[Dict]:
    """Merge all raw data into a single file."""
    all_data = []
    for path in raw_data_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
    merged_path = RAW_DIR / "all_crawled.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged {len(all_data)} total items into {merged_path}")
    return all_data


def main():
    parser = argparse.ArgumentParser(description="PreConfig Dataset Downloader")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--fallback", action="store_true", help="Use fallback synthetic data only")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    all_datasets = []

    if args.fallback:
        results = _generate_fallback_cisco_data(500)
        save_results(results, "fallback_synthetic")
        all_datasets.append(results)
    else:
        for name, fn in [("netops7b", download_netops7b),
                         ("netconfeval", download_netconfeval),
                         ("nit", download_nit)]:
            try:
                results = fn()
                if results:
                    save_results(results, name)
                    all_datasets.append(results)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")

    # If HuggingFace failed, use fallback
    if not all_datasets:
        logger.warning("No datasets downloaded, using fallback synthetic data")
        results = _generate_fallback_cisco_data(500)
        save_results(results, "fallback_synthetic")
        all_datasets.append(results)

    # Merge everything into all_crawled.json
    raw_files = list(RAW_DIR.glob("*.json"))
    merge_all(raw_files)

    # Count by vendor
    merged = []
    for rf in raw_files:
        with open(rf, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    vendors = {}
    for item in merged:
        v = item.get("source", "unknown")
        vendors[v] = vendors.get(v, 0) + 1
    logger.info(f"Dataset composition by vendor: {vendors}")
    logger.info(f"Total samples: {len(merged)}")


if __name__ == "__main__":
    main()
