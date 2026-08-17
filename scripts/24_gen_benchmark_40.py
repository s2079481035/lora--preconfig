"""
Step 24: Expand Unseen Benchmark to 40 Pairs
============================================
用 NetworkConfigPro 生成器生成 40 对全新平行配置 (Cisco/Junos),
扩大未见配置基准 (原 8 对 → 40 对), 增强统计显著性。

与原 8 对(pairs.json) 无重复(不同 seed, 新参数采样)。

用法:
  python scripts/24_gen_benchmark_40.py [--num 40] [--seed 2024]
"""

import json, logging, random, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/tmp/opencode/datasets/NetworkConfigPro-main")

OUTPUT = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "benchmark_pairs_40.json"

import importlib.util
spec = importlib.util.spec_from_file_location(
    "gen14", PROJECT_ROOT / "scripts" / "14_gen_translation_pairs.py")
gen14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen14)
random_scenario = gen14.random_scenario


def main():
    import argparse
    from src.core.generators.config_generator import ConfigGenerator
    from src.core.models import Vendor

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gen = ConfigGenerator()
    pairs = []
    fails = 0

    idx = 0
    while len(pairs) < args.num and idx < args.num * 4:
        cfg = random_scenario(rng, idx)
        idx += 1
        try:
            cisco = gen.generate_from_dict(Vendor.CISCO_IOS, {
                "hostname": cfg.hostname, "vendor": cfg.vendor, "interfaces": cfg.interfaces,
                "vlans": cfg.vlans, "acls": cfg.acls, "static_routes": cfg.static_routes,
                "ospf": cfg.ospf, "eigrp": cfg.eigrp, "bgp": cfg.bgp, "stp": cfg.stp,
                "prefix_lists": cfg.prefix_lists, "route_maps": cfg.route_maps,
                "enable_secret": cfg.enable_secret, "domain_name": cfg.domain_name,
                "dns_servers": cfg.dns_servers, "ntp_servers": cfg.ntp_servers,
                "banner_motd": cfg.banner_motd})
            juniper = gen.generate_from_dict(Vendor.JUNIPER_JUNOS, {
                "hostname": cfg.hostname, "vendor": cfg.vendor, "interfaces": cfg.interfaces,
                "vlans": cfg.vlans, "acls": cfg.acls, "static_routes": cfg.static_routes,
                "ospf": cfg.ospf, "eigrp": cfg.eigrp, "bgp": cfg.bgp, "stp": cfg.stp,
                "prefix_lists": cfg.prefix_lists, "route_maps": cfg.route_maps,
                "enable_secret": cfg.enable_secret, "domain_name": cfg.domain_name,
                "dns_servers": cfg.dns_servers, "ntp_servers": cfg.ntp_servers,
                "banner_motd": cfg.banner_motd})
        except Exception as e:
            fails += 1
            continue

        if len(cisco.strip()) < 50 or len(juniper.strip()) < 50:
            fails += 1
            continue

        pairs.append({
            "sample": len(pairs),
            "scenario": f"scenario-{len(pairs)}",
            "hostname": cfg.hostname,
            "cisco": cisco.strip(),
            "juniper": juniper.strip(),
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Generated {len(pairs)} pairs (fails={fails}) → {OUTPUT}")

    # 检查与原 8 对重叠 (hostname 不同即可视为新配置)
    old = json.load(open(PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "pairs.json"))
    old_hosts = {p["hostname"] for p in old}
    new_hosts = {p["hostname"] for p in pairs}
    logger.info(f"Old hosts: {len(old_hosts)}, new hosts: {len(new_hosts)}, overlap: {len(old_hosts & new_hosts)}")


if __name__ == "__main__":
    main()