"""
Step 13: NetworkConfigPro Translation Benchmark
================================================
用 NetworkConfigPro 的 generate_from_dict 机制，用同一份设备描述字典
渲染 Cisco IOS 与 Juniper Junos 两份配置，构成平行的翻译对基准。

用途:
  - 替代原来只有 14 条的模板翻译测试集
  - 在 v2 / v3 / 基座模型之间进行公平对比
  - 作为论文中的"跨厂商翻译基准"

用法:
  python scripts/13_translation_benchmark.py --prepare          # 只生成基准
  python scripts/13_translation_benchmark.py --evaluate MODEL   # 生成+评估
"""

import json, logging, os, random, sys, argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/tmp/opencode/datasets/NetworkConfigPro-main")

OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "translation_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


def _make_cfg(hostname, ifaces, vlans=None, ospf=None, bgp=None, static_routes=None,
              acls=None, prefix_lists=None, route_maps=None, dns=None, ntp=None, domain=None):
    from src.core.models import DeviceConfig, Vendor
    return DeviceConfig(
        hostname=hostname,
        vendor=Vendor.JUNIPER_JUNOS,  # 占位, generate 前会覆写
        interfaces=ifaces,
        vlans=vlans or [],
        ospf=ospf, bgp=bgp,
        static_routes=static_routes or [],
        acls=acls or [],
        prefix_lists=prefix_lists or [],
        route_maps=route_maps or [],
        dns_servers=dns or [],
        ntp_servers=ntp or [],
        domain_name=domain or "example.com",
    )


def generate_scenarios():
    """构造 N 个覆盖不同协议/拓扑规模的场景。"""
    from src.core.models import (Interface, InterfaceType, VLAN, StaticRoute,
                                  OSPFConfig, OSPFNetwork, BGPConfig, BGPNeighbor)

    scenarios = []

    # ---- 场景 1: 纯二层 L2 (VLAN + 接口) ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "access-uplink", "10.10.1.1", "255.255.255.0"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "server", "10.10.2.1", "255.255.255.0"),
    ]
    vlans = [VLAN(10, "MANAGEMENT"), VLAN(20, "SERVERS")]
    scenarios.append(_make_cfg("gw-01", ifaces, vlans=vlans))

    # ---- 场景 2: 静态路由 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "wan", "203.0.113.1", "255.255.255.252"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "lan", "192.168.1.1", "255.255.255.0"),
    ]
    static = [StaticRoute("0.0.0.0", "0.0.0.0", "203.0.113.2")]
    scenarios.append(_make_cfg("edge-01", ifaces, static_routes=static,
                               dns=["8.8.8.8", "8.8.4.4"], ntp=["pool.ntp.org"]))

    # ---- 场景 3: OSPF 单区域 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "core0", "10.0.1.1", "255.255.255.0"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "core1", "10.0.2.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.1.1", "255.255.255.255"),
    ]
    ospf = OSPFConfig(process_id=0, router_id="10.255.1.1",
                      networks=[OSPFNetwork("10.0.1.0", "0.0.0.255", 0),
                                OSPFNetwork("10.0.2.0", "0.0.0.255", 0)])
    scenarios.append(_make_cfg("core-01", ifaces, ospf=ospf))

    # ---- 场景 4: OSPF 多区域 + 默认路由注入 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "area0", "10.0.0.1", "255.255.255.0"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "area1", "10.0.10.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.2.1", "255.255.255.255"),
    ]
    ospf = OSPFConfig(process_id=0, router_id="10.255.2.1",
                      networks=[OSPFNetwork("10.0.0.0", "0.0.0.255", 0),
                                OSPFNetwork("10.0.10.0", "0.0.0.255", 1)],
                      default_information_originate=True)
    scenarios.append(_make_cfg("abr-01", ifaces, ospf=ospf))

    # ---- 场景 5: eBGP 单邻居 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "to-peer", "192.168.100.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.3.1", "255.255.255.255"),
    ]
    bgp = BGPConfig(local_as=65000, router_id="10.255.3.1",
                    neighbors=[BGPNeighbor("192.168.100.2", 65001, "upstream-peer")],
                    networks=["10.0.0.0/24", "10.0.1.0/24"])
    scenarios.append(_make_cfg("bgw-01", ifaces, bgp=bgp))

    # ---- 场景 6: eBGP 多邻居 + 网络声明 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "peer-a", "192.168.101.1", "255.255.255.0"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "peer-b", "192.168.102.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.4.1", "255.255.255.255"),
    ]
    bgp = BGPConfig(local_as=65010, router_id="10.255.4.1",
                    neighbors=[BGPNeighbor("192.168.101.2", 65011),
                               BGPNeighbor("192.168.102.2", 65012)],
                    networks=["10.20.0.0/16"])
    scenarios.append(_make_cfg("bgw-02", ifaces, bgp=bgp))

    # ---- 场景 7: OSPF + 静态路由 + DNS/NTP（综合） ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "upstream", "203.0.114.1", "255.255.255.252"),
        Interface("ge-0/0/1", InterfaceType.GIGABIT, "internal", "172.16.0.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.5.1", "255.255.255.255"),
    ]
    ospf = OSPFConfig(process_id=0, router_id="10.255.5.1",
                      networks=[OSPFNetwork("172.16.0.0", "0.0.0.255", 0)])
    static = [StaticRoute("0.0.0.0", "0.0.0.0", "203.0.114.2")]
    scenarios.append(_make_cfg("site-01", ifaces, ospf=ospf, static_routes=static,
                               dns=["1.1.1.1"], ntp=["ntp.example.com"]))

    # ---- 场景 8: BGP + 静态黑洞路由 ----
    ifaces = [
        Interface("ge-0/0/0", InterfaceType.GIGABIT, "to-isp", "198.51.100.1", "255.255.255.0"),
        Interface("lo0", InterfaceType.LOOPBACK, "router-id", "10.255.6.1", "255.255.255.255"),
    ]
    bgp = BGPConfig(local_as=65020, router_id="10.255.6.1",
                    neighbors=[BGPNeighbor("198.51.100.2", 64512, "isp")],
                    networks=["10.30.0.0/24"])
    static = [StaticRoute("10.33.33.33", "255.255.255.255", "Null0")]
    scenarios.append(_make_cfg("bh-01", ifaces, bgp=bgp, static_routes=static))

    return scenarios


def prepare():
    from src.core.generators.config_generator import ConfigGenerator
    from src.core.models import Vendor

    gen = ConfigGenerator()
    scenarios = generate_scenarios()
    pairs = []

    for idx, cfg in enumerate(scenarios):
        cfg_c = _clone_with_vendor(cfg, Vendor.CISCO_IOS)
        cfg_j = _clone_with_vendor(cfg, Vendor.JUNIPER_JUNOS)
        try:
            cisco = gen.generate_from_dict(Vendor.CISCO_IOS, _to_dict(cfg_c))
            juniper = gen.generate_from_dict(Vendor.JUNIPER_JUNOS, _to_dict(cfg_j))
        except Exception as e:
            logger.warning(f"Scen {idx} 渲染失败: {e}")
            continue
        pairs.append({
            "sample": idx,
            "scenario": f"scenario-{idx}",
            "hostname": cfg.hostname,
            "cisco": cisco.strip(),
            "juniper": juniper.strip(),
        })

    out = OUTPUT_DIR / "pairs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Generated {len(pairs)} translation pairs → {out}")
    return pairs


def _clone_with_vendor(cfg, vendor):
    import copy
    from src.core.models import DeviceConfig
    c = copy.deepcopy(cfg)
    c.vendor = vendor
    return c


def _to_dict(cfg):
    return {
        "hostname": cfg.hostname,
        "vendor": cfg.vendor,
        "interfaces": cfg.interfaces,
        "vlans": cfg.vlans,
        "acls": cfg.acls,
        "static_routes": cfg.static_routes,
        "ospf": cfg.ospf,
        "eigrp": cfg.eigrp,
        "bgp": cfg.bgp,
        "stp": cfg.stp,
        "prefix_lists": cfg.prefix_lists,
        "route_maps": cfg.route_maps,
        "enable_secret": cfg.enable_secret,
        "domain_name": cfg.domain_name,
        "dns_servers": cfg.dns_servers,
        "ntp_servers": cfg.ntp_servers,
        "banner_motd": cfg.banner_motd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="只生成基准, 不评估")
    parser.add_argument("--evaluate", type=str, default=None, help="模型路径")
    args = parser.parse_args()

    pairs = prepare()

    if args.evaluate:
        from scripts.bb_eval_translation import evaluate_translation
        evaluate_translation(args.evaluate, pairs)


if __name__ == "__main__":
    main()