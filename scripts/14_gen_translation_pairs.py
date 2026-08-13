"""
Step 14: Generate Translation Training Pairs via NetworkConfigPro
=================================================================
用 NetworkConfigPro 的模板渲染机制，随机组合网络场景，为同一个
设备描述渲染 Cisco IOS 和 Juniper Junos 两个版本，生成平行翻译对。

关键: 同一 DeviceConfig 描述 → 两种厂商渲染 → 天然平行翻译对。
用于扩充翻译训练集 (v4)，解决 v2/v3 翻译任务"模板记忆"问题。

用法:
  python scripts/14_gen_translation_pairs.py [--num 150] [--seed 42]
"""

import json, logging, random, sys, copy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/tmp/opencode/datasets/NetworkConfigPro-main")

OUTPUT = PROJECT_ROOT / "data" / "external" / "translation_benchmark" / "train_pairs.json"


def random_scenario(rng: random.Random, idx: int):
    """随机生成一个设备配置场景 (覆盖多种协议/拓扑组合)。"""
    from src.core.models import (Interface, InterfaceType, VLAN, StaticRoute,
                                  OSPFConfig, OSPFNetwork, BGPConfig, BGPNeighbor,
                                  STPConfig, STPMode, ACL, ACLEntry, ACLAction, ACLProtocol,
                                  PrefixList, PrefixListEntry, RouteMap, RouteMapEntry,
                                  DeviceConfig, Vendor, SwitchportMode)

    n_ifaces = rng.randint(1, 4)
    ifaces = []
    for i in range(n_ifaces):
        ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        itype = rng.choice([InterfaceType.GIGABIT, InterfaceType.GIGABIT,
                            InterfaceType.TEN_GIGABIT, InterfaceType.ETHERNET])
        iface = Interface(
            name=f"ge-0/0/{i}" if itype == InterfaceType.GIGABIT else f"et-0/0/{i}",
            interface_type=itype,
            description=rng.choice(["uplink", "access", "to-core", "wan", "lan", "server", "peer"]),
            ip_address=ip,
            subnet_mask="255.255.255.0",
        )
        # 一半概率做成二层口
        if rng.random() < 0.4:
            iface.switchport_mode = SwitchportMode.ACCESS
            iface.access_vlan = rng.randint(1, 30) * 10
        ifaces.append(iface)

    # Loopback
    if rng.random() < 0.6:
        ifaces.append(Interface(
            name="lo0", interface_type=InterfaceType.LOOPBACK, description="router-id",
            ip_address=f"10.255.{rng.randint(1, 250)}.1", subnet_mask="255.255.255.255"))

    vlans = []
    if rng.random() < 0.5:
        for v in range(rng.randint(1, 3)):
            vlans.append(VLAN((rng.randint(1, 30)) * 10, f"VLAN_{rng.randint(1, 99)}"))

    ospf = None
    if rng.random() < 0.4:
        n_net = rng.randint(1, 3)
        nets = []
        base = rng.randint(0, 20)
        for k in range(n_net):
            area = rng.choice([0, 0, 1, 2])
            nets.append(OSPFNetwork(f"10.{base + k}.0.0", "0.0.0.255", area))
        ospf = OSPFConfig(process_id=0, router_id=f"10.255.{rng.randint(1, 250)}.1",
                          networks=nets,
                          default_information_originate=rng.random() < 0.2)

    bgp = None
    if rng.random() < 0.4:
        n_nei = rng.randint(1, 3)
        local_as = rng.choice([65000, 65010, 65100, 65200, 65300, 64512, 64496, 64516, 64600])
        neighbors = []
        for k in range(n_nei):
            neighbors.append(BGPNeighbor(
                ip_address=f"192.168.{rng.randint(1, 250)}.{rng.randint(1, 254)}",
                remote_as=rng.choice([64512, 64513, 65001, 65002, 64522]),
                description=rng.choice(["", "upstream", "peer-a", "transit", "customer"])))
        networks = [f"10.{rng.randint(0, 200)}.0.0/24"]
        bgp = BGPConfig(local_as=local_as, router_id=f"10.255.{rng.randint(1, 250)}.1",
                        neighbors=neighbors, networks=networks)

    static = []
    if rng.random() < 0.5:
        static.append(StaticRoute("0.0.0.0", "0.0.0.0", f"203.0.{rng.randint(110, 120)}.2"))

    acls = []
    if rng.random() < 0.4:
        acl = ACL(name=f"ACL-{rng.randint(100, 199)}", is_extended=True)
        acl.entries.append(ACLEntry(10, ACLAction.PERMIT, ACLProtocol.IP,
                                    source=f"10.{rng.randint(0, 200)}.0.0", source_wildcard="0.0.255.255"))
        acl.entries.append(ACLEntry(20, ACLAction.DENY, ACLProtocol.IP,
                                    source="any", destination="any"))
        acls.append(acl)

    prefix_lists, route_maps = [], []
    if rng.random() < 0.25:
        pl = PrefixList(name=f"PL-{rng.randint(1, 99)}")
        pl.entries.append(PrefixListEntry(10, "permit", f"10.{rng.randint(0, 200)}.0.0/16"))
        prefix_lists.append(pl)
    if rng.random() < 0.25:
        rm = RouteMap(name=f"RM-{rng.randint(1, 99)}")
        rm.entries.append(RouteMapEntry(10, "permit", set_local_pref=rng.randint(100, 200)))
        route_maps.append(rm)

    return DeviceConfig(
        hostname=f"rt-{idx:03d}",
        vendor=Vendor.JUNIPER_JUNOS,  # 占位
        interfaces=ifaces,
        vlans=vlans,
        acls=acls,
        static_routes=static,
        ospf=ospf,
        bgp=bgp,
        stp=STPConfig(mode=STPMode.RAPID_PVST) if rng.random() < 0.3 else None,
        prefix_lists=prefix_lists,
        route_maps=route_maps,
        enable_secret="admin",
        domain_name=f"net{rng.randint(1, 9)}.example.com",
        dns_servers=["8.8.8.8", "8.8.4.4"] if rng.random() < 0.5 else [],
        ntp_servers=["pool.ntp.org"] if rng.random() < 0.5 else [],
        banner_motd="Authorized access only." if rng.random() < 0.3 else "",
    )


def main():
    import argparse
    from src.core.generators.config_generator import ConfigGenerator
    from src.core.models import Vendor

    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gen = ConfigGenerator()
    pairs = []
    fails = 0

    idx = 0
    while len(pairs) < args.num and idx < args.num * 3:
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
            "hostname": cfg.hostname,
            "cisco": cisco.strip(),
            "juniper": juniper.strip(),
            "protocols": {
                "ospf": cfg.ospf is not None,
                "bgp": cfg.bgp is not None,
                "static": len(cfg.static_routes) > 0,
                "acls": len(cfg.acls) > 0,
                "vlans": len(cfg.vlans) > 0,
                "route_maps": len(cfg.route_maps) > 0,
                "prefix_lists": len(cfg.prefix_lists) > 0,
            },
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"Generated {len(pairs)} pairs (fails={fails}) → {OUTPUT}")

    from collections import Counter
    proto = Counter()
    for p in pairs:
        for k, v in p["protocols"].items():
            if v:
                proto[k] += 1
    logger.info(f"Protocol coverage: {dict(proto)}")


if __name__ == "__main__":
    main()