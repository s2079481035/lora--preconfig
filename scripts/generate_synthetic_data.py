"""
Synthetic Network Configuration Data Generator
===============================================
Generates Cisco IOS + Juniper Junos configs with NL descriptions
for all training tasks. No API keys or internet needed.

Usage:
    python scripts/generate_synthetic_data.py --num-samples 5000
    python scripts/generate_synthetic_data.py --num-samples 10000 --output data/raw/synthetic_large.json
"""

import re, json, logging, argparse, random
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
random.seed(42)


# ═══════════════════════════════════════════════════════════════
# Cisco Config Templates
# ═══════════════════════════════════════════════════════════════

def cisco_bgp(asn: int, peer_ip: str, remote_as: int, net: str) -> str:
    return (
        f"router bgp {asn}\n"
        f" bgp router-id {peer_ip}\n"
        f" neighbor {peer_ip} remote-as {remote_as}\n"
        f" address-family ipv4 unicast\n"
        f"  network {net} mask 255.255.255.0\n"
        f"  neighbor {peer_ip} activate\n"
        f" exit-address-family"
    )

def cisco_ospf(proc: int, net: str, area: int = 0) -> str:
    return (
        f"router ospf {proc}\n"
        f" router-id 1.1.1.{proc}\n"
        f" network {net} 0.0.0.255 area {area}\n"
        f" default-information originate"
    )

def cisco_acl(acl_num: int, subnet: str, protocol: str = "tcp", port: int = 80) -> str:
    first_octet = int(subnet.split(".")[0])
    second_octet = int(subnet.split(".")[1])
    wildcard = f"0.0.0.255"
    return (
        f"access-list {acl_num} permit {protocol} {subnet} {wildcard} any eq {port}\n"
        f"access-list {acl_num} deny ip any any"
    )

def cisco_static_route(net: str, next_hop: str, metric: int = 1) -> str:
    return f"ip route {net} 255.255.255.0 {next_hop} {metric}"

def cisco_route_map(name: str, asn: int, seq: int = 10) -> str:
    return (
        f"route-map {name} permit {seq}\n"
        f" match ip address prefix-list PL-{name}\n"
        f" set local-preference 150\n"
        f" set community {asn}:100"
    )

def cisco_interface(name: str, ip: str, desc: str = "") -> str:
    desc_line = f" description {desc}\n" if desc else ""
    return (
        f"interface {name}\n"
        f"{desc_line}"
        f" ip address {ip} 255.255.255.0\n"
        f" no shutdown"
    )

def cisco_nat(acl_num: int, pool_name: str, start_ip: str, end_ip: str) -> str:
    return (
        f"ip nat pool {pool_name} {start_ip} {end_ip} netmask 255.255.255.0\n"
        f"ip nat inside source list {acl_num} pool {pool_name} overload\n"
        f"interface GigabitEthernet0/0\n"
        f" ip nat inside\n"
        f"interface GigabitEthernet0/1\n"
        f" ip nat outside"
    )

def cisco_vlan(vlan_id: int, name: str, ip: str) -> str:
    return (
        f"vlan {vlan_id}\n"
        f" name {name}\n"
        f"!\n"
        f"interface Vlan{vlan_id}\n"
        f" ip address {ip} 255.255.255.0\n"
        f" no shutdown"
    )

def cisco_prefix_list(name: str, net: str, seq: int = 5) -> str:
    return (
        f"ip prefix-list {name} seq {seq} permit {net}/24"
    )


# ═══════════════════════════════════════════════════════════════
# Juniper Config Templates
# ═══════════════════════════════════════════════════════════════

def juniper_bgp(asn: int, peer_ip: str, remote_as: int, net: str) -> str:
    return (
        f"routing-options {{\n"
        f"    autonomous-system {asn};\n"
        f"}}\n"
        f"protocols {{\n"
        f"    bgp {{\n"
        f"        group external {{\n"
        f"            type external;\n"
        f"            peer-as {remote_as};\n"
        f"            neighbor {peer_ip};\n"
        f"        }}\n"
        f"    }}\n"
        f"}}\n"
        f"policy-options {{\n"
        f"    policy-statement export-direct {{\n"
        f"        term 1 {{\n"
        f"            from {{\n"
        f"                route-filter {net}/24 exact;\n"
        f"            }}\n"
        f"            then accept;\n"
        f"        }}\n"
        f"    }}\n"
        f"}}"
    )

def juniper_ospf(area: int, interface: str, net: str) -> str:
    return (
        f"protocols {{\n"
        f"    ospf {{\n"
        f"        area {area} {{\n"
        f"            interface {interface};\n"
        f"        }}\n"
        f"    }}\n"
        f"}}\n"
        f"routing-options {{\n"
        f"    static {{\n"
        f"        route {net}/24 discard;\n"
        f"    }}\n"
        f"}}"
    )

def juniper_firewall(filter_name: str, subnet: str, port: int = 80) -> str:
    return (
        f"firewall {{\n"
        f"    family inet {{\n"
        f"        filter {filter_name} {{\n"
        f"            term PERMIT-HTTP {{\n"
        f"                from {{\n"
        f"                    source-address {subnet}/24;\n"
        f"                    protocol tcp;\n"
        f"                    destination-port {port};\n"
        f"                }}\n"
        f"                then accept;\n"
        f"            }}\n"
        f"            term DENY-ALL {{\n"
        f"                then reject;\n"
        f"            }}\n"
        f"        }}\n"
        f"    }}\n"
        f"}}"
    )

def juniper_static_route(net: str, next_hop: str) -> str:
    return (
        f"routing-options {{\n"
        f"    static {{\n"
        f"        route {net}/24 {{\n"
        f"            next-hop {next_hop};\n"
        f"        }}\n"
        f"    }}\n"
        f"}}"
    )

def juniper_interface(name: str, ip: str, desc: str = "") -> str:
    desc_line = f"            description \"{desc}\";\n" if desc else ""
    return (
        f"interfaces {{\n"
        f"    {name} {{\n"
        f"        unit 0 {{\n"
        f"            family inet {{\n"
        f"{desc_line}"
        f"                address {ip}/24;\n"
        f"            }}\n"
        f"        }}\n"
        f"    }}\n"
        f"}}"
    )


# ═══════════════════════════════════════════════════════════════
# NL Description Templates (per config type)
# ═══════════════════════════════════════════════════════════════

CISCO_NL_TEMPLATES = {
    "bgp": [
        "Configure BGP on the Cisco router with AS {asn}, peer with {peer_ip} in AS {remote_as}, and advertise network {net}/24.",
        "Set up BGP routing: local AS {asn}, neighbor {peer_ip} remote-as {remote_as}, network {net}/24.",
        "Enable BGP process {asn}, establish peering with {peer_ip} (AS {remote_as}), and announce {net}/24.",
        "Configure eBGP on the router: AS {asn}, neighbor {peer_ip} remote-as {remote_as}, network advertisement {net}/24.",
        "Implement BGP with local autonomous system {asn}, adding neighbor {peer_ip} with remote AS {remote_as}, and network {net}/24.",
    ],
    "ospf": [
        "Configure OSPF process {proc} on the router, advertise network {net}/24 in area {area}.",
        "Set up OSPF routing with process ID {proc}, network {net}/24 in area {area}.",
        "Enable OSPF process {proc}, add network {net}/24 to area {area} with default route origination.",
        "Configure OSPF routing: process {proc}, network {net}/24 in backbone area {area}.",
        "Implement OSPF process {proc} to advertise {net}/24 within area {area}.",
    ],
    "acl": [
        "Create ACL {acl_num} to permit {protocol} traffic from {subnet}/24 to any destination on port {port}, deny all other traffic.",
        "Configure access-list {acl_num}: allow {protocol} port {port} from {subnet}/24, deny everything else.",
        "Set up an extended ACL (number {acl_num}) permitting {protocol} port {port} traffic from {subnet}/24.",
        "Implement security policy via ACL {acl_num}: permit {protocol} {subnet}/24 to any eq {port}, deny ip any any.",
        "Define ACL {acl_num} to control traffic: permit {protocol} from source {subnet}/24 to destination port {port}.",
    ],
    "static": [
        "Configure a static route to reach {net}/24 via next-hop {next_hop}.",
        "Add a static route for network {net}/24 pointing to gateway {next_hop}.",
        "Set up static routing: network {net}/24 with next-hop address {next_hop}.",
        "Create a static route entry for {net}/24 with the next hop set to {next_hop}.",
        "Define a static route to forward traffic destined for {net}/24 through {next_hop}.",
    ],
    "route_map": [
        "Create route-map {name} to match prefix-list PL-{name} and set local preference to 150 and community {asn}:100.",
        "Configure route-map {name} that matches the prefix list and adjusts BGP attributes (local-pref 150, community {asn}:100).",
        "Set up route-map {name} permit {seq} to match specific prefixes and modify BGP path attributes.",
    ],
    "interface": [
        "Configure interface {name} with IP address {ip}/24 and enable it.",
        "Set up interface {name}: assign IP {ip}/24 and bring it up with no shutdown.",
        "Configure the {name} interface with address {ip}/24 and activate it.",
    ],
    "nat": [
        "Configure NAT using ACL {acl_num} to translate internal traffic through pool {pool_name} ({start_ip}-{end_ip}) with overload.",
        "Set up dynamic NAT with overload: match ACL {acl_num}, translate to pool {pool_name} ({start_ip}-{end_ip}).",
        "Implement NAT overload: inside interfaces translate to pool {pool_name} ({start_ip}-{end_ip}) matching ACL {acl_num}.",
    ],
    "vlan": [
        "Create VLAN {vlan_id} named {vlan_name} with SVI interface IP {ip}/24.",
        "Configure VLAN {vlan_id} ({vlan_name}) and its layer 3 interface with address {ip}/24.",
        "Set up VLAN {vlan_name} (ID {vlan_id}) with switch virtual interface IP {ip}/24.",
    ],
}

JUNIPER_NL_TEMPLATES = {
    "bgp": [
        "Configure BGP on the Juniper device with AS {asn}, peer {peer_ip} remote-as {remote_as}, advertise {net}/24.",
        "Set up BGP routing on JunOS: autonomous-system {asn}, external peer group to {peer_ip} (AS {remote_as}), route filter for {net}/24.",
        "Enable BGP on Juniper: local AS {asn}, external peering with {peer_ip} in AS {remote_as}, advertise {net}/24 via export policy.",
        "Configure Juniper BGP with AS {asn}, establishing an external BGP session to {peer_ip} (AS {remote_as}) and advertising {net}/24.",
        "Implement BGP on JunOS router: set autonomous-system {asn}, configure external peer group with neighbor {peer_ip} (remote AS {remote_as}).",
    ],
    "ospf": [
        "Configure OSPF on Juniper, area {area}, interface {interface}, with a static discard route for {net}/24.",
        "Set up Juniper OSPF routing: area {area}, enable on {interface}, and add static discard for {net}/24.",
        "Enable OSPF on JunOS in area {area} on interface {interface} with a static route to {net}/24.",
    ],
    "firewall": [
        "Configure a Juniper firewall filter {filter_name} to permit HTTP (port {port}) from {subnet}/24 and deny all other traffic.",
        "Set up Juniper firewall family inet filter {filter_name}: allow TCP port {port} from source {subnet}/24, reject everything else.",
        "Create a Juniper firewall filter {filter_name} that permits traffic from {subnet}/24 on port {port} and rejects other traffic.",
    ],
    "static": [
        "Configure a static route on Juniper to reach {net}/24 via next-hop {next_hop}.",
        "Set up JunOS static routing: route {net}/24 with next-hop {next_hop}.",
        "Add a static route in Juniper for network {net}/24 pointing to gateway {next_hop}.",
    ],
    "interface": [
        "Configure interface {name} on Juniper with IP address {ip}/24.",
        "Set up Juniper interface {name} unit 0 with family inet address {ip}/24.",
        "Configure JunOS interface {name}: assign IP {ip}/24 in unit 0.",
    ],
}


# ═══════════════════════════════════════════════════════════════
# Parameter Generators
# ═══════════════════════════════════════════════════════════════

def _ip_parts(asn: int) -> Tuple[int, int, int, int]:
    return (10, (asn // 256) % 256, asn % 256, 1)

def _subnet(asn: int) -> str:
    return f"10.{(asn // 256) % 256}.{asn % 256}.0"

def _peer_ip(asn: int) -> str:
    return f"192.168.{asn % 256}.1"

def _next_hop(asn: int) -> str:
    return f"10.0.0.{asn % 254 + 1}"

def _intf_name(i: int) -> str:
    return f"GigabitEthernet0/{i % 8}"

def _juniper_intf(i: int) -> str:
    return f"ge-0/0/{i % 8}"


# ═══════════════════════════════════════════════════════════════
# Main Generator
# ═══════════════════════════════════════════════════════════════

def generate_cisco_samples(num: int) -> List[Dict]:
    samples = []
    types = ["bgp", "ospf", "acl", "static", "route_map", "interface", "nat", "vlan"]
    for i in range(num):
        t = types[i % len(types)]
        asn = 65000 + (i % 500)
        proc = (i % 99) + 1
        seq = (i % 100) + 1
        acl_n = 100 + (i % 50)

        if t == "bgp":
            peer = _peer_ip(asn)
            rasn = 64512 + (i % 100)
            net = _subnet(asn)
            cfg = cisco_bgp(asn, peer, rasn, net)
            nl_templates = CISCO_NL_TEMPLATES["bgp"]
        elif t == "ospf":
            net = _subnet(asn + 100)
            cfg = cisco_ospf(proc, net)
            nl_templates = CISCO_NL_TEMPLATES["ospf"]
        elif t == "acl":
            subnet = _subnet(asn)
            cfg = cisco_acl(acl_n, subnet, "tcp", 80)
            nl_templates = CISCO_NL_TEMPLATES["acl"]
        elif t == "static":
            net = _subnet(asn + 200)
            nh = _next_hop(asn)
            cfg = cisco_static_route(net, nh)
            nl_templates = CISCO_NL_TEMPLATES["static"]
        elif t == "route_map":
            cfg = cisco_route_map(f"RM-{t}-{seq}", asn, seq)
            nl_templates = CISCO_NL_TEMPLATES["route_map"]
        elif t == "interface":
            ip = f"10.1.{i % 255}.1"
            cfg = cisco_interface(_intf_name(i), ip, f"Link to {_peer_ip(asn)}")
            nl_templates = CISCO_NL_TEMPLATES["interface"]
        elif t == "nat":
            start_ip = f"172.16.0.{(i % 254) + 1}"
            end_ip = f"172.16.0.{(i % 254) + 10}"
            cfg = cisco_nat(acl_n, f"POOL-{seq}", start_ip, end_ip)
            nl_templates = CISCO_NL_TEMPLATES["nat"]
        elif t == "vlan":
            vlan_id = (i % 100) + 10
            ip = f"10.{vlan_id}.1.1"
            cfg = cisco_vlan(vlan_id, f"VLAN-{vlan_id}", ip)
            nl_templates = CISCO_NL_TEMPLATES["vlan"]

        nl = random.choice(nl_templates)
        try:
            nl = nl.format(asn=asn, peer_ip=_peer_ip(asn), remote_as=64512,
                          net=_subnet(asn), proc=proc, area=0, acl_num=acl_n,
                          protocol="tcp", port=80, subnet=_subnet(asn),
                          next_hop=_next_hop(asn), seq=seq, name=f"RM-{t}-{seq}",
                          asn_or_proc=asn, num=acl_n, vlan_id=(i%100)+10,
                          vlan_name=f"VLAN-{(i%100)+10}", pool_name=f"POOL-{seq}",
                          start_ip=start_ip, end_ip=end_ip,
                          interface=_intf_name(i))
        except:
            nl = f"Configure {t} on Cisco device (sample {i})."

        samples.append({
            "source": "cisco",
            "doc_type": "synthetic",
            "url": "synthetic",
            "nl_text": nl,
            "config_text": cfg,
            "config_type": t,
            "metadata": {"variant": f"cisco_{t}_{i}"},
        })

    return samples


def generate_juniper_samples(num: int) -> List[Dict]:
    samples = []
    types = ["bgp", "ospf", "firewall", "static", "interface"]
    for i in range(num):
        t = types[i % len(types)]
        asn = 65000 + (i % 500)

        if t == "bgp":
            peer = _peer_ip(asn)
            rasn = 64512 + (i % 100)
            net = _subnet(asn)
            cfg = juniper_bgp(asn, peer, rasn, net)
            nl_templates = JUNIPER_NL_TEMPLATES["bgp"]
        elif t == "ospf":
            net = _subnet(asn + 100)
            cfg = juniper_ospf(0, _juniper_intf(i), net)
            nl_templates = JUNIPER_NL_TEMPLATES["ospf"]
        elif t == "firewall":
            subnet = _subnet(asn)
            cfg = juniper_firewall(f"FILTER-{i % 100}", subnet, 80)
            nl_templates = JUNIPER_NL_TEMPLATES["firewall"]
        elif t == "static":
            net = _subnet(asn + 200)
            nh = _next_hop(asn)
            cfg = juniper_static_route(net, nh)
            nl_templates = JUNIPER_NL_TEMPLATES["static"]
        elif t == "interface":
            ip = f"10.1.{i % 255}.1"
            cfg = juniper_interface(_juniper_intf(i), ip, f"Link to {_peer_ip(asn)}")
            nl_templates = JUNIPER_NL_TEMPLATES["interface"]

        nl = random.choice(nl_templates)
        try:
            nl = nl.format(asn=asn, peer_ip=_peer_ip(asn), remote_as=64512,
                          net=_subnet(asn), area=0, interface=_juniper_intf(i),
                          filter_name=f"FILTER-{i % 100}", port=80,
                          subnet=_subnet(asn), next_hop=_next_hop(asn),
                          name=f"ge-0/0/{i % 8}")
        except:
            nl = f"Configure {t} on Juniper device (sample {i})."

        samples.append({
            "source": "juniper",
            "doc_type": "synthetic",
            "url": "synthetic",
            "nl_text": nl,
            "config_text": cfg,
            "config_type": t,
            "metadata": {"variant": f"juniper_{t}_{i}"},
        })

    return samples


def generate_all(num_cisco: int, num_juniper: int) -> List[Dict]:
    cisco = generate_cisco_samples(num_cisco)
    juniper = generate_juniper_samples(num_juniper)
    combined = cisco + juniper
    random.shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network config data")
    parser.add_argument("--num-cisco", type=int, default=3000, help="Number of Cisco samples")
    parser.add_argument("--num-juniper", type=int, default=2000, help="Number of Juniper samples")
    parser.add_argument("--output", type=str,
                       default=str(PROJECT_ROOT / "data" / "raw" / "synthetic_large.json"))
    args = parser.parse_args()

    logger.info(f"Generating {args.num_cisco} Cisco + {args.num_juniper} Juniper synthetic samples...")
    data = generate_all(args.num_cisco, args.num_juniper)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    from collections import Counter
    vendors = Counter(d["source"] for d in data)
    types = Counter(d["config_type"] for d in data)
    logger.info(f"Saved {len(data)} samples to {output_path}")
    logger.info(f"  Vendors: {dict(vendors)}")
    logger.info(f"  Types: {dict(types)}")


if __name__ == "__main__":
    main()
