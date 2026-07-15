"""
Step 1b: Generate Sample Training Data
=======================================
当爬虫无法获取真实数据时，使用预定义的配置样本跑通整个流程。
这些样本覆盖了论文中的主要配置类型：BGP、OSPF、Static Route、ACL。
"""

import json
import random
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# Cisco 配置样本
# ═══════════════════════════════════════════════════════════

CISCO_BGP = [
    """router bgp 65000
 bgp router-id 1.1.1.1
 neighbor 192.168.1.1 remote-as 64512
 neighbor 192.168.1.1 description Peer-to-ISP-A
 address-family ipv4 unicast
  network 10.0.0.0 mask 255.255.255.0
  network 172.16.0.0 mask 255.255.0.0
  neighbor 192.168.1.1 activate
  neighbor 192.168.1.1 route-map FILTER-IN in
  neighbor 192.168.1.1 route-map FILTER-OUT out
 exit-address-family""",
    """router bgp 65001
 neighbor 10.0.0.2 remote-as 65001
 neighbor 10.0.0.2 update-source Loopback0
 neighbor 10.0.0.3 remote-as 65001
 neighbor 10.0.0.3 update-source Loopback0
 address-family ipv4 unicast
  neighbor 10.0.0.2 next-hop-self
  neighbor 10.0.0.3 next-hop-self
 exit-address-family""",
    """router bgp 65000
 bgp log-neighbor-changes
 neighbor 203.0.113.1 remote-as 64500
 address-family ipv4 unicast
  network 192.168.0.0
  neighbor 203.0.113.1 activate
  neighbor 203.0.113.1 soft-reconfiguration inbound
 exit-address-family""",
    """router bgp 65000
 bgp bestpath compare-routerid
 neighbor 10.1.1.1 remote-as 65001
 neighbor 10.2.2.2 remote-as 65002
 address-family ipv4 unicast
  neighbor 10.1.1.1 weight 100
  neighbor 10.2.2.2 weight 200
  neighbor 10.1.1.1 route-map SET-MED out
 exit-address-family""",
    """router bgp 65000
 neighbor 192.168.10.1 remote-as 65100
 neighbor 192.168.20.1 remote-as 65200
 address-family ipv4 multicast
  neighbor 192.168.10.1 activate
  neighbor 192.168.20.1 activate
 exit-address-family""",
]

CISCO_OSPF = [
    """router ospf 1
 router-id 1.1.1.1
 network 10.0.0.0 0.0.0.255 area 0
 network 10.0.1.0 0.0.0.255 area 0
 network 192.168.1.0 0.0.0.255 area 1
 default-information originate always""",
    """router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
 passive-interface GigabitEthernet0/1
 default-information originate metric 100""",
    """router ospf 10
 router-id 2.2.2.2
 network 172.16.0.0 0.0.0.255 area 0
 network 172.16.1.0 0.0.0.255 area 1
 area 1 stub""",
    """router ospf 1
 network 10.10.10.0 0.0.0.255 area 0
 network 10.10.20.0 0.0.0.255 area 0
 auto-cost reference-bandwidth 10000""",
    """router ospf 1
 network 192.168.0.0 0.0.255.255 area 0
 redistribute static subnets
 default-information originate""",
]

CISCO_STATIC = [
    """ip route 0.0.0.0 0.0.0.0 10.0.0.254
ip route 10.1.1.0 255.255.255.0 10.0.0.1
ip route 172.16.0.0 255.255.0.0 10.0.0.2""",
    """ip route 192.168.100.0 255.255.255.0 10.0.0.1 150
ip route 192.168.200.0 255.255.255.0 10.0.0.2 100""",
    """ip route 10.0.0.0 255.0.0.0 Null0
ip route 172.16.0.0 255.240.0.0 10.0.0.254""",
    """ip route 0.0.0.0 0.0.0.0 203.0.113.1
ip route 10.0.0.0 255.255.255.0 10.0.0.254""",
    """ip route 10.1.0.0 255.255.0.0 192.168.1.1
ip route 10.2.0.0 255.255.0.0 192.168.1.2
ip route 0.0.0.0 0.0.0.0 10.0.0.254 10""",
]

CISCO_ACL = [
    """ip access-list extended PERMIT_HTTP
 permit tcp 10.0.0.0 0.0.0.255 any eq 80
 permit tcp 10.0.0.0 0.0.0.255 any eq 443
 deny ip any any""",
    """access-list 100 permit tcp 192.168.1.0 0.0.0.255 any eq 22
access-list 100 permit tcp 192.168.1.0 0.0.0.255 any eq 443
access-list 100 deny ip any any log""",
    """ip access-list extended BLOCK_TELNET
 deny tcp any any eq 23
 permit ip any any""",
    """access-list 1 permit 10.0.0.0 0.0.0.255
access-list 1 deny any""",
    """ip access-list extended RESTRICT_SNMP
 permit udp 10.0.0.0 0.0.0.255 host 10.0.0.10 eq 161
 deny udp any any eq 161
 permit ip any any""",
]

CISCO_ROUTE_MAP = [
    """route-map FILTER-IN permit 10
 match ip address prefix-list ALLOWED_PREFIXES
 set local-preference 200
route-map FILTER-IN deny 20""",
    """route-map SET-MED permit 10
 match ip address 1
 set metric 100
route-map SET-MED permit 20
 set metric 200""",
    """route-map PREFER-PATH permit 10
 match as-path 1
 set local-preference 300
route-map PREFER-PATH permit 20""",
]


# ═══════════════════════════════════════════════════════════
# Juniper 配置样本
# ═══════════════════════════════════════════════════════════

JUNIPER_BGP = [
    """protocols {
    bgp {
        group external {
            type external;
            neighbor 192.168.1.1 {
                peer-as 64512;
                description "Peer-to-ISP-A";
            }
        }
    }
}
routing-options {
    autonomous-system 65000;
}""",
    """protocols {
    bgp {
        group internal {
            type internal;
            local-address 1.1.1.1;
            neighbor 10.0.0.2;
            neighbor 10.0.0.3;
        }
    }
}""",
    """protocols {
    bgp {
        group external-peers {
            type external;
            neighbor 203.0.113.1 {
                peer-as 64500;
                family inet {
                    unicast;
                }
            }
        }
    }
}
routing-options {
    autonomous-system 65000;
}""",
]

JUNIPER_OSPF = [
    """protocols {
    ospf {
        area 0.0.0.0 {
            interface ge-0/0/0 {
                metric 10;
            }
            interface ge-0/0/1 {
                metric 10;
            }
        }
        area 0.0.0.1 {
            interface ge-0/0/2 {
                metric 20;
            }
        }
    }
}""",
    """protocols {
    ospf {
        area 0.0.0.0 {
            interface lo0;
            interface ge-0/0/0 {
                passive;
            }
        }
    }
}""",
    """protocols {
    ospf {
        area 0.0.0.0 {
            interface ge-0/0/0 {
                metric 100;
            }
        }
    }
}
routing-options {
    static {
        route 0.0.0.0/0 {
            next-hop 10.0.0.254;
            preference 150;
        }
    }
}""",
]

JUNIPER_STATIC = [
    """routing-options {
    static {
        route 0.0.0.0/0 {
            next-hop 10.0.0.254;
        }
        route 10.1.1.0/24 {
            next-hop 10.0.0.1;
        }
        route 172.16.0.0/16 {
            next-hop 10.0.0.2;
        }
    }
}""",
    """routing-options {
    static {
        route 192.168.100.0/24 {
            next-hop 10.0.0.1;
            preference 150;
        }
        route 192.168.200.0/24 {
            next-hop 10.0.0.2;
            preference 100;
        }
    }
}""",
    """routing-options {
    static {
        route 10.0.0.0/8 {
            discard;
        }
        route 172.16.0.0/12 {
            next-hop 10.0.0.254;
        }
    }
}""",
]

JUNIPER_ACL = [
    """security {
    policies {
        from-zone trust to-zone untrust {
            policy permit-web {
                match {
                    source-address internal-net;
                    destination-address any;
                    application [ junos-http junos-https ];
                }
                then {
                    permit;
                }
            }
        }
    }
}""",
    """firewall {
    family inet {
        filter restrict-telnet {
            term block-telnet {
                from {
                    protocol tcp;
                    destination-port 23;
                }
                then {
                    discard;
                }
            }
            term allow-all {
                then accept;
            }
        }
    }
}""",
    """security {
    policies {
        from-zone trust to-zone dmz {
            policy allow-ssh {
                match {
                    source-address admin-subnet;
                    destination-address servers;
                    application junos-ssh;
                }
                then {
                    permit;
                }
            }
        }
    }
}""",
]


# ═══════════════════════════════════════════════════════════
# 自然语言描述（用于生成/分析任务）
# ═══════════════════════════════════════════════════════════

NL_DESCRIPTIONS = {
    "bgp": [
        "Configure BGP with AS 65000, establish peering with ISP (AS 64512) at 192.168.1.1, advertise networks 10.0.0.0/24 and 172.16.0.0/16, and apply route filtering policies.",
        "Set up iBGP peering between routers in AS 65001 using loopback interfaces, with next-hop-self configured for all internal neighbors.",
        "Configure BGP to peer with external AS 64500, enable soft-reconfiguration inbound for monitoring, and advertise the 192.168.0.0 network.",
        "Implement BGP traffic engineering by setting different weights for two external peers to prefer one path over another, and apply MED using route-maps.",
        "Configure BGP multicast address family with two external peers for multicast routing between autonomous systems.",
    ],
    "ospf": [
        "Enable OSPF process 1 with router-id 1.1.1.1, advertise three networks across areas 0 and 1, and always originate a default route into OSPF.",
        "Configure OSPF with passive interface on GigabitEthernet0/1 to prevent routing updates on that segment, and originate default route with metric 100.",
        "Set up OSPF with two areas, configuring area 1 as a stub area to reduce LSA flooding and improve convergence time.",
        "Configure OSPF with a custom reference bandwidth of 10Gbps for accurate cost calculation on high-speed links.",
        "Enable OSPF and redistribute static routes into OSPF, also originating a default route for external connectivity.",
    ],
    "static": [
        "Configure three static routes: a default route via 10.0.0.254, a specific route to 10.1.1.0/24 via 10.0.0.1, and a route to 172.16.0.0/16 via 10.0.0.2.",
        "Set up floating static routes with different administrative distances: primary path via 10.0.0.2 (AD 100) and backup via 10.0.0.1 (AD 150).",
        "Configure a null0 route for 10.0.0.0/8 to prevent routing loops, and a default route via 10.0.0.254.",
        "Set up a default route to the ISP at 203.0.113.1 and a static route for internal network 10.0.0.0/24.",
        "Configure two static routes for load balancing across two paths, plus a backup default route with higher administrative distance.",
    ],
    "acl": [
        "Create an extended ACL named PERMIT_HTTP that allows HTTP and HTTPS traffic from 10.0.0.0/24 to any destination, denying all other traffic.",
        "Configure ACL 100 to permit SSH and HTTPS from 192.168.1.0/24, deny all other traffic with logging enabled.",
        "Create an ACL named BLOCK_TELNET that denies Telnet (port 23) traffic while permitting all other protocols.",
        "Configure standard ACL 1 to permit only traffic from 10.0.0.0/24 and deny everything else.",
        "Create an extended ACL RESTRICT_SNMP that permits SNMP queries only from the 10.0.0.0/24 subnet to the SNMP server at 10.0.0.10.",
    ],
}


def generate_cisco_configs() -> List[Dict]:
    """Generate Cisco configuration samples."""
    configs = []
    all_groups = [
        ("bgp", CISCO_BGP),
        ("ospf", CISCO_OSPF),
        ("static", CISCO_STATIC),
        ("acl", CISCO_ACL),
        ("route_policy", CISCO_ROUTE_MAP),
    ]

    for config_type, samples in all_groups:
        for i, config in enumerate(samples):
            configs.append({
                "source": "cisco",
                "doc_type": "sample",
                "url": "sample_data",
                "nl_text": NL_DESCRIPTIONS.get(config_type, [""])[i % len(NL_DESCRIPTIONS.get(config_type, [""]))],
                "config_text": config,
                "config_type": config_type,
            })
    return configs


def generate_juniper_configs() -> List[Dict]:
    """Generate Juniper configuration samples."""
    configs = []
    all_groups = [
        ("bgp", JUNIPER_BGP),
        ("ospf", JUNIPER_OSPF),
        ("static", JUNIPER_STATIC),
        ("acl", JUNIPER_ACL),
    ]

    for config_type, samples in all_groups:
        for i, config in enumerate(samples):
            configs.append({
                "source": "juniper",
                "doc_type": "sample",
                "url": "sample_data",
                "nl_text": NL_DESCRIPTIONS.get(config_type, [""])[i % len(NL_DESCRIPTIONS.get(config_type, [""]))],
                "config_text": config,
                "config_type": config_type,
            })
    return configs


def main():
    print("=" * 60)
    print("Generating Sample Training Data")
    print("=" * 60)

    cisco_configs = generate_cisco_configs()
    juniper_configs = generate_juniper_configs()
    all_configs = cisco_configs + juniper_configs

    # Save individual sources
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_DATA_DIR / "cisco" / "docs_results.json", "w", encoding="utf-8") as f:
        json.dump(cisco_configs, f, ensure_ascii=False, indent=2)

    with open(RAW_DATA_DIR / "juniper" / "docs_results.json", "w", encoding="utf-8") as f:
        json.dump(juniper_configs, f, ensure_ascii=False, indent=2)

    # Save combined
    with open(RAW_DATA_DIR / "all_crawled.json", "w", encoding="utf-8") as f:
        json.dump(all_configs, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated:")
    print(f"  Cisco configs:   {len(cisco_configs)}")
    print(f"  Juniper configs: {len(juniper_configs)}")
    print(f"  Total:           {len(all_configs)}")
    print(f"\nSaved to: {RAW_DATA_DIR / 'all_crawled.json'}")
    print(f"\nConfig type distribution:")
    from collections import Counter
    types = Counter(c["config_type"] for c in all_configs)
    for t, count in types.items():
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
