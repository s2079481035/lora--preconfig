"""
Sanitize reference configurations: replace concrete parameters with placeholders.
保留结构, 去除参数干扰 (hostname/IP/AS/端口/密码), 防止模型照抄参考参数。
"""

import re

_PATTERNS = [
    # hostname (Cisco / Juniper)
    (re.compile(r"(hostname\s+)\S+", re.I), r"\1<HOSTNAME>"),
    (re.compile(r"(host-name\s+)\S+;", re.I), r"\1<HOSTNAME>;"),
    # 私有网段 IPv4 (10.x / 192.168.x / 172.16-31.x), 含可选中缀 /len
    (re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?(?:/\d{1,2})?\b"),
     "<IP>"),
    # netmask
    (re.compile(r"\b255\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<MASK>"),
    # AS 号 / remote-as / peer-as (仅在这些关键字后)
    (re.compile(r"\b(router bgp|autonomous-system|peer-as|remote-as)\s+\d+\b", re.I),
     r"\1 <AS>"),
    # 端口号
    (re.compile(r"(destination-port\s+)\d+\b", re.I), r"\1<PORT>"),
    # 密码
    (re.compile(r"(enable secret\s+)\S+", re.I), r"\1<PASSWORD>"),
    (re.compile(r"(authentication-password\s+)\S+;", re.I), r"\1<PASSWORD>;"),
]


def sanitize_config(cfg: str) -> str:
    for pattern, repl in _PATTERNS:
        cfg = pattern.sub(repl, cfg)
    return cfg