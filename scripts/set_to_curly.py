"""
Junos `set` 命令 → 花括号层级格式转换器
用于把 NIT 数据集的 set 风格配置统一成 pipeline 的花括号风格。

算法: set 命令 token 流按三类处理
  - 实例关键词 (interface, unit, vlan, community, ...): 与后一个 token 组成节点
  - 值关键词 (port-mode, members, peer-as, ...): 与后一个 token 组成叶子
  - 纯节点 (interfaces, protocols, system, ...): 单独成节点
"""

import re

# 实例关键词: 后跟实例名组成 "keyword instance" 节点
INSTANCE_KEYWORDS = {
    "interface", "unit", "vlan", "community", "policy-statement", "term",
    "group", "neighbor", "prefix-list", "trap-group", "traceoptions", "peer",
    "route", "instance", "station", "apply-groups", "top", "aggr", "user",
    "chassis-group", "redundancy-group", "bfd-liveness-detection", "route-filter",
    "icmp-type", "icmp-code", "protocol", "family", "switch-options",
}

# 值关键词: 后跟一个值组成 "key value;" 叶子
VALUE_KEYWORDS = {
    "members", "interval", "limit", "port-mode", "categories", "link", "address",
    "peer-as", "autonomous-system", "prefix", "color", "metric", "preference",
    "priority", "timeout", "mode", "status", "value", "disable-timeout",
    "mac-move-limit", "aging-time", "source-address", "destination", "level",
    "type", "name", "class", "bandwidth", "rate", "traffic-type", "role",
    "inactivity-timeout", "recover", "options", "trusted",
    "passive", "enable", "both", "only", "no", "disable", "family", "severity",
    "password", "key", "secret", "alarm", "ignore", "link-down", "dhcp-trusted",
    "no-root-port", "root-protect", "auto-conversion", "auto-image-upgrade",
    "allow-duplicates", "source", "destination-address", "next-hop", "tag",
}

# 值模式: 数字 / IP / 前缀 / MAC / 占位符 / 大写常量
VALUE_RE = re.compile(
    r"^(<[^>]+>|\d+([./:]\d+){0,4}([/%]\d+)?|[A-Z][A-Z0-9._/-]*|[a-f0-9]{2}(:[a-f0-9]{2})+|0x[0-9a-fA-F]+)$"
)

def _is_value(tok: str) -> bool:
    if tok in VALUE_KEYWORDS:
        return False
    return bool(VALUE_RE.match(tok))

def set_to_curly(cmd: str, indent: str = "    ") -> str:
    """把单条 set 命令转成花括号层级格式。"""
    cmd = cmd.strip()
    if not cmd.startswith("set "):
        return cmd
    tokens = cmd[4:].strip().split() if len(cmd) > 4 else []
    if not tokens:
        return ""

    lines = []
    depth = 0
    stack = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < n else None

        # 最后一个 token: 叶子
        if nxt is None:
            lines.append(indent * depth + f"{tok};")
            break

        # 实例关键词 + 实例名 (实例名不能是关键词)
        if tok in INSTANCE_KEYWORDS and nxt not in INSTANCE_KEYWORDS and nxt not in VALUE_KEYWORDS:
            lines.append(indent * depth + f"{tok} {nxt} {{")
            depth += 1
            i += 2
            continue
        # 值关键词 + 值
        if tok in VALUE_KEYWORDS:
            lines.append(indent * depth + f"{tok} {nxt};")
            i += 2
            continue
        # 纯节点
        lines.append(indent * depth + f"{tok} {{")
        depth += 1
        i += 1

    # 闭合
    while depth > 0:
        depth -= 1
        lines.append(indent * depth + "}")

    return "\n".join(lines)


def convert_set_config(config: str) -> str:
    """把整段 set 风格配置转成花括号格式。"""
    blocks = []
    for raw in config.split("\n"):
        line = raw.strip()
        if line.startswith("set "):
            converted = set_to_curly(line)
            if converted:
                blocks.append(converted)
    return "\n\n".join(blocks)
