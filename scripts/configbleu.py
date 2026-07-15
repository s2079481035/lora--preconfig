"""
ConfigBLEU Core Module (importable)
======================================
ConfigBLEU = alpha * BLEU + beta * BLEU_weight + gamma * Match_syn

Usage:
    from scripts.configbleu import compute_all_metrics, compute_configbleu
"""

import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
from dataclasses import dataclass

CONFIG_KEYWORDS = {
    "router", "interface", "ip", "access-list", "route-map", "prefix-list",
    "neighbor", "network", "permit", "deny", "match", "set", "redistribute",
    "address-family", "default-information", "hostname", "no", "bgp", "ospf",
    "static", "vlan", "switchport", "spanning-tree", "ntp", "snmp",
    "protocols", "routing-options", "interfaces", "security", "policy-options",
    "firewall", "groups", "system", "chassis", "class-of-service",
    "forwarding-options", "services",
}

KEYWORD_WEIGHT = 5.0


@dataclass
class ASTNode:
    name: str
    children: List['ASTNode']
    value: str = ""
    depth: int = 0

    def __hash__(self):
        return hash((self.name, self.value, self.depth))

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value and self.depth == other.depth


class ConfigASTParser:
    def parse_cisco(self, config: str) -> ASTNode:
        root = ASTNode(name="root", children=[], depth=0)
        indent_stack = [(-1, root)]
        for line in config.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            indent = len(line) - len(line.lstrip())
            parts = stripped.split()
            if not parts:
                continue
            node = ASTNode(
                name=parts[0], children=[],
                value=" ".join(parts[1:]) if len(parts) > 1 else "",
                depth=indent,
            )
            while len(indent_stack) > 1 and indent_stack[-1][0] >= indent:
                indent_stack.pop()
            parent = indent_stack[-1][1]
            parent.children.append(node)
            indent_stack.append((indent, node))
        return root

    def parse_juniper(self, config: str) -> ASTNode:
        root = ASTNode(name="root", children=[], depth=0)
        stack = [root]
        for line in config.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("set "):
                parts = stripped[4:].split()
                current = root
                for part in parts:
                    if part in ("{", "}"):
                        continue
                    child = None
                    for c in current.children:
                        if c.name == part:
                            child = c
                            break
                    if child is None:
                        child = ASTNode(name=part, children=[], depth=len(stack))
                        current.children.append(child)
                    current = child
            elif "{" in stripped:
                parts = stripped.replace("{", "").strip().split()
                if parts:
                    node = ASTNode(name=parts[0], children=[], depth=len(stack))
                    stack[-1].children.append(node)
                    stack.append(node)
            elif stripped == "}":
                if len(stack) > 1:
                    stack.pop()
        return root

    def parse(self, config: str, vendor: str = "auto") -> ASTNode:
        if vendor == "auto":
            if any(kw in config.lower() for kw in ["set ", "protocols", "routing-options"]):
                vendor = "juniper"
            else:
                vendor = "cisco"
        if vendor == "juniper":
            return self.parse_juniper(config)
        return self.parse_cisco(config)


def extract_subtrees(node: ASTNode, max_depth: int = 5) -> Set[Tuple]:
    subtrees = set()
    def _extract(n: ASTNode, depth: int):
        if depth > max_depth:
            return
        child_names = tuple(sorted(c.name for c in n.children))
        subtrees.add((n.name, child_names, n.depth))
        for child in n.children:
            _extract(child, depth + 1)
    _extract(node, 0)
    return subtrees


def compute_ast_match(candidate: str, reference: str, vendor: str = "auto") -> float:
    parser = ConfigASTParser()
    cand_tree = parser.parse(candidate, vendor)
    ref_tree = parser.parse(reference, vendor)
    cand_subtrees = extract_subtrees(cand_tree)
    ref_subtrees = extract_subtrees(ref_tree)
    if not ref_subtrees:
        return 0.0
    intersection = cand_subtrees & ref_subtrees
    return len(intersection) / len(ref_subtrees)


def tokenize_config(text: str) -> List[str]:
    return re.findall(r"\b\w[\w-]*\b|[{}();]", text.lower())


def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return ngrams


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand_tokens = tokenize_config(candidate)
    ref_tokens = tokenize_config(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = compute_ngram_counts(cand_tokens, n)
        ref_ngrams = compute_ngram_counts(ref_tokens, n)
        clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(clipped / total)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def compute_weighted_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    cand_tokens = tokenize_config(candidate)
    ref_tokens = tokenize_config(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    def token_weight(token: str) -> float:
        return KEYWORD_WEIGHT if token in CONFIG_KEYWORDS else 1.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = compute_ngram_counts(cand_tokens, n)
        ref_ngrams = compute_ngram_counts(ref_tokens, n)
        weighted_clipped = 0.0
        weighted_total = 0.0
        for ng, count in cand_ngrams.items():
            w = sum(token_weight(t) for t in ng) / n
            clipped = min(count, ref_ngrams.get(ng, 0))
            weighted_clipped += w * clipped
            weighted_total += w * count
        if weighted_total > 0:
            precisions.append(weighted_clipped / weighted_total)
        else:
            precisions.append(0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def compute_configbleu(
    candidate: str, reference: str,
    alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
    vendor: str = "auto",
) -> Dict[str, float]:
    bleu = compute_bleu(candidate, reference)
    bleu_weight = compute_weighted_bleu(candidate, reference)
    match_syn = compute_ast_match(candidate, reference, vendor)
    config_bleu = alpha * bleu + beta * bleu_weight + gamma * match_syn
    return {
        "config_bleu": round(config_bleu, 4),
        "bleu": round(bleu, 4),
        "bleu_weight": round(bleu_weight, 4),
        "match_syn": round(match_syn, 4),
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }


def compute_rouge_l(candidate: str, reference: str) -> float:
    cand_tokens = tokenize_config(candidate)
    ref_tokens = tokenize_config(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0
    m, n = len(cand_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if cand_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / max(m, 1)
    recall = lcs_len / max(n, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_meteor(candidate: str, reference: str) -> float:
    cand_tokens = set(tokenize_config(candidate))
    ref_tokens = set(tokenize_config(reference))
    if not cand_tokens or not ref_tokens:
        return 0.0
    matches = cand_tokens & ref_tokens
    precision = len(matches) / max(len(cand_tokens), 1)
    recall = len(matches) / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(candidate: str, reference: str, vendor: str = "auto") -> Dict[str, float]:
    config_bleu_results = compute_configbleu(candidate, reference, vendor=vendor)
    rouge_l = compute_rouge_l(candidate, reference)
    meteor = compute_meteor(candidate, reference)
    return {
        **config_bleu_results,
        "rouge_l": round(rouge_l, 4),
        "meteor": round(meteor, 4),
    }
