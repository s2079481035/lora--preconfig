"""
修正版数据爬虫：从 Cisco/Juniper 公开资源获取配置数据
============================================================
使用以下来源（URL 稳定，不易 404）：

Cisco:
  1. Cisco DevNet GitHub (官方配置示例仓库)
  2. Cisco Configuration Professional (CCP) 示例
  3. Cisco Community API

Juniper:
  1. Juniper GitHub (官方配置示例)
  2. Juniper TechLibrary 公开页面
  3. Juniper vSRX 配置示例

使用方式:
    python scripts/01c_real_crawler.py --source cisco
    python scripts/01c_real_crawler.py --source juniper
    python scripts/01c_real_crawler.py --source all
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class ConfigPair:
    """一个 (NL描述, 配置片段) 配对."""
    source: str           # "cisco" / "juniper"
    doc_type: str         # "docs" / "github" / "forum"
    url: str
    nl_text: str          # 自然语言描述
    config_text: str      # 配置片段
    config_type: str      # "bgp" / "ospf" / "static" / "acl" / "route_policy"


# ═══════════════════════════════════════════════════════════
# 1. Cisco DevNet GitHub 配置示例
# ═══════════════════════════════════════════════════════════

CISCO_GITHUB_REPOS = [
    # Cisco DevNet 官方示例仓库
    "https://api.github.com/repos/CiscoDevNet/ios-xe-config-examples/contents",
    "https://api.github.com/repos/CiscoDevNet/xe-config-examples/contents",
    # Cisco 网络自动化示例
    "https://api.github.com/repos/scottcress/cisco-config-examples/contents",
    "https://api.github.com/search/code?q=cisco+router+config+bgp+language:IOS&per_page=50",
]

JUNIPER_GITHUB_REPOS = [
    # Juniper 官方配置示例
    "https://api.github.com/repos/Juniper/junos-automation-examples/contents",
    "https://api.github.com/repos/Juniper/junos-examples/contents",
    # GitHub 搜索 Juniper 配置
    "https://api.github.com/search/code?q=juniper+config+set+protocols+language:Junos&per_page=50",
]


# ═══════════════════════════════════════════════════════════
# 2. Cisco Community / DevNet API
# ═══════════════════════════════════════════════════════════

CISCO_COMMUNITY_URLS = {
    "bgp": "https://community.cisco.com/t5/forums/searchpage/tab/message?q=BGP+configuration+example&sort_by=-topicPostDate&collapse_discussion=true",
    "ospf": "https://community.cisco.com/t5/forums/searchpage/tab/message?q=OSPF+configuration+example&sort_by=-topicPostDate&collapse_discussion=true",
    "acl": "https://community.cisco.com/t5/forums/searchpage/tab/message?q=access-list+configuration+example&sort_by=-topicPostDate&collapse_discussion=true",
}

JUNIPER_COMMUNITY_URLS = {
    "bgp": "https://community.juniper.net/search?q=BGP+junos+configuration+example+set+protocols",
    "ospf": "https://community.juniper.net/search?q=OSPF+junos+configuration+example+set+protocols",
    "acl": "https://community.juniper.net/search?q=firewall+filter+junos+configuration+example",
}


# ═══════════════════════════════════════════════════════════
# 3. 直接从 HTML 页面爬取配置教程
# ═══════════════════════════════════════════════════════════

CISCO_CONFIG_GUIDE_URLS = [
    # Cisco 基础知识配置指南（URL 相对稳定）
    "https://www.cisco.com/c/en/us/support/docs/ip/border-gateway-protocol-bgp/13753-25.html",
    "https://www.cisco.com/c/en/us/support/docs/ip/open-shortest-path-first-ospf/13684-12.html",
    "https://www.cisco.com/c/en/us/support/docs/security/ios-firewall/23602-confaccesslists.html",
    "https://www.cisco.com/c/en/us/support/docs/ip/routing-information-protocol-rip/16448-default.html",
    "https://www.cisco.com/c/en/us/support/docs/ip/border-gateway-protocol-bgp/13759-23.html",
    "https://www.cisco.com/c/en/us/support/docs/ip/open-shortest-path-first-ospf/13689-13.html",
]

JUNIPER_CONFIG_GUIDE_URLS = [
    "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/index.html",
    "https://www.juniper.net/documentation/us/en/software/junos/junos-getting-started/index.html",
]


def detect_config_type(text: str) -> str:
    """检测配置类型."""
    t = text.lower()
    if "router bgp" in t or "bgp" in t:
        return "bgp"
    elif "router ospf" in t or "ospf" in t:
        return "ospf"
    elif "access-list" in t or "firewall" in t:
        return "acl"
    elif "ip route" in t and "router" not in t:
        return "static"
    elif "route-map" in t or "route map" in t:
        return "route_policy"
    return "general"


def is_valid_config(text: str, vendor: str = "cisco") -> bool:
    """检查是否有效配置片段."""
    if not text or len(text) < 20:
        return False

    cisco_kw = ["router", "interface", "ip ", "access-list", "route-map",
                 "prefix-list", "neighbor", "network", "!"]
    juniper_kw = ["set ", "protocols", "routing-options", "interfaces {",
                  "security {", "policy-options", "firewall {"]

    keywords = juniper_kw if "juniper" in vendor.lower() else cisco_kw
    return any(kw in text.lower() for kw in keywords)


def extract_config_blocks(html_text: str, vendor: str = "cisco") -> List[str]:
    """从 HTML 文本中提取配置代码块."""
    configs = []

    if vendor == "cisco":
        # Cisco 配置通常在 <pre> 标签中
        pre_blocks = re.findall(r"<pre[^>]*>(.*?)</pre>", html_text, re.DOTALL)
        for block in pre_blocks:
            cleaned = block.strip()
            cleaned = re.sub(r"<.*?>", "", cleaned)  # 移除 HTML 标签
            if is_valid_config(cleaned, "cisco"):
                configs.append(cleaned)
    else:
        # Juniper 配置通常在 <code> 或 <div class="cli">
        code_blocks = re.findall(r"<code[^>]*>(.*?)</code>", html_text, re.DOTALL)
        for block in code_blocks:
            cleaned = block.strip()
            cleaned = re.sub(r"<.*?>", "", cleaned)
            if is_valid_config(cleaned, "juniper"):
                configs.append(cleaned)

        # Juniper "set" 格式配置
        set_lines = re.findall(r"set [\w-]+.*", html_text)
        if len(set_lines) >= 3:
            configs.append("\n".join(set_lines))

    return configs


def extract_nearby_text(html_text: str, config_block: str) -> str:
    """提取配置附近的一段自然语言描述."""
    # 取配置前 200-500 字符作为描述
    idx = html_text.find(config_block[:30])
    if idx < 0:
        return ""
    start = max(0, idx - 500)
    context = html_text[start:idx]
    # 用 BeautifulSoup 提取纯文本
    soup = BeautifulSoup(context, "lxml")
    text = soup.get_text()
    # 取最后一段作为描述
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    return paragraphs[-1] if paragraphs else ""


class GitHubConfigCrawler:
    """从 GitHub 爬取配置示例."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PreConfig-Research-Crawler/1.0",
            "Accept": "application/vnd.github.v3+json",
        })

    def search_github_code(self, query: str, max_results: int = 50) -> List[str]:
        """搜索 GitHub 代码."""
        url = f"https://api.github.com/search/code?q={query}&per_page={min(max_results, 50)}"
        configs = []

        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 403:
                logger.warning(f"GitHub API rate limit exceeded. Try with token.")
                return []
            if resp.status_code != 200:
                logger.warning(f"GitHub search failed: {resp.status_code}")
                return []

            data = resp.json()
            for item in data.get("items", []):
                html_url = item.get("html_url", "")
                # 获取文件内容
                raw_url = html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                try:
                    content_resp = self.session.get(raw_url, timeout=15)
                    if content_resp.status_code == 200:
                        content = content_resp.text
                        # 提取配置块
                        lines = content.split("\n")
                        config_lines = []
                        in_config = False
                        for line in lines:
                            if is_valid_config(line.strip()):
                                in_config = True
                                config_lines.append(line.strip())
                            elif in_config and line.strip() == "":
                                if len(config_lines) >= 3:
                                    configs.append("\n".join(config_lines))
                                config_lines = []
                                in_config = False
                        if len(config_lines) >= 3:
                            configs.append("\n".join(config_lines))
                except:
                    continue
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"GitHub search error: {e}")

        logger.info(f"GitHub '{query}': found {len(configs)} config blocks")
        return configs

    def crawl_all_cisco(self) -> List[ConfigPair]:
        """从 GitHub 爬取 Cisco 配置."""
        pairs = []
        queries = [
            "router+bgp+neighbor+remote-as+filename:rtr OR filename:cfg",
            "router+ospf+network+area+filename:rtr OR filename:cfg",
            "ip+route+255+filename:rtr OR filename:cfg",
            "access-list+permit OR deny+filename:rtr OR filename:cfg",
            "route-map+match+set+filename:rtr OR filename:cfg",
        ]

        for query in queries:
            configs = self.search_github_code(f"cisco+{query}")
            for cfg in configs:
                pairs.append(ConfigPair(
                    source="cisco",
                    doc_type="github",
                    url=f"https://github.com/search?q=cisco+{query}",
                    nl_text="",
                    config_text=cfg,
                    config_type=detect_config_type(cfg),
                ))
            time.sleep(2)

        return pairs

    def crawl_all_juniper(self) -> List[ConfigPair]:
        """从 GitHub 爬取 Juniper 配置."""
        pairs = []
        queries = [
            "set+protocols+bgp+group",
            "set+protocols+ospf+area",
            "set+routing-options+static+route",
            "set+security+policies",
            "set+firewall+family+inet+filter",
        ]

        for query in queries:
            configs = self.search_github_code(f"{query}+filename:junos OR filename:conf")
            for cfg in configs:
                pairs.append(ConfigPair(
                    source="juniper",
                    doc_type="github",
                    url=f"https://github.com/search?q={query}",
                    nl_text="",
                    config_text=cfg,
                    config_type=detect_config_type(cfg),
                ))
            time.sleep(2)

        return pairs


class DocsGuideCrawler:
    """从 Cisco/Juniper 配置指南页面爬取."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_page(self, url: str, vendor: str) -> List[ConfigPair]:
        """爬取单个配置指南页面."""
        pairs = []
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return []

            html = resp.text
            configs = extract_config_blocks(html, vendor)

            for cfg in configs:
                nl = extract_nearby_text(html, cfg)
                pairs.append(ConfigPair(
                    source=vendor,
                    doc_type="docs",
                    url=url,
                    nl_text=nl,
                    config_text=cfg,
                    config_type=detect_config_type(cfg),
                ))

            logger.info(f"{vendor} Guides [{url.split('/')[3]}]: extracted {len(pairs)} configs")
        except Exception as e:
            logger.error(f"Failed {url}: {e}")

        return pairs

    def crawl_all_cisco(self) -> List[ConfigPair]:
        """爬取所有 Cisco 配置指南."""
        all_pairs = []
        for url in CISCO_CONFIG_GUIDE_URLS:
            pairs = self.crawl_page(url, "cisco")
            all_pairs.extend(pairs)
            time.sleep(2)
        return all_pairs

    def crawl_all_juniper(self) -> List[ConfigPair]:
        """爬取所有 Juniper 配置指南."""
        all_pairs = []
        for url in JUNIPER_CONFIG_GUIDE_URLS:
            pairs = self.crawl_page(url, "juniper")
            all_pairs.extend(pairs)
            time.sleep(2)
        return all_pairs


class ForumCrawler:
    """从 Cisco/Juniper 社区论坛爬取配置讨论."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_cisco_community(self, config_type: str) -> List[ConfigPair]:
        """爬取 Cisco 社区论坛."""
        url = CISCO_COMMUNITY_URLS.get(config_type, "")
        if not url:
            return []

        pairs = []
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "lxml")
            # 提取帖子内容
            messages = soup.find_all("div", class_="lia-message-body")
            for msg in messages:
                text = msg.get_text()
                # 提取配置块
                config_blocks = self._extract_cisco_configs(text)
                for cfg in config_blocks:
                    # 取配置前的文本作为描述
                    idx = text.find(cfg[:30])
                    context = text[max(0, idx - 300):idx] if idx >= 0 else ""
                    pairs.append(ConfigPair(
                        source="cisco",
                        doc_type="forum",
                        url=url,
                        nl_text=context.strip()[-300:],
                        config_text=cfg,
                        config_type=config_type,
                    ))

            logger.info(f"Cisco Forum [{config_type}]: {len(pairs)} configs")
        except Exception as e:
            logger.error(f"Cisco forum error: {e}")

        return pairs

    def _extract_cisco_configs(self, text: str) -> List[str]:
        """从论坛文本中提取 Cisco 配置."""
        configs = []
        lines = text.split("\n")
        current = []
        in_config = False

        cisco_prefixes = ["interface", "ip ", "router", "network", "neighbor",
                          "access-list", "route-map", "hostname", "!"]

        for line in lines:
            stripped = line.strip()
            if any(stripped.lower().startswith(p) for p in cisco_prefixes):
                in_config = True
                current.append(stripped)
            elif in_config:
                if not stripped or stripped.startswith(("Hello", "Hi", "Thanks", "---", "***")):
                    if len(current) >= 3:
                        configs.append("\n".join(current))
                    current = []
                    in_config = False
                else:
                    current.append(stripped)

        if len(current) >= 3:
            configs.append("\n".join(current))
        return configs


def save_results(pairs: List[ConfigPair], filepath: Path):
    """保存结果."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in pairs]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} pairs to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Config Data Crawler v2")
    parser.add_argument("--source", choices=["cisco", "juniper", "all"], default="all")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub search (API limit)")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    if args.source in ["cisco", "all"]:
        cisco_pairs = []

        # 来源 1: Cisco 配置指南
        logger.info("=" * 50)
        logger.info("Cisco: Configuration Guides")
        logger.info("=" * 50)
        docs = DocsGuideCrawler()
        cisco_pairs.extend(docs.crawl_all_cisco())

        # 来源 2: Cisco 社区论坛
        logger.info("=" * 50)
        logger.info("Cisco: Community Forum")
        logger.info("=" * 50)
        forum = ForumCrawler()
        for ctype in ["bgp", "ospf", "acl"]:
            cisco_pairs.extend(forum.crawl_cisco_community(ctype))
            time.sleep(3)

        # 来源 3: GitHub（有 API 频率限制）
        if not args.skip_github:
            logger.info("=" * 50)
            logger.info("Cisco: GitHub Search")
            logger.info("=" * 50)
            github = GitHubConfigCrawler()
            cisco_pairs.extend(github.crawl_all_cisco())

        save_results(cisco_pairs, RAW_DATA_DIR / "cisco" / "docs_results.json")
        all_pairs.extend(cisco_pairs)

    if args.source in ["juniper", "all"]:
        juniper_pairs = []

        # 来源 1: Juniper 配置指南
        logger.info("=" * 50)
        logger.info("Juniper: Configuration Guides")
        logger.info("=" * 50)
        docs = DocsGuideCrawler()
        juniper_pairs.extend(docs.crawl_all_juniper())

        # 来源 2: GitHub
        if not args.skip_github:
            logger.info("=" * 50)
            logger.info("Juniper: GitHub Search")
            logger.info("=" * 50)
            github = GitHubConfigCrawler()
            juniper_pairs.extend(github.crawl_all_juniper())

        save_results(juniper_pairs, RAW_DATA_DIR / "juniper" / "docs_results.json")
        all_pairs.extend(juniper_pairs)

    # 保存汇总
    save_results(all_pairs, RAW_DATA_DIR / "all_crawled.json")

    # 统计
    print("\n" + "=" * 60)
    print(f"Total crawled: {len(all_pairs)} config pairs")
    types = Counter(p.config_type for p in all_pairs)
    for t, c in types.items():
        print(f"  {t}: {c}")
    print("=" * 60)


if __name__ == "__main__":
    main()