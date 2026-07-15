"""
Step 1: Data Crawling from Cisco/Juniper Official Docs & Forums
===============================================================
对应论文 Section III-B: Data Mining - 方法一 (HTML Parser)
从厂商官网和社区论坛爬取原始 HTML 数据。

使用方式:
    python scripts/01_data_crawler.py --source cisco --type docs
    python scripts/01_data_crawler.py --source juniper --type forum
    python scripts/01_data_crawler.py --all
"""

import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Cisco configuration documentation URLs (IOS-XE examples)
CISCO_DOC_URLS = [
    # BGP 配置示例 (2025/26 verified)
    "https://www.cisco.com/c/en/us/td/docs/routers/ios/config/17-x/ip-routing/b-ip-routing/m_irg-bgp4.html",
    # OSPF 配置指南 (2025 updated)
    "https://www.cisco.com/c/en/us/td/docs/switches/lan/c9000/lyr3-fwd/ospf/ospf-configuration-guide/ospf.html",
    # ACL 配置指南 (Catalyst 9300)
    "https://www.cisco.com/c/en/us/td/docs/switches/lan/catalyst9300/software/release/17-14/configuration_guide/sec/b_1714_sec_9300_cg/configuring_ipv4_acls.html",
    # 基础 OSPF 配置示例
    "https://www.cisco.com/c/en/us/td/docs/routers/ios/config/17-x/ip-routing/b-ip-routing/m_iro-cfg-0.html",
    # 静态路由配置指南
    "https://www.cisco.com/c/en/us/td/docs/routers/access/wireless/software/guide/StaticRouteConfig.html",
]

# Juniper documentation URLs
JUNIPER_DOC_URLS = [
    # BGP 用户指南 (2026 verified)
    "https://www.juniper.net/documentation/us/en/software/junos/bgp/index.html",
    # BGP 配置概述
    "https://www.juniper.net/documentation/us/en/software/junos/bgp/topics/task/routing-protocol-bgp-security-configuring.html",
    # OSPF 逻辑系统配置示例
    "https://www.juniper.net/documentation/us/en/software/junos/logical-systems/topics/topic-map/logical-systems-ospf.html",
    # OSPF 用户指南
    "https://www.juniper.net/documentation/us/en/software/junos/information-products/pathway-pages/config-guide-routing/config-guide-ospf.html",
]


@dataclass
class CrawlResult:
    """A single crawled (NL description, config snippet) pair."""
    source: str           # "cisco" or "juniper"
    doc_type: str         # "docs" or "forum"
    url: str
    nl_text: str          # Natural language description
    config_text: str      # Configuration snippet
    config_type: str      # "bgp", "ospf", "static", "acl", etc.
    metadata: Dict = field(default_factory=dict)


class CiscoDocsCrawler:
    """Crawl Cisco official documentation pages for config examples."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir / "cisco" / "docs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_page(self, url: str) -> List[CrawlResult]:
        """Extract (description, config) pairs from a Cisco doc page."""
        results = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Cisco docs use <pre> for config blocks and <p> for descriptions
            pre_blocks = soup.find_all("pre")
            for pre in pre_blocks:
                config_text = pre.get_text(strip=True)
                if not self._is_valid_config(config_text):
                    continue

                # Find the nearest preceding <p> tag as the description
                desc_tag = pre.find_previous(["p", "h3", "h4", "h5"])
                nl_text = desc_tag.get_text(strip=True) if desc_tag else ""

                config_type = self._detect_config_type(config_text)
                results.append(CrawlResult(
                    source="cisco",
                    doc_type="docs",
                    url=url,
                    nl_text=nl_text,
                    config_text=config_text,
                    config_type=config_type,
                ))

            logger.info(f"Cisco Docs: Extracted {len(results)} pairs from {url}")
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
        return results

    def _is_valid_config(self, text: str) -> bool:
        """Check if text looks like a valid Cisco config snippet."""
        keywords = ["router", "interface", "ip route", "access-list",
                     "route-map", "prefix-list", "neighbor", "network",
                     "hostname", "ip address", "permit", "deny"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords) and len(text) > 20

    def _detect_config_type(self, text: str) -> str:
        text_lower = text.lower()
        if "router bgp" in text_lower or "neighbor" in text_lower:
            return "bgp"
        elif "router ospf" in text_lower:
            return "ospf"
        elif "ip route" in text_lower and "router" not in text_lower:
            return "static"
        elif "access-list" in text_lower:
            return "acl"
        elif "route-map" in text_lower:
            return "route_policy"
        else:
            return "other"

    def crawl_all(self) -> List[CrawlResult]:
        all_results = []
        for url in CISCO_DOC_URLS:
            results = self.crawl_page(url)
            all_results.extend(results)
            time.sleep(2)  # Rate limiting
        return all_results


class CiscoForumCrawler:
    """Crawl Cisco community forum for config examples."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir / "cisco" / "forum"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_forum_search(self, query: str, max_pages: int = 5) -> List[CrawlResult]:
        """Search Cisco community forum and extract config snippets."""
        results = []
        # Cisco community search API
        base_url = "https://community.cisco.com/t5/forums/searchpage/tab/message"
        params = {
            "q": query,
            "page": 0,
        }
        for page in range(max_pages):
            try:
                params["page"] = page
                resp = self.session.get(base_url, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                soup = BeautifulSoup(resp.text, "lxml")

                # Extract forum post content
                posts = soup.find_all("div", class_="lia-message-body")
                for post in posts:
                    text = post.get_text()
                    configs = self._extract_config_from_text(text)
                    for cfg in configs:
                        results.append(CrawlResult(
                            source="cisco",
                            doc_type="forum",
                            url=base_url,
                            nl_text=self._extract_context(text, cfg),
                            config_text=cfg,
                            config_type=self._detect_config_type(cfg),
                        ))
                time.sleep(3)
            except Exception as e:
                logger.error(f"Forum crawl error page {page}: {e}")
                break
        logger.info(f"Cisco Forum: Extracted {len(results)} config snippets for query '{query}'")
        return results

    def _extract_config_from_text(self, text: str) -> List[str]:
        """Extract config blocks from forum text (heuristic)."""
        configs = []
        lines = text.split("\n")
        current_config = []
        in_config = False

        for line in lines:
            stripped = line.strip()
            if self._looks_like_config_line(stripped):
                in_config = True
                current_config.append(stripped)
            elif in_config:
                if stripped == "" or not self._looks_like_config_line(stripped):
                    if len(current_config) >= 3:
                        configs.append("\n".join(current_config))
                    current_config = []
                    in_config = False

        if len(current_config) >= 3:
            configs.append("\n".join(current_config))
        return configs

    def _looks_like_config_line(self, line: str) -> bool:
        if not line or len(line) < 5:
            return False
        config_prefixes = [
            "interface", "ip ", "router", "network", "neighbor",
            "access-list", "route-map", "prefix-list", "hostname",
            "!", "permit", "deny", "set ", "match ", "no "
        ]
        return any(line.lower().startswith(p) for p in config_prefixes)

    def _extract_context(self, full_text: str, config: str) -> str:
        """Extract the natural language context around a config block."""
        idx = full_text.find(config[:30])
        if idx < 0:
            return ""
        start = max(0, idx - 300)
        context = full_text[start:idx].strip()
        # Take last paragraph as context
        paragraphs = context.split("\n")
        return paragraphs[-1].strip() if paragraphs else ""

    def _detect_config_type(self, text: str) -> str:
        text_lower = text.lower()
        if "router bgp" in text_lower:
            return "bgp"
        elif "router ospf" in text_lower:
            return "ospf"
        elif "ip route" in text_lower:
            return "static"
        elif "access-list" in text_lower:
            return "acl"
        return "other"

    def crawl_all(self) -> List[CrawlResult]:
        queries = ["BGP configuration", "OSPF configuration", "static route",
                    "access-list", "route-map", "network configuration"]
        all_results = []
        for q in queries:
            results = self.crawl_forum_search(q, max_pages=3)
            all_results.extend(results)
            time.sleep(3)
        return all_results


class JuniperDocsCrawler:
    """Crawl Juniper documentation for config examples."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir / "juniper" / "docs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_page(self, url: str) -> List[CrawlResult]:
        results = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Juniper docs use <code> or <pre> for config blocks
            code_blocks = soup.find_all(["code", "pre"])
            for block in code_blocks:
                config_text = block.get_text(strip=True)
                if not self._is_juniper_config(config_text):
                    continue

                desc_tag = block.find_previous(["p", "h3", "h4", "h5"])
                nl_text = desc_tag.get_text(strip=True) if desc_tag else ""

                results.append(CrawlResult(
                    source="juniper",
                    doc_type="docs",
                    url=url,
                    nl_text=nl_text,
                    config_text=config_text,
                    config_type=self._detect_config_type(config_text),
                ))
            logger.info(f"Juniper Docs: Extracted {len(results)} pairs from {url}")
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
        return results

    def _is_juniper_config(self, text: str) -> bool:
        juniper_keywords = ["set ", "protocols", "routing-options", "interfaces",
                            "security", "policy-options", "firewall"]
        return any(kw in text.lower() for kw in juniper_keywords) and len(text) > 20

    def _detect_config_type(self, text: str) -> str:
        text_lower = text.lower()
        if "bgp" in text_lower:
            return "bgp"
        elif "ospf" in text_lower:
            return "ospf"
        elif "static" in text_lower or "routing-options" in text_lower:
            return "static"
        elif "security" in text_lower or "firewall" in text_lower:
            return "acl"
        return "other"

    def crawl_all(self) -> List[CrawlResult]:
        all_results = []
        for url in JUNIPER_DOC_URLS:
            results = self.crawl_page(url)
            all_results.extend(results)
            time.sleep(2)
        return all_results


class JuniperForumCrawler:
    """Crawl Juniper community forum for config examples."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir / "juniper" / "forum"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_all(self) -> List[CrawlResult]:
        # Similar structure to CiscoForumCrawler
        queries = ["BGP junos configuration", "OSPF junos setup",
                    "static route junos", "firewall filter junos"]
        all_results = []
        for q in queries:
            # Juniper community search
            try:
                url = f"https://community.juniper.net/search?q={q.replace(' ', '+')}"
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "lxml")
                    posts = soup.find_all("div", class_="message-body")
                    for post in posts:
                        text = post.get_text()
                        config = self._extract_juniper_config(text)
                        if config:
                            all_results.append(CrawlResult(
                                source="juniper",
                                doc_type="forum",
                                url=url,
                                nl_text=text[:300],
                                config_text=config,
                                config_type=self._detect_config_type(config),
                            ))
            except Exception as e:
                logger.error(f"Juniper forum error for '{q}': {e}")
            time.sleep(3)
        logger.info(f"Juniper Forum: Extracted {len(all_results)} snippets total")
        return all_results

    def _extract_juniper_config(self, text: str) -> str:
        lines = text.split("\n")
        config_lines = [l.strip() for l in lines if l.strip().startswith("set ")]
        return "\n".join(config_lines) if len(config_lines) >= 3 else ""

    def _detect_config_type(self, text: str) -> str:
        if "protocols bgp" in text:
            return "bgp"
        elif "protocols ospf" in text:
            return "ospf"
        elif "routing-options" in text:
            return "static"
        return "other"


def save_results(results: List[CrawlResult], filepath: Path):
    """Save crawl results to JSON."""
    data = [asdict(r) for r in results]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} results to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="PreConfig Data Crawler")
    parser.add_argument("--source", choices=["cisco", "juniper", "all"], default="all")
    parser.add_argument("--type", choices=["docs", "forum", "all"], default="all")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    if args.source in ["cisco", "all"]:
        if args.type in ["docs", "all"]:
            crawler = CiscoDocsCrawler(RAW_DATA_DIR)
            results = crawler.crawl_all()
            all_results.extend(results)
            save_results(results, RAW_DATA_DIR / "cisco" / "docs_results.json")

        if args.type in ["forum", "all"]:
            crawler = CiscoForumCrawler(RAW_DATA_DIR)
            results = crawler.crawl_all()
            all_results.extend(results)
            save_results(results, RAW_DATA_DIR / "cisco" / "forum_results.json")

    if args.source in ["juniper", "all"]:
        if args.type in ["docs", "all"]:
            crawler = JuniperDocsCrawler(RAW_DATA_DIR)
            results = crawler.crawl_all()
            all_results.extend(results)
            save_results(results, RAW_DATA_DIR / "juniper" / "docs_results.json")

        if args.type in ["forum", "all"]:
            crawler = JuniperForumCrawler(RAW_DATA_DIR)
            results = crawler.crawl_all()
            all_results.extend(results)
            save_results(results, RAW_DATA_DIR / "juniper" / "forum_results.json")

    # Save all combined
    save_results(all_results, RAW_DATA_DIR / "all_crawled.json")
    logger.info(f"Total crawled: {len(all_results)} pairs")


if __name__ == "__main__":
    main()
