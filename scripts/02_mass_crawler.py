#!/usr/bin/env python3
"""
大规模配置数据爬虫 v3 — 专为 PreConfig 复现设计
=================================================
目标: 爬取 GitHub 上所有 IPv4/IPv6 路由器的真实配置文件，
     覆盖 Cisco IOS、Juniper Junos 等主流厂商格式。

来源:
 1. GitHub Code Search — 搜索 router config 代码仓库 (主力，量最大)
 2. Cisco/Juniper 技术支持文档 — 抽取配置模板和示例
 3. 网络自动化开源项目中的配置模板 (如 Ansible/NAPALM 仓库)

爬取策略: 分层搜索 + 去重 + 按厂商/协议分类

VPN 要求: GitHub API 无墙但有 rate limit (60次/小时无token, 5000次/小时有token)

使用方式:
    python scripts/02_mass_crawler.py                     # 默认全量
    python scripts/02_mass_crawler.py --source github     # 仅GitHub
    python scripts/02_mass_crawler.py --source docs       # 仅文档
    python scripts/02_mass_crawler.py --max-pages 20      # 限制搜索页数
"""

import os, re, json, time, hashlib, logging, argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ============================================================
#  GitHub 代码搜索关键词 — 按厂商 + 协议 + 配置类型分层
# ============================================================
GITHUB_SEARCH_QUERIES = {
    "cisco": {
        "bgp": [
            "router bgp neighbor remote-as Cisco",
            "router bgp network mask Cisco configuration",
            "router bgp address-family ipv4 Cisco",
        ],
        "ospf": [
            "router ospf network area Cisco configuration",
            "router ospf redistribute Cisco",
            "router ospf router-id Cisco",
        ],
        "static": [
            "ip route Cisco router configuration",
            "ip route static Cisco",
        ],
        "acl": [
            "access-list permit deny Cisco configuration",
            "ip access-list extended Cisco",
        ],
        "route_policy": [
            "route-map match set Cisco",
            "route-map permit Cisco configuration",
        ],
        "general": [
            "interface GigabitEthernet ip address Cisco",
            "hostname Cisco router configuration",
            "Cisco IOS configuration example",
        ],
    },
    "juniper": {
        "bgp": [
            "set protocols bgp group Juniper",
            "set protocols bgp neighbor Juniper configuration",
            "set routing-options autonomous-system Juniper",
        ],
        "ospf": [
            "set protocols ospf area interface Juniper",
            "set protocols ospf Juniper configuration",
        ],
        "static": [
            "set routing-options static route Juniper",
            "set routing-options static Juniper configuration",
        ],
        "acl": [
            "set security policies Juniper configuration",
            "set firewall family inet filter Juniper",
        ],
        "general": [
            "set interfaces ge- unit family inet address Juniper",
            "set system host-name Juniper",
            "Junos configuration example set protocols",
        ],
    },
}

# ============================================================
# 开源项目仓库 — 直接拉取已知的配置模板库
# ============================================================
OSS_CONFIG_REPOS = [
    # Ansible 网络模块中的配置模板
    "https://api.github.com/repos/ansible-collections/cisco.ios/contents/tests/integration",
    "https://api.github.com/repos/ansible-collections/junipernetworks.junos/contents/tests/integration",
    # NAPALM (网络自动化库)
    "https://api.github.com/search/code?q=router+config+repo:networktocode/ntc-templates+extension:txt",
    # 网络配置示例集合
    "https://api.github.com/search/code?q=router+bgp+ospf+config+extension:cfg+extension:txt+extension:conf",
    # Batfish 测试配置 (大量真实配置)
    "https://api.github.com/search/code?q=config+repo:batfish/batfish+extension:cfg",
    # 网络实验室/Packet Tracer 配置
    "https://api.github.com/search/code?q=router+config+PacketTracer+extension:txt",
    # CCIE/CCNP 实验配置 (大量完整配置)
    "https://api.github.com/search/code?q=CCIE+config+bgp+ospf+extension:txt+extension:cfg",
]


@dataclass
class ConfigPair:
    source: str           # cisco / juniper
    doc_type: str         # github / docs / forum / oss
    url: str
    nl_text: str
    config_text: str
    config_type: str      # bgp / ospf / static / acl / route_policy / general
    config_hash: str = "" # 去重用

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = hashlib.md5(self.config_text.encode()).hexdigest()[:12]


class GitHubMassCrawler:
    """GitHub 代码搜索引擎 — 大规模获取配置"""

    def __init__(self, github_token: str = None, max_pages: int = 30):
        self.session = requests.Session()
        self.max_pages = max_pages
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()

        # GitHub API 认证
        if github_token:
            self.session.headers.update({
                "Authorization": f"token {github_token}",
                "User-Agent": "PreConfig-Research-Bot/2.0",
                "Accept": "application/vnd.github.v3.text-match+json",
            })
        else:
            logger.warning("No GitHub token — rate limit = 60 req/hour (slow)")
            self.session.headers.update({
                "User-Agent": "PreConfig-Research-Bot/2.0",
                "Accept": "application/vnd.github.v3.text-match+json",
            })

    def search_code(self, query: str, vendor: str, config_type: str, max_pages: int = None) -> List[ConfigPair]:
        # """搜索 GitHub 代码并下载匹配文件（带重试机制）."""
        if max_pages is None:
            max_pages = self.max_pages
        pairs = []
        encoded_query = quote_plus(query)
    
        for page in range(1, max_pages + 1):
            url = f"https://api.github.com/search/code?q={encoded_query}&per_page=100&page={page}"
            
            # ---------- 重试循环 ----------
            retries = 3
            success = False
            for attempt in range(retries):
                try:
                    resp = self.session.get(url, timeout=60)  # 超时加长到 60 秒
                    # 临时性错误（408, 429, 5xx）都重试
                    if resp.status_code in [408, 429, 500, 502, 503, 504]:
                        wait = (attempt + 1) * 3  # 3, 6, 9 秒退避
                        logger.warning(f"GitHub {resp.status_code} at page {page}, retry {attempt+1}/{retries} after {wait}s")
                        time.sleep(wait)
                        continue
                    if resp.status_code == 403:
                        logger.warning(f"GitHub rate limit hit at page {page}. Stop.")
                        return pairs  # 速率限制不可恢复，直接返回已有数据
                    if resp.status_code != 200:
                        logger.warning(f"GitHub API error {resp.status_code} at page {page}, skipping this page")
                        break  # 其他错误（如404）跳过该页
                    
                    # ---- 正常处理响应 ----
                    data = resp.json()
                    items = data.get("items", [])
                    if not items:
                        logger.info(f"No items on page {page}, stopping query")
                        return pairs

                    logger.info(f"GitHub page {page}: {len(items)} files for '{query[:50]}...'")

                    for item in items:
                        raw_url = self._get_raw_url(item.get("html_url", ""))
                        if raw_url in self.seen_urls:
                            continue
                        self.seen_urls.add(raw_url)

                        configs = self._download_and_extract(raw_url, vendor)
                        for cfg in configs:
                            h = hashlib.md5(cfg.encode()).hexdigest()[:12]
                            if h not in self.seen_hashes:
                                self.seen_hashes.add(h)
                                pairs.append(ConfigPair(
                                    source=vendor,
                                    doc_type="github",
                                    url=raw_url,
                                    nl_text="",
                                    config_text=cfg,
                                    config_type=config_type,
                                    config_hash=h,
                                ))

                    time.sleep(2)  # 每页后休息
                    success = True
                    break  # 成功，跳出重试循环

                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout on page {page}, attempt {attempt+1}/{retries}")
                    time.sleep((attempt+1)*3)
                except Exception as e:
                    logger.error(f"Unexpected error page {page}: {e}")
                    break
            
            if not success:
                logger.warning(f"All retries failed for page {page}, skipping")
                # 继续下一页，而不是直接中断

        logger.info(f"GitHub [{vendor}/{config_type}]: {len(pairs)} unique configs")
        return pairs
    


    def _get_raw_url(self, html_url: str) -> str:
        """GitHub HTML URL → raw 文件 URL."""
        if "github.com" not in html_url:
            return html_url
        return html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    def _download_and_extract(self, raw_url: str, vendor: str) -> List[str]:
        """下载单个文件并提取配置块."""
        try:
            resp = self.session.get(raw_url, timeout=15)
            if resp.status_code != 200:
                return []
            content = resp.text
            return self._extract_configs(content, vendor)
        except:
            return []

    def _extract_configs(self, content: str, vendor: str) -> List[str]:
        """从文件内容中提取配置段落."""
        configs = []
        lines = content.split("\n")
        current = []
        in_config = False
        depth = 0

        if vendor == "cisco":
            # Cisco 配置特征
            triggers = {"interface", "router", "ip route", "access-list",
                       "route-map", "hostname", "!"}
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if in_config and len(current) >= 5:
                        configs.append("\n".join(current))
                    current = []
                    in_config = False
                    continue
                first_word = stripped.split()[0] if stripped.split() else ""
                if first_word.lower() in triggers or stripped.startswith("!"):
                    if not in_config and len(current) >= 5:
                        configs.append("\n".join(current))
                        current = []
                    in_config = True
                    current.append(stripped)
                elif in_config:
                    if any(kw in stripped.lower() for kw in triggers):
                        current.append(stripped)
                    else:
                        if len(current) >= 5:
                            configs.append("\n".join(current))
                        current = []
                        in_config = False
        else:
            # Juniper 配置特征
            triggers = {"set ", "interfaces {", "protocols {", "routing-options {",
                       "security {", "system {", "policy-options {", "firewall {"}
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if any(stripped.startswith(t) or stripped == t for t in triggers):
                    if len(current) >= 3:
                        configs.append("\n".join(current))
                    current = [stripped]
                    in_config = True
                    if "{" in stripped and "}" not in stripped:
                        depth = 1
                elif in_config:
                    current.append(stripped)
                    if "{" in stripped:
                        depth += 1
                    if "}" in stripped:
                        depth -= 1
                        if depth <= 0:
                            if len(current) >= 3:
                                configs.append("\n".join(current))
                            current = []
                            in_config = False
                            depth = 0

        # 尾部处理
        if current and in_config and len(current) >= 3:
            configs.append("\n".join(current))

        return configs

    def crawl_all(self, vendor: str) -> List[ConfigPair]:
        """爬取指定厂商的所有 GitHub 配置."""
        all_pairs = []
        queries = GITHUB_SEARCH_QUERIES.get(vendor, {})

        for config_type, query_list in queries.items():
            for query in query_list:
                pairs = self.search_code(query, vendor, config_type,
                                        max_pages=self.max_pages // len(query_list))
                all_pairs.extend(pairs)
                time.sleep(3)  # 每组查询之间休息

        return all_pairs


class OSSRepoCrawler:
    """从开源网络自动化仓库拉取配置模板."""

    def __init__(self, github_token: str = None):
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({"Authorization": f"token {github_token}"})
        self.session.headers.update({
            "User-Agent": "PreConfig-Research-Bot/2.0",
        })

    def crawl_repo_tree(self, repo_api_url: str) -> List[str]:
        """递归遍历 GitHub 仓库，找到所有配置文件."""
        config_urls = []
        try:
            resp = self.session.get(repo_api_url, timeout=30)
            if resp.status_code != 200:
                return []
            items = resp.json()
            for item in items:
                name = item.get("name", "")
                if item.get("type") == "file" and self._is_config_file(name):
                    config_urls.append(item.get("download_url", ""))
                elif item.get("type") == "dir":
                    time.sleep(1)
                    sub_urls = self.crawl_repo_tree(item.get("url", ""))
                    config_urls.extend(sub_urls)
        except:
            pass
            return config_urls

    def _is_config_file(self, filename: str) -> bool:
        ext = filename.lower()
        return any(ext.endswith(e) for e in [".cfg", ".conf", ".txt", ".rtr", ".ios", ".junos"])

    def crawl_all(self) -> List[ConfigPair]:
        """爬取所有已知 OSS 仓库的配置."""
        all_pairs = []
        seen = set()

        for repo_url in OSS_CONFIG_REPOS:
            if "search/code" in repo_url:
                # GitHub Code Search — 直接搜索
                try:
                    resp = self.session.get(repo_url, timeout=30)
                    if resp.status_code == 200:
                        items = resp.json().get("items", [])
                        for item in items:
                            raw_url = item.get("html_url", "").replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                            if raw_url in seen:
                                continue
                            seen.add(raw_url)
                            content_resp = self.session.get(raw_url, timeout=15)
                            if content_resp.status_code == 200:
                                vendor = "cisco" if "cisco" in raw_url.lower() else "juniper"
                                configs = self._extract_configs(content_resp.text, vendor)
                                for cfg in configs:
                                    all_pairs.append(ConfigPair(
                                        source=vendor,
                                        doc_type="oss",
                                        url=raw_url,
                                        nl_text="",
                                        config_text=cfg,
                                        config_type=self._detect_type(cfg),
                                    ))
                except:
                    continue
            else:
                # API tree URL — 遍历目录
                urls = self.crawl_repo_tree(repo_url)
                for url in urls:
                    if url in seen:
                        continue
                    seen.add(url)
                    try:
                        content_resp = self.session.get(url, timeout=15)
                        if content_resp.status_code == 200:
                            vendor = "cisco" if "cisco" in url.lower() else "juniper"
                            configs = self._extract_configs(content_resp.text, vendor)
                            for cfg in configs:
                                all_pairs.append(ConfigPair(
                                    source=vendor,
                                    doc_type="oss",
                                    url=url,
                                    nl_text="",
                                    config_text=cfg,
                                    config_type=self._detect_type(cfg),
                                ))
                    except:
                        continue
                time.sleep(2)

            logger.info(f"OSS repos: {len(all_pairs)} configs so far")

        return all_pairs

    def _extract_configs(self, content: str, vendor: str) -> List[str]:
        """复用 GitHubMassCrawler 的提取逻辑."""
        crawler = GitHubMassCrawler()
        return crawler._extract_configs(content, vendor)

    def _detect_type(self, text: str) -> str:
        t = text.lower()
        if "router bgp" in t or "bgp" in t:
            return "bgp"
        if "router ospf" in t or "ospf" in t:
            return "ospf"
        if "access-list" in t or "firewall" in t:
            return "acl"
        if "ip route" in t and "router" not in t:
            return "static"
        if "route-map" in t:
            return "route_policy"
        return "general"


def deduplicate(pairs: List[ConfigPair]) -> List[ConfigPair]:
    """按 config_hash 去重."""
    seen = {}
    for p in pairs:
        h = p.config_hash or hashlib.md5(p.config_text.encode()).hexdigest()[:12]
        p.config_hash = h
        if h not in seen or len(p.config_text) > len(seen[h].config_text):
            seen[h] = p
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Mass Config Crawler v3")
    parser.add_argument("--source", choices=["github", "docs", "oss", "all"], default="all")
    parser.add_argument("--max-pages", type=int, default=20,
                       help="Max GitHub search pages per query (default 20)")
    parser.add_argument("--token", type=str, default=None,
                       help="GitHub personal access token (5000 req/hour)")
    parser.add_argument("--vendor", choices=["cisco", "juniper", "all"], default="all")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs: List[ConfigPair] = []

    # ===== GitHub 大规模搜索 =====
    if args.source in ["github", "all"]:
        logger.info("=" * 60)
        logger.info("GITHUB MASS SEARCH — 主力数据源")
        logger.info("=" * 60)

        gh_token = args.token or os.environ.get("GITHUB_TOKEN")
        crawler = GitHubMassCrawler(github_token=gh_token, max_pages=args.max_pages)

        if args.vendor in ["cisco", "all"]:
            cisco_pairs = crawler.crawl_all("cisco")
            logger.info(f"Cisco GitHub: {len(cisco_pairs)} configs")
            all_pairs.extend(cisco_pairs)

        if args.vendor in ["juniper", "all"]:
            juniper_pairs = crawler.crawl_all("juniper")
            logger.info(f"Juniper GitHub: {len(juniper_pairs)} configs")
            all_pairs.extend(juniper_pairs)

    # ===== OSS 开源仓库 =====
    if args.source in ["oss", "all"]:
        logger.info("=" * 60)
        logger.info("OSS REPOS — 开源项目配置模板")
        logger.info("=" * 60)

        oss = OSSRepoCrawler(github_token=args.token or os.environ.get("GITHUB_TOKEN"))
        oss_pairs = oss.crawl_all()
        logger.info(f"OSS repos: {len(oss_pairs)} configs")
        all_pairs.extend(oss_pairs)

    # ===== 去重 =====
    logger.info("=" * 60)
    before = len(all_pairs)
    all_pairs = deduplicate(all_pairs)
    logger.info(f"Deduplication: {before} → {len(all_pairs)} ({before - len(all_pairs)} removed)")

    # ===== 保存 =====
    cisco = [p for p in all_pairs if p.source == "cisco"]
    juniper = [p for p in all_pairs if p.source == "juniper"]

    save_to_json([asdict(p) for p in cisco], RAW_DATA_DIR / "cisco" / "docs_results.json")
    save_to_json([asdict(p) for p in juniper], RAW_DATA_DIR / "juniper" / "docs_results.json")
    save_to_json([asdict(p) for p in all_pairs], RAW_DATA_DIR / "all_crawled.json")

    # ===== 统计 =====
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total unique configs: {len(all_pairs)}")
    print(f"  Cisco:   {len(cisco)}")
    print(f"  Juniper: {len(juniper)}")
    print(f"Types: {dict(Counter(p.config_type for p in all_pairs))}")
    print(f"Sources: {dict(Counter(p.doc_type for p in all_pairs))}")
    print(f"\nData saved to: {RAW_DATA_DIR / 'all_crawled.json'}")

    if len(all_pairs) < 500:
        logger.warning("⚠️  数据量较少 (< 500)，建议:")
        logger.warning("  1. 设置 GITHUB_TOKEN 提升 API limit")
        logger.warning("  2. 增大 --max-pages 参数")
        logger.warning("  3. 运行 python scripts/01b_generate_sample_data.py 补充")


def save_to_json(data: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} to {path}")


if __name__ == "__main__":
    main()