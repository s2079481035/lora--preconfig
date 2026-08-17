"""
Step 32: Parameter-Accuracy Evaluation for Analysis Task
========================================================
Ana 任务的参考是退化模板句, ROUGE 等指标测的是"与模板的字符串匹配"。
本脚本直接评测分析的**参数正确性**: 从源配置提取参数集, 从预测提取
参数集, 计算 recall/precision/F1 — 绕开风格差异, 聚焦分析质量。

  venv/bin/python scripts/32_ana_param_accuracy.py
"""

import json, re, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOGS = PROJECT_ROOT / "logs"


def extract_params(text: str) -> set:
    """从配置/描述文本提取 (类型, 值) 参数集。"""
    params = set()
    text_l = text.lower()

    # BGP
    m = re.findall(r'router bgp (\d+)', text, re.I)
    m += re.findall(r'local-as (\d+)', text, re.I)
    m += re.findall(r'\bas (\d+)\b', text_l)
    for v in m:
        params.add(("bgp_as", v))
    # 邻居
    for ip, asn in re.findall(r'neighbor ([\d.]+) (?:remote-)?as (\d+)', text, re.I):
        params.add(("neighbor", ip))
        params.add(("neighbor_as", asn))
    for ip, asn in re.findall(r'peer with ([\d.]+).*?as (\d+)', text_l):
        params.add(("neighbor", ip))
        params.add(("neighbor_as", asn))
    # OSPF
    m = re.findall(r'router ospf (\d+)', text, re.I)
    m += re.findall(r'ospf process (\d+)', text_l)
    for v in m:
        params.add(("ospf_proc", v))
    for net in re.findall(r'network ([\d.]+) (?:0\.0\.0\.255|255\.255\.255\.0) area (\d+)', text, re.I):
        params.add(("ospf_net", net[0]))
        params.add(("ospf_area", net[1]))
    for net in re.findall(r'area (\d[\d.]*) \{', text):
        params.add(("ospf_area", net))
    # 静态路由
    for gw in re.findall(r'ip route [\d.]+ [\d.]+ ([\d.]+)', text, re.I):
        params.add(("static_gw", gw))
    for gw in re.findall(r'route 0\.0\.0\.0/0[\s\S]*?next-hop ([\d.]+)', text):
        params.add(("static_gw", gw))
    # ACL
    for v in re.findall(r'access-list (\d+)', text, re.I):
        params.add(("acl", v))
    # VLAN
    for v in re.findall(r'access vlan (\d+)', text, re.I):
        params.add(("vlan", v))
    for v in re.findall(r'vlan (\d+)', text, re.I):
        params.add(("vlan", v))
    # 接口
    for v in re.findall(r'interface (\S+)', text, re.I):
        params.add(("iface", v))
    for v in re.findall(r'ge-0/0/\d+|et-0/0/\d+|lo\d', text):
        params.add(("iface", v))
    # router-id
    for v in re.findall(r'router-id ([\d.]+)', text, re.I):
        params.add(("router_id", v))
    return params


def main():
    test = json.load(open(PROJECT_ROOT / "data" / "processed" / "test_data_multitask.json"))
    ana_inputs = {i: x["input"] for i, x in enumerate(test) if x["task"] == "config_analysis"}
    norag = json.load(open(LOGS / "rag_eval_v4_norag.json"))["analysis"]
    rag = json.load(open(LOGS / "rag_eval_v4_k3.json"))["analysis"]

    rows = []
    for a, b in zip(norag, rag):
        src = extract_params(ana_inputs.get(a["sample"], ""))
        pn = extract_params(a["prediction"])
        pr = extract_params(b["prediction"])

        def f1(pred):
            if not pred or not src:
                return 0.0, 0.0, 0.0, set()
            inter = pred & src
            rec = len(inter) / len(src)
            prec = len(inter) / len(pred)
            f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            return f, rec, prec, inter

        f1n, recn, precn, intern = f1(pn)
        f1r, recr, precr, interr = f1(pr)
        rows.append({"sample": a["sample"], "src_params": len(src),
                     "f1_norag": f1n, "rec_norag": recn, "prec_norag": precn,
                     "f1_rag": f1r, "rec_rag": recr, "prec_rag": precr,
                     "src": sorted(src), "hit_norag": sorted(intern), "hit_rag": sorted(interr)})

    n = len(rows)
    for key, label in [("f1_norag", "no-RAG F1"), ("f1_rag", "RAG F1"),
                       ("rec_norag", "no-RAG recall"), ("rec_rag", "RAG recall")]:
        vals = [r[key] for r in rows]
        print(f"{label:<14}: {sum(vals)/n:.4f}")

    win_rag = sum(1 for r in rows if r["f1_rag"] > r["f1_norag"])
    print(f"RAG F1 更高: {win_rag}/{n}")

    with open(LOGS / "ana_param_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logger.info("Saved → logs/ana_param_accuracy.json")


if __name__ == "__main__":
    main()