# -*- coding: utf-8 -*-
"""Rule check on the ten real reports. These rates are not a Qwen accuracy."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventkg.corpus import load_corpus
from eventkg.graph_induction import (
    _clause_event_time,
    _relation_pairs,
    attach_mentions,
    report_series_id,
)


def placed(text, specs):
    mentions, cursor = [], 0
    for name, linked in specs:
        start = text.find(name, cursor)
        if start < 0:
            raise AssertionError("mention not in sentence: " + name)
        mentions.append({
            "mention_id": "%s@%d" % (linked, start),
            "char_start": start, "char_end": start + len(name),
            "linked_event_id": linked, "alignment_confidence": 0.9,
        })
        cursor = start + len(name)
    return mentions


def direct_pairs(text, specs):
    pairs = _relation_pairs(text, placed(text, specs))
    return sorted((pair["cause"]["linked_event_id"], pair["effect"]["linked_event_id"])
                  for pair in pairs if not pair["mood"] and pair.get("role", "direct") == "direct")


CASES = [
    {
        "id": "background_price",
        "date": "2026-05-22",
        "need": "随着近期成材价格大幅下跌，而部分钢厂开始切换至高价原料，成本推升，导致盈利率小幅回落",
        "mentions": [("价格", "价格"), ("成本", "成本"), ("盈利", "盈利")],
        "expect": [("成本", "盈利")],
    },
    {
        "id": "intermediate_purchase",
        "date": "2026-05-15",
        "need": "受区域内精粉价格走高影响，下游钢厂出于成本管控考量缩减采购力度，进而导致本地矿山企业库存持续堆积",
        "mentions": [("价格", "价格"), ("采购", "采购"), ("库存", "库存")],
        "expect": [("价格", "采购"), ("采购", "库存")],
    },
    {
        "id": "expectation_is_content",
        "date": "2026-05-22",
        "need": "成本预期松动叠加需求走弱、流动性收紧预期造成黑色价格持续走弱",
        "mentions": [("成本", "成本"), ("流动性", "流动性"), ("价格", "价格")],
        "expect": [("成本", "价格"), ("流动性", "价格")],
    },
    {
        "id": "coke_affects_profit",
        "date": "2026-05-29",
        "need": "受焦炭期现货上涨影响，钢厂盘面和现货利润同步走缩",
        "mentions": [("焦炭", "焦炭"), ("利润", "利润")],
        "expect": [("焦炭", "利润")],
    },
    {
        "id": "negation_blocks_its_clause",
        "date": "2026-06-05",
        "need": "原油价格的上涨已难以通过成本推升传导至铁矿",
        "mentions": [("原油", "原油"), ("成本", "成本"), ("铁矿", "铁矿")],
        "expect": [],
    },
    {
        "id": "maintenance_causes_output",
        "date": "2026-04-17",
        "need": "个别矿企开展短期检修，导致区域内精粉产量小幅下降回落",
        "mentions": [("检修", "检修"), ("产量", "产量")],
        "expect": [("检修", "产量")],
    },
    {
        "id": "two_causes_one_effect",
        "date": "2026-06-26",
        "need": "由于基本面弱势与宏观利空形成双重压制，黑色板块共振下行",
        "mentions": [("基本面", "基本面"), ("宏观", "宏观"), ("黑色", "黑色")],
        "expect": [("基本面", "黑色"), ("宏观", "黑色")],
    },
    {
        "id": "forecast_result_is_not_asserted",
        "date": "2026-06-05",
        "need": "由于钢厂利润收窄叠加矿山较高的折扣，部分市场参与者预期低品价格开始触底",
        "mentions": [("利润", "利润"), ("价格", "价格")],
        "expect": [],
    },
    {
        "id": "coke_squeezes_profit",
        "date": "2026-06-12",
        "need": "焦炭提涨继续挤压钢厂利润",
        "mentions": [("焦炭", "焦炭"), ("利润", "利润")],
        "expect": [("焦炭", "利润")],
    },
    {
        "id": "hurricane_then_arrival",
        "date": "2026-04-17",
        "need": "3月底澳矿发运受到飓风的影响，体现在了本期到港量中，导致到港量大幅下降",
        "mentions": [("发运", "发运"), ("飓风", "飓风"), ("到港", "到港")],
        "expect": [("飓风", "到港")],
    },
]


def event(event_id, doc, direction, evidence):
    return {"id": event_id, "doc_id": doc, "var": "铁矿石|发运量||当期", "dir": direction,
            "status": "已发生", "name": event_id, "evidence": evidence, "salience": "核心",
            "commodity": "铁矿石", "indicator": "发运量", "sections": ["s"], "page": 1}


def mention_cases(docs):
    by_date = {doc.get("report_date"): doc for doc in docs}
    iron = [doc for doc in docs if report_series_id(doc) == "mysteel:铁矿石:weekly"]
    rows = []
    samples = [
        ("2026-04-17", "2026-04-24", "下期发运增加"),
        ("2026-04-24", "", "下期发运增加"),
        ("2026-06-12", "", "下期发运增加"),
    ]
    catalog_events = []
    items = []
    for doc in iron:
        catalog_events.append(event(doc["id"] + ":ship", doc["id"], "上升", "发运增加"))
    for host_date, expected_date, text in samples:
        host = by_date[host_date]
        evidence_id = "ev:" + host["id"]
        items.append({"evidence_id": evidence_id, "document_id": host["id"], "text": text,
                      "report_date": host_date, "event_time": ""})
        rows.append((host_date, expected_date, evidence_id))
    pvc = next(doc for doc in docs if report_series_id(doc) == "longzhong:PVC:daily")
    catalog_events.append({
        "id": "pvc:stock", "doc_id": pvc["id"], "var": "PVC|库存||当期", "dir": "上升",
        "status": "已发生", "name": "pvc", "evidence": "库存增加", "commodity": "PVC",
        "indicator": "港口库存", "salience": "核心",
    })
    items.append({"evidence_id": "ev:pvc", "document_id": pvc["id"], "text": "上周PVC社会库存增加",
                  "report_date": pvc.get("report_date") or "", "event_time": ""})
    rows.append(("pvc", "", "ev:pvc"))
    attach_mentions(items, catalog_events, docs)
    judged = []
    for host_date, expected_date, evidence_id in rows:
        item = next(row for row in items if row["evidence_id"] == evidence_id)
        mention = item["mentions"][0]
        linked = mention["linked_event_id"]
        if expected_date:
            ok = linked.endswith(":ship") and any(
                doc["id"] + ":ship" == linked and doc.get("report_date") == expected_date for doc in docs)
        else:
            ok = linked == ""
        judged.append({
            "host": host_date, "expected": expected_date or "UNBOUND", "linked": linked or "UNBOUND",
            "event_time": mention["event_time"], "report_date": mention["report_date"],
            "series": mention["report_series_id"], "ok": ok and mention["event_time"] == "",
        })
    return judged


def main():
    docs = load_corpus(str(ROOT / "example" / "example_data" / "unified10"))
    narratives = {doc.get("report_date") or doc["id"]: doc.get("narrative") or "" for doc in docs}
    parse_hits = 0
    predicted, gold_hit = 0, 0
    print("SERIES")
    for doc in docs:
        print("%s\t%s\t%s" % (doc.get("report_date") or "NO_DATE", report_series_id(doc), doc["id"]))
    print("RELATIONS")
    for case in CASES:
        blob = narratives.get(case["date"], "")
        if case["need"] not in blob:
            raise SystemExit("sentence not in report %s: %s" % (case["date"], case["need"]))
        got = direct_pairs(case["need"], case["mentions"])
        expect = sorted(case["expect"])
        ok = got == expect
        parse_hits += int(ok)
        predicted += len(got)
        gold_hit += len([pair for pair in got if pair in expect])
        print(case["id"], "OK" if ok else "DIFF", "got", got, "expect", expect)
    precision = (gold_hit / predicted) if predicted else 0
    mentions = mention_cases(docs)
    mention_hits = sum(1 for row in mentions if row["ok"])
    print("MENTIONS")
    for row in mentions:
        print(row)
    dated = "Mysteel铁矿石分析团队3月31日预计，2026年4月份中国铁矿石市场供小于求，47港铁矿石库存去库"
    if dated not in narratives["2026-04-17"]:
        raise SystemExit("dated sentence missing from 2026-04-17")
    stated = _clause_event_time(dated)
    print("EVENT_TIME report_date=2026-04-17 stated=%r copied=%s" % (stated, stated == "2026-04-17"))
    if stated == "2026-04-17":
        return 1
    print("relation_parse_accuracy %.3f (%d/%d)" % (parse_hits / len(CASES), parse_hits, len(CASES)))
    print("mention_alignment_accuracy %.3f (%d/%d)" % (mention_hits / len(mentions), mention_hits, len(mentions)))
    print("relation_evidence_precision %.3f (%d/%d)" % (precision, gold_hit, predicted))
    print("note 规则核对，不是 Qwen 准确率")
    if parse_hits != len(CASES) or mention_hits != len(mentions) or gold_hit != predicted:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
