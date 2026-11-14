# -*- coding: utf-8 -*-
"""v1.1.2 regression on the ten real reports. These rates are rule checks, not Qwen accuracy."""
import json
import unittest
from pathlib import Path

from atlas_rag.kg_construction.cypher_writer import props_map
from eventkg.export import instance_query_props
from eventkg.graph_induction import build_candidate_graph, merge_catalog
from eventkg.identity import coreferent, same_instance, stamp_instance
from eventkg.link import merge
from eventkg.observations import parse_table_measures
from eventkg.paths import evidence_state
from eventkg.retrieve import Index
from eventkg.trends import interpret_trend, rewrite_market_label, same_basis_conflict
from eventkg.validate import clean_event

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "example" / "example_data" / "unified10" / "structures"
RATES = {}


def snippet(report_id, marker, before=80, after=220):
    text = (STRUCT / ("%s.json" % report_id)).read_text(encoding="utf-8")
    at = text.find(marker)
    if at < 0:
        raise AssertionError("真实语料里没有：%s" % marker)
    return text[max(0, at - before):at + after]


def event(event_id, doc, name, indicator, scope, when, direction="下降"):
    return {
        "id": event_id, "doc_id": doc, "var": "铁矿石|%s|%s|当期" % (indicator, scope),
        "abstract": "铁矿石|%s|%s|当期|%s" % (indicator, scope, direction),
        "dir": direction, "status": "已发生", "name": name, "evidence": name,
        "evidences": [name], "sections": ["s"], "salience": "一般", "change": "", "value": "",
        "subject": "", "scope": scope, "commodity": "铁矿石", "indicator": indicator,
        "time": when, "period": "当期",
    }


class TrendRegressionTest(unittest.TestCase):
    def test_real_report_rows_and_sentences(self):
        trend_hits = comparison_hits = identity_hits = evidence_hits = trace_hits = 0
        trend_total = comparison_total = identity_total = evidence_total = trace_total = 0

        rising = snippet("report_4cd24013444f9dda917e", "本周铁矿石价格持涨运行")
        narrowed = snippet("report_250a9ad800956f8cc7d6", "降幅收窄")
        discount = snippet("report_4cd24013444f9dda917e", "成交折扣收窄", before=0, after=24)
        window = snippet("report_13bd61e477cb0e43850e", "同比去年减少0.29个百分点")
        start = window.find("高炉炼铁产能利用率")
        both_bases = window[start:window.find("；", start)]
        april = snippet("report_4cd24013444f9dda917e", "列2[2026/4/17]=108.10", before=120, after=260)
        pvc = snippet("report_b72db6c0f8db00abc45d", "指标（每周五更新）]=下游开工", before=20, after=240)
        grade_62 = snippet("report_4cd24013444f9dda917e", "Mysteel62%澳粉")
        grade_65 = snippet("report_4cd24013444f9dda917e", "65%球团")
        scope_47 = snippet("report_4cd24013444f9dda917e", "47港铁矿石库存环比去库")
        scope_45 = snippet("report_4cd24013444f9dda917e", "45港铁矿石港口库存")
        last_week = snippet("report_5a4db9e616a872cce176", "上周PVC社会库存")
        this_week = snippet("report_b72db6c0f8db00abc45d", "本周PVC")

        rise = interpret_trend(rising)
        narrow = interpret_trend(narrowed)
        price_narrow = interpret_trend("价格上涨但涨幅收窄")
        discount_trend = interpret_trend(discount)
        bare = interpret_trend("108.10")
        bases = interpret_trend(both_bases)
        trend_total += 6
        trend_hits += rise["level_direction"] == "上升"
        trend_hits += narrow["level_direction"] == "下降" and narrow["momentum"] == "收窄"
        trend_hits += price_narrow["level_direction"] == "上升" and price_narrow["momentum"] == "收窄"
        trend_hits += discount_trend["level_direction"] != "下降"
        trend_hits += bare["level_direction"] == "UNKNOWN"
        trend_hits += bases["level_direction"] == "上升"
        self.assertEqual(rise["level_direction"], "上升")
        self.assertNotEqual(narrow["level_direction"], "上升")
        self.assertEqual(price_narrow["level_direction"], "上升")
        self.assertNotEqual(discount_trend["level_direction"], "下降")
        self.assertEqual(bare["level_direction"], "UNKNOWN")

        comparison_total += 3
        comparison_hits += (not same_basis_conflict(bases["trend_signals"])
                            and {item["comparison_basis"] for item in bases["trend_signals"]} >= {"环比", "同比"})
        april_rows = parse_table_measures(april, document_id="report_4cd24013444f9dda917e", source_table_id="t-april")
        mom = next(item for item in april_rows if item["column_header"].startswith("环比"))
        yoy = [item for item in april_rows if "去年同期" in item["column_header"]]
        comparison_hits += mom["comparison_basis"] == "环比" and mom["direction"] == "上升" and mom["status"] == "confirmed"
        comparison_hits += all(item["status"] != "confirmed" for item in yoy)
        self.assertFalse(same_basis_conflict(bases["trend_signals"]))
        self.assertEqual(mom["comparison_basis"], "环比")
        self.assertEqual(mom["observation_period"], "2026-04-17")
        self.assertEqual(mom["comparison_period"], "2026-04-10")
        self.assertTrue(all(item["status"] != "confirmed" for item in yoy))
        level_only = april.split("；列4[")[0]
        lone = parse_table_measures(level_only, document_id="report_4cd24013444f9dda917e", source_table_id="t-level")
        self.assertTrue(lone)
        self.assertTrue(all(item.get("direction") != "上升" for item in lone))
        self.assertTrue(any(item["value_numeric"] == 108.10 for item in april_rows))

        pvc_rows = parse_table_measures(pvc, document_id="report_b72db6c0f8db00abc45d", source_table_id="t-pvc")
        change = next(item for item in pvc_rows if item["column_header"].startswith("涨跌"))
        self.assertEqual(change["direction"], "下降")
        self.assertEqual(change["change_unit"], "百分点")
        self.assertTrue(any(item["value_numeric"] == 40.22 for item in pvc_rows))
        self.assertTrue(any(item["value_numeric"] == 22.29 for item in pvc_rows))
        props = instance_query_props({
            "name": "本期铁矿石价格上升", "doc_id": "report_4cd24013444f9dda917e",
            "commodity": "62%澳粉", "indicator": "价格", "scope": "", "dim": "价格",
            "dir": "上升", "level_direction": "上升", "momentum": "", "comparison_basis": "环比",
            "time": "本周", "period": "当期", "status": "已发生", "subject": "",
            "names": [], "sections": [], "salience": "核心", "var": "v", "abstract": "a",
            "members": [], "id": "e1", "value": "108.10", "change": "1.75", "evidence": rising,
        })
        rendered = props_map(props)
        self.assertNotIn("108.10", rendered)
        self.assertNotIn("数值", rendered)
        self.assertNotIn("value_numeric", rendered)
        self.assertIn("62%", rendered)
        signal_props = props_map({
            "指标": change["indicator_id"], "趋势方向": change["direction"],
            "比较维度": change["comparison_basis"], "时期": change["observation_period"],
        })
        self.assertNotIn("40.22", signal_props)
        self.assertNotIn("22.29", signal_props)

        powder_62 = event("g62", "report_4cd24013444f9dda917e", "62%澳粉上涨", "价格指数", "", "本周", "上升")
        powder_65 = event("g65", "report_4cd24013444f9dda917e", "65%澳粉上涨", "价格指数", "", "本周", "上升")
        powder_62["evidence"] = grade_62
        powder_65["evidence"] = grade_65
        port_47 = event("p47", "report_4cd24013444f9dda917e", "47港库存下降", "港口库存", "47港", "本周")
        port_45 = event("p45", "report_4cd24013444f9dda917e", "45港库存下降", "港口库存", "45港", "本周")
        port_47["evidence"] = scope_47
        port_45["evidence"] = scope_45
        earlier = event("w1", "report_5a4db9e616a872cce176", "上周港口库存下降", "港口库存", "47港", "上周")
        later = event("w2", "report_5a4db9e616a872cce176", "本周港口库存下降", "港口库存", "47港", "本周")
        earlier["evidence"] = last_week
        later["evidence"] = this_week
        same_price = event("a", "d", "价格上升", "价格", "", "本周", "上升")
        other_price = event("b", "d", "价格上升", "价格", "", "本周", "上升")
        same_price["value"] = "108.10"
        other_price["value"] = "109.05"
        identity_total += 4
        identity_hits += not same_instance(powder_62, powder_65)
        identity_hits += not same_instance(port_47, port_45)
        identity_hits += not same_instance(earlier, later)
        identity_hits += same_instance(same_price, other_price)
        self.assertFalse(same_instance(powder_62, powder_65))
        self.assertFalse(same_instance(port_47, port_45))
        self.assertFalse(same_instance(earlier, later))
        self.assertTrue(same_instance(same_price, other_price))

        hot = event("hot", "d1", "日均铁水产量上升", "日均铁水产量", "", "本周", "上升")
        stock = event("stock", "d1", "港口库存下降", "港口库存", "47港", "本周")
        by_id = {item["id"]: item for item in (hot, stock)}
        missing = {"head": "hot", "tail": "stock", "type": "导致", "relation_class": "ASSERTED_CAUSE",
                   "evidence_ids": ["EV_DOES_NOT_EXIST"]}
        unrelated = {"head": "hot", "tail": "stock", "type": "导致", "relation_class": "ASSERTED_CAUSE",
                     "evidence_ids": ["EV-other"]}
        supported = {"head": "hot", "tail": "stock", "type": "导致", "relation_class": "ASSERTED_CAUSE",
                     "evidence_ids": ["EV-ok"]}
        catalog = {
            "EV-other": {"evidence_id": "EV-other", "text": "今天天气很好。"},
            "EV-ok": {"evidence_id": "EV-ok", "text": "日均铁水产量上升导致47港港口库存下降"},
        }
        evidence_total += 3
        evidence_hits += evidence_state(missing, catalog, by_id) == "MISSING"
        evidence_hits += evidence_state(unrelated, catalog, by_id) == "UNVERIFIED"
        evidence_hits += evidence_state(supported, catalog, by_id) == "VERIFIED"
        self.assertEqual(evidence_state(missing, catalog, by_id), "MISSING")
        self.assertEqual(evidence_state(unrelated, catalog, by_id), "UNVERIFIED")
        self.assertEqual(evidence_state(supported, catalog, by_id), "VERIFIED")

        left = event("m1", "report_4cd24013444f9dda917e", scope_47, "港口库存", "47港", "2026-04-17")
        right = event("m2", "report_e77ae7e0dc3eac41b92e", scope_45, "港口库存", "47港", "2026-04-24")
        left["event_time"] = "2026-04-17"
        right["event_time"] = "2026-04-24"
        left["evidence"] = scope_47
        right["evidence"] = scope_45
        self.assertFalse(coreferent(left, right))
        shared = dict(left)
        stamp_instance(shared, [left, right], "coreference")
        trace_total += 2
        trace_hits += {item["document_id"] for item in shared["mentions"]} == {
            "report_4cd24013444f9dda917e", "report_e77ae7e0dc3eac41b92e"}
        trace_hits += {item["source_text"] for item in shared["mentions"]} == {scope_47, scope_45}
        self.assertEqual({item["document_id"] for item in shared["mentions"]}, {
            left["doc_id"], right["doc_id"]})
        self.assertEqual({item["source_text"] for item in shared["mentions"]}, {scope_47, scope_45})

        cause = event("c", "d1", "发运下降导致到港下降", "发运量", "澳洲", "本周")
        effect = event("e", "d1", "到港下降", "到港量", "中国", "本周", "下降")
        cause["dir"] = "下降"
        relations = [
            {"head": "c", "tail": "e", "type": "导致", "confidence": 0.8, "evidence": "发运下降导致到港下降"},
            {"head": "c", "tail": "e", "type": "抑制", "confidence": 0.7, "evidence": "发运下降抑制到港下降"},
        ]
        kept_events, kept_relations, _ = merge([cause, effect], relations, [])
        kinds = sorted(item["type"] for item in kept_relations)
        self.assertEqual(kinds, ["导致", "抑制"])
        self.assertTrue(all(item.get("conflict_claims") for item in kept_relations))
        self.assertEqual(len(kept_events), 2)

        base = [{"evidence_id": "ev:old", "text": "旧证据"}]
        extra = [{"evidence_id": "ev:new", "text": narrowed}]
        catalog = merge_catalog(base, extra)
        graph = build_candidate_graph([{
            "target": "e",
            "hypotheses": [{"id": "h", "edge_scores": [{
                "head": "c", "tail": "e", "relation": "导致", "evidence_ids": ["ev:new"],
                "relation_class": "ASSERTED_CAUSE", "confirmed": True, "edge_score": 0.5,
            }]}],
        }], catalog)
        self.assertIn("降幅收窄", graph[0]["evidence_sources"][0]["text"])

        self.assertEqual(rewrite_market_label("108.1美元/吨"), "")
        self.assertEqual(rewrite_market_label("62%澳粉"), "62%澳粉")
        self.assertEqual(rewrite_market_label("47港"), "47港")
        self.assertEqual(rewrite_market_label("2026-04-17"), "2026-04-17")
        self.assertEqual(rewrite_market_label("上涨1.75美元/吨"), "")

        view = Index({
            "events": [dict(powder_62, evidence=rising)],
            "documents": [{"id": powder_62["doc_id"], "report_date": "2026-04-17"}],
            "relations": [], "continuity": [], "expectations": [], "cross": [],
        })
        self.assertNotIn("108.1", view.describe(powder_62["id"]))

        source = rising
        cleaned = clean_event({
            "id": "E1", "name": "本期铁矿石价格上升", "commodity": "铁矿石", "indicator": "价格",
            "scope": "", "dim": "价格", "dir": "下降", "value": "108.1美元/吨", "change": "上涨1.75美元/吨",
            "time": "本周", "period": "当期", "status": "已发生", "subject": "",
            "evidence": "本周铁矿石价格持涨运行", "salience": "核心",
        }, source, [source], ["铁矿石"])
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["dir"], "不明")
        self.assertEqual(cleaned["level_direction"], "UNKNOWN")

        RATES.update({
            "趋势抽取正确率": "%d/%d" % (trend_hits, trend_total),
            "比较维度正确率": "%d/%d" % (comparison_hits, comparison_total),
            "EventInstance错误合并率": "%d/%d" % (identity_total - identity_hits, identity_total),
            "关系证据有效率": "%d/%d" % (evidence_hits, evidence_total),
            "跨文档来源追溯正确率": "%d/%d" % (trace_hits, trace_total),
        })
        self.assertEqual(trend_hits, trend_total)
        self.assertEqual(comparison_hits, comparison_total)
        self.assertEqual(identity_hits, identity_total)
        self.assertEqual(evidence_hits, evidence_total)
        self.assertEqual(trace_hits, trace_total)

    @classmethod
    def tearDownClass(cls):
        if RATES:
            print("V112_RATES " + json.dumps(RATES, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
