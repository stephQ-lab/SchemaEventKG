# -*- coding: utf-8 -*-
"""v1.1.3 regressions for the counterexamples that v1.1.2 still got wrong.

These checks are rule regressions. They are not Qwen precision, recall or F1.
"""
import json
import unittest

from atlas_rag.kg_construction.cypher_writer import props_map
from eventkg.export import instance_query_props
from eventkg.identity import same_instance
from eventkg.link import merge
from eventkg.paths import evidence_state
from eventkg.retrieve import Index, search
from eventkg.trends import interpret_trend, rewrite_market_label
from eventkg.validate import clean_event
from tests.test_v112 import event, snippet


def _clean(name, indicator, direction, evidence, status="已发生", when="本周", scope=""):
    source = evidence
    raw = {
        "id": "E1", "name": name, "commodity": "铁矿石", "indicator": indicator,
        "scope": scope, "dim": "价格", "dir": direction, "value": "", "change": "",
        "time": when, "period": "当期", "status": status, "subject": "",
        "evidence": evidence, "salience": "一般",
    }
    return clean_event(raw, source, [source], ["铁矿石"])


class TargetTrendTest(unittest.TestCase):
    def test_each_indicator_keeps_its_own_direction(self):
        sentence = "价格下跌，但库存上涨"
        self.assertEqual(interpret_trend(sentence)["level_direction"], "UNKNOWN")
        self.assertEqual(interpret_trend(sentence, indicator="价格")["level_direction"], "下降")
        self.assertEqual(interpret_trend(sentence, indicator="库存")["level_direction"], "上升")

        mixed = "钢厂利润下降，但铁矿石价格上涨"
        self.assertEqual(interpret_trend(mixed, indicator="钢厂利润", commodity="铁矿石")["level_direction"], "下降")
        self.assertEqual(interpret_trend(mixed, indicator="价格", commodity="铁矿石")["level_direction"], "上升")
        cleaned = _clean("本周钢厂利润下降", "钢厂利润", "下降", "本周钢厂利润下降，但铁矿石价格上涨。")
        self.assertEqual(cleaned["dir"], "下降")
        self.assertIn("下降", cleaned["name"])
        self.assertNotIn("上升", cleaned["name"])

    def test_negation_stays_on_the_target_verb(self):
        denied = interpret_trend("价格并未上涨", indicator="价格")
        self.assertNotEqual(denied["level_direction"], "上升")
        self.assertIn("上升", denied["negated"])
        kept = _clean("本期价格下降", "价格", "下降", "价格没有明显上升，仍然下降")
        self.assertEqual(kept["dir"], "下降")
        rejected = _clean("本期价格上升", "价格", "上升", "价格没有明显上升，仍然下降")
        self.assertEqual(rejected["dir"], "不明")

    def test_forecast_clause_is_not_marked_as_already_happened(self):
        cleaned = _clean("本期铁矿石到港量上升", "到港量", "上升", "展望下周到港量环比增加。", when="本期")
        self.assertEqual(cleaned["status"], "预测中")
        self.assertEqual(cleaned["time"], "下周")
        self.assertNotIn("本期", cleaned["name"])
        current = _clean("本周价格上升", "价格", "上升", "本周价格上涨，预计下周继续上涨。")
        self.assertEqual(current["status"], "已发生")
        self.assertEqual(current["dir"], "上升")

    def test_real_sentence_reaches_query_graph_and_answer(self):
        source = snippet("report_4cd24013444f9dda917e", "本周铁矿石价格持涨运行", before=0, after=12)
        self.assertIn("持涨", source)
        cleaned = _clean("本期铁矿石价格下降", "价格", "下降", source, when="本周")
        self.assertEqual(cleaned["dir"], "不明")
        agreed = _clean("本期铁矿石价格上升", "价格", "上升", source, when="本周")
        self.assertEqual(agreed["dir"], "上升")
        agreed.update({
            "id": "price", "doc_id": "report_4cd24013444f9dda917e", "names": [agreed["name"]],
            "sections": ["正文"], "members": ["price"], "var": "铁矿石|价格||当期",
            "abstract": "铁矿石|价格||当期|上升", "event_instance_id": "price",
        })
        props = instance_query_props(agreed, [{"id": agreed["doc_id"], "report_date": "2026-04-17"}])
        cypher = props_map(props)
        graph = {"events": [agreed], "documents": [{"id": agreed["doc_id"], "report_date": "2026-04-17"}],
                 "relations": [], "continuity": [], "expectations": [], "cross": [], "abstract_edges": []}
        payload = json.dumps({"nodes": [props]}, ensure_ascii=False)
        self.assertIn("上升", cypher)
        self.assertNotIn('"数值"', payload)
        self.assertNotIn("108.10", payload)
        self.assertNotIn("108.1", Index(graph).describe(agreed["id"]))
        found = search(graph, "铁矿石价格为什么上升")
        self.assertTrue(found["seeds"])


class IdentityConstraintTest(unittest.TestCase):
    def test_containment_does_not_merge_different_indicators(self):
        price = event("a", "d", "价格上涨", "价格", "", "本周", "上升")
        spot = event("b", "d", "现货价格上涨", "现货价格", "", "本周", "上升")
        port = event("c", "d", "47港价格上涨", "价格", "47港", "本周", "上升")
        self.assertFalse(same_instance(price, spot))
        self.assertFalse(same_instance(price, port))

    def test_one_observation_keeps_both_comparison_bases(self):
        mom = event("a", "d", "库存环比上涨", "港口库存", "47港", "本周", "上升")
        yoy = event("b", "d", "库存同比下降", "港口库存", "47港", "本周", "下降")
        mom["comparison_basis"] = "环比"
        yoy["comparison_basis"] = "同比"
        mom["trend_signals"] = [{"comparison_basis": "环比", "level_direction": "上升", "momentum": ""}]
        yoy["trend_signals"] = [{"comparison_basis": "同比", "level_direction": "下降", "momentum": ""}]
        self.assertTrue(same_instance(mom, yoy))
        merged, _, stats = merge([mom, yoy], [], [])
        self.assertEqual(stats["events_after"], 1)
        bases = {item["comparison_basis"] for item in merged[0]["trend_signals"]}
        self.assertEqual(bases, {"环比", "同比"})
        self.assertEqual(merged[0]["dir"], "上升")
        opposite = event("c", "d", "库存下降", "港口库存", "47港", "本周", "下降")
        self.assertFalse(same_instance(mom, opposite))


class TrendLabelTest(unittest.TestCase):
    def test_market_quantity_goes_and_the_object_stays(self):
        expected = {
            "澳洲发运下降75万吨": "澳洲发运下降",
            "本周港口库存下降75万吨": "本周港口库存下降",
            "矿价上涨1.75美元/吨": "矿价上涨",
            "62%澳粉价格上涨1.75美元/吨": "62%澳粉价格上涨",
            "47港库存下降75万吨": "47港库存下降",
            "2026-04-17铁矿石价格108.1美元/吨": "2026-04-17铁矿石价格",
            "上涨1.75美元/吨": "",
            "108.1美元/吨": "",
        }
        for raw, label in expected.items():
            self.assertEqual(rewrite_market_label(raw), label)


class EvidenceContractTest(unittest.TestCase):
    def test_order_negation_and_scope_block_verification(self):
        events = {
            "ship": {"id": "ship", "indicator": "发运量", "name": "发运下降"},
            "arrive": {"id": "arrive", "indicator": "到港量", "name": "到港下降"},
        }
        cases = {
            "发运下降导致到港下降": "VERIFIED",
            "到港下降导致发运下降": "UNVERIFIED",
            "发运下降没有导致到港下降": "UNVERIFIED",
            "发运下降不一定导致到港下降": "UNVERIFIED",
            "发运下降。员工培训导致其他事情。到港下降。": "UNVERIFIED",
        }
        for text, state in cases.items():
            edge = {"head": "ship", "tail": "arrive", "type": "导致", "relation": "导致",
                    "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["EV"], "evidence": "过期摘要"}
            catalog = {"EV": {"evidence_id": "EV", "text": text}}
            self.assertEqual(evidence_state(edge, catalog, events), state, text)

        graph = {
            "events": [
                dict(events["ship"], doc_id="d", dir="下降", status="已发生", salience="核心",
                     commodity="铁矿石", scope="", subject="", sections=["s"], event_instance_id="ship"),
                dict(events["arrive"], doc_id="d", dir="下降", status="已发生", salience="核心",
                     commodity="铁矿石", scope="", subject="", sections=["s"], event_instance_id="arrive"),
            ],
            "documents": [{"id": "d", "report_date": "2026-04-17"}],
            "relations": [{"head": "ship", "tail": "arrive", "type": "导致", "relation": "导致",
                           "relation_class": "ASSERTED_CAUSE", "evidence": "过期摘要",
                           "evidence_ids": ["EV"]}],
            "evidence": [{"evidence_id": "EV", "text": "发运下降导致到港下降"}],
            "continuity": [], "expectations": [], "cross": [], "abstract_edges": [],
        }
        found = search(graph, "到港量为什么下降")
        self.assertTrue(found["confirmed_causal"])
        shown = " ".join(item["relation"]["text"] for item in found["evidence"] if "relation" in item)
        self.assertIn("发运下降导致到港下降", shown)
        self.assertNotIn("过期摘要", shown)


if __name__ == "__main__":
    unittest.main()
