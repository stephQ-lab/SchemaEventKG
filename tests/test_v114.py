# -*- coding: utf-8 -*-
"""v1.1.4 regressions for the counterexamples v1.1.3 still missed.

These checks are rule regressions. They are not Qwen precision, recall or F1.
"""
import unittest

from eventkg.identity import same_instance
from eventkg.link import attach_observations, merge
from eventkg.paths import evidence_state, explain_paths, reasoning_chains
from eventkg.trends import interpret_trend, rewrite_market_label
from eventkg.validate import clean_event
from tests.test_v112 import event


def _clean(name, indicator, direction, evidence, status="已发生", when="本周"):
    raw = {
        "id": "E1", "name": name, "commodity": "铁矿石", "indicator": indicator,
        "scope": "", "dim": "价格", "dir": direction, "value": "", "change": "",
        "time": when, "period": "当期", "status": status, "subject": "",
        "evidence": evidence, "salience": "一般",
    }
    return clean_event(raw, evidence, [evidence], ["铁矿石"])


class MentionTimeTest(unittest.TestCase):
    def test_this_week_and_next_week_stay_apart(self):
        text = "本周价格上涨，预计下周价格下跌。"
        current = interpret_trend(text, indicator="价格", when="本周")
        later = interpret_trend(text, indicator="价格", when="下周")
        self.assertEqual(current["level_direction"], "上升")
        self.assertFalse(current["forecast"])
        self.assertEqual([item["level_direction"] for item in current["trend_signals"]], ["上升"])
        self.assertEqual(later["level_direction"], "下降")
        self.assertTrue(later["forecast"])
        kept = _clean("本周价格上涨", "价格", "上升", text, when="本周")
        forecast = _clean("下周价格下跌", "价格", "下降", text, when="下周")
        self.assertEqual(kept["dir"], "上升")
        self.assertEqual(kept["status"], "已发生")
        self.assertNotIn("下降", kept["name"])
        self.assertEqual(forecast["dir"], "下降")
        self.assertEqual(forecast["status"], "预测中")
        self.assertIn("下降", forecast["name"])

    def test_enumeration_keeps_each_comparison_basis(self):
        trend = interpret_trend("库存环比上涨、同比下降")
        self.assertEqual(
            [(item["comparison_basis"], item["level_direction"]) for item in trend["trend_signals"]],
            [("环比", "上升"), ("同比", "下降")])
        denied = interpret_trend("价格没有明显上升，仍然下降", indicator="价格")
        self.assertEqual(denied["level_direction"], "下降")


class PercentRoleTest(unittest.TestCase):
    def test_change_rates_drop_and_grades_stay(self):
        self.assertEqual(rewrite_market_label("价格上涨1.65%"), "价格上涨")
        self.assertEqual(rewrite_market_label("开工率下降17.93%"), "开工率下降")
        self.assertEqual(rewrite_market_label("62%澳粉价格上涨1.75美元/吨"), "62%澳粉价格上涨")
        self.assertEqual(rewrite_market_label("65%精粉价格上涨1.75美元/吨"), "65%精粉价格上涨")
        cleaned = _clean("本周铁矿石价格上涨1.65%", "价格", "上升", "本周铁矿石价格上涨1.65%。")
        self.assertEqual(cleaned["commodity"], "铁矿石")
        self.assertNotIn("1.65%", cleaned["name"])
        self.assertEqual(cleaned["dir"], "上升")


class ArgumentScopeTest(unittest.TestCase):
    def _state(self, head, tail, relation, texts):
        events = {
            "ship": {"id": "ship", "indicator": "发运量"},
            "arrive": {"id": "arrive", "indicator": "到港量"},
            "price": {"id": "price", "indicator": "矿价"},
            "hot": {"id": "hot", "indicator": "铁水产量"},
        }
        ids = ["E%d" % index for index in range(len(texts))]
        edge = {"head": head, "tail": tail, "relation": relation, "type": relation,
                "relation_class": "ASSERTED_CAUSE", "evidence_ids": ids}
        catalog = {item: {"text": text} for item, text in zip(ids, texts)}
        return evidence_state(edge, catalog, events), edge

    def test_one_sentence_does_not_join_two_relations(self):
        text = "发运下降导致到港下降，铁水上升推高矿价上涨。"
        crossed, _ = self._state("ship", "price", "导致", [text])
        direct, _ = self._state("ship", "arrive", "导致", [text])
        other, _ = self._state("hot", "price", "促进", [text])
        self.assertEqual(crossed, "UNVERIFIED")
        self.assertEqual(direct, "VERIFIED")
        self.assertEqual(other, "VERIFIED")

    def test_modality_and_conflicting_sources_are_not_confirmed(self):
        self.assertEqual(self._state("ship", "arrive", "导致", ["预计发运下降将导致到港下降"])[0], "FORECASTS")
        self.assertEqual(self._state("ship", "arrive", "导致", ["发运下降可能导致到港下降"])[0], "UNVERIFIED")
        self.assertEqual(self._state("ship", "arrive", "促进", ["发运下降促进到港下降"])[0], "VERIFIED")
        self.assertEqual(self._state("ship", "arrive", "拖累", ["发运下降挤压到港下降"])[0], "VERIFIED")
        self.assertEqual(self._state("ship", "arrive", "抑制", ["发运下降挤压到港下降"])[0], "UNVERIFIED")
        self.assertEqual(self._state("ship", "arrive", "导致", ["发运下降未导致到港下降"])[0], "UNVERIFIED")
        self.assertEqual(self._state("ship", "arrive", "导致", ["由于发运下降，到港下降"])[0], "VERIFIED")
        self.assertEqual(self._state("ship", "arrive", "导致", ["受发运下降影响，到港下降"])[0], "VERIFIED")
        both = "发运下降导致到港下降，到港下降导致发运下降"
        state, edge = self._state("ship", "arrive", "导致", [both])
        self.assertEqual(state, "UNVERIFIED")
        sources, edge = self._state("ship", "arrive", "导致", ["发运下降导致到港下降。", "到港下降导致发运下降。"])
        self.assertEqual(sources, "DISPUTED")
        self.assertIn("发运下降导致到港下降", edge["verification"]["text"])
        self.assertNotIn("到港下降导致发运下降", edge["verification"]["text"])
        self.assertIn("E1", edge["verification"]["counter_evidence_ids"])
        reused = dict(edge, tail="price", relation="导致", type="导致")
        reused["verification"] = dict(edge["verification"], verification_status="VERIFIED")
        self.assertEqual(evidence_state(reused, {"E0": {"text": "发运下降导致到港下降。"},
                                                  "E1": {"text": "到港下降导致发运下降。"}},
                                       {"ship": {"id": "ship", "indicator": "发运量"},
                                        "arrive": {"id": "arrive", "indicator": "到港量"},
                                        "price": {"id": "price", "indicator": "矿价"}}), "UNVERIFIED")


class InstanceAndPathTest(unittest.TestCase):
    def test_a_narrower_comparison_signal_can_merge(self):
        wide = event("a", "d", "库存上升", "港口库存", "47港", "本周", "上升")
        narrow = event("b", "d", "库存下降", "港口库存", "47港", "本周", "下降")
        wide["trend_signals"] = [
            {"comparison_basis": "环比", "level_direction": "上升"},
            {"comparison_basis": "同比", "level_direction": "下降"},
        ]
        narrow["trend_signals"] = [{"comparison_basis": "同比", "level_direction": "下降"}]
        wide["comparison_basis"] = "环比；同比"
        narrow["comparison_basis"] = "同比"
        self.assertTrue(same_instance(wide, narrow))
        merged, _, stats = merge([wide, narrow], [], [])
        self.assertEqual(stats["events_after"], 1)
        opposite = event("c", "d", "库存下降", "港口库存", "47港", "本周", "下降")
        opposite["trend_signals"] = [{"comparison_basis": "环比", "level_direction": "下降"}]
        self.assertFalse(same_instance(wide, opposite))

    def test_current_period_links_to_the_report_observation_without_copying_the_date(self):
        text = "原表第1行：列1[指标]=到港量；列2[2026/4/17]=80；列3[2026/4/10]=100；列4[环比]=-20%"
        documents = [{"id": "doc", "report_date": "2026-04-17",
                      "observations": [{"id": "t1", "text": text, "page": 1, "section": "数据"}]}]
        current = {"id": "now", "doc_id": "doc", "indicator": "到港量", "scope": "", "dir": "下降",
                   "time": "本周", "event_time": "", "comparison_basis": "环比"}
        spot = {"id": "spot", "doc_id": "doc", "indicator": "现货价格", "scope": "", "dir": "下降",
                "time": "本周", "event_time": ""}
        _, links = attach_observations([current, spot], documents)
        self.assertEqual({item["head"] for item in links}, {"now"})
        self.assertEqual(current["event_time"], "")
        self.assertEqual(current["time"], "本周")
        self.assertEqual(links[0]["observation_period"], "2026-04-17")

    def test_a_missing_time_bridge_keeps_the_mixed_path_unconfirmed(self):
        def node(name, doc, indicator):
            return {"id": name, "doc_id": doc, "dir": "下降", "status": "已发生", "indicator": indicator,
                    "name": indicator + "下降", "salience": "核心", "sections": ["正文"], "event_instance_id": name}
        events = [node("A", "d1", "发运量"), node("B", "d1", "到港量"),
                  node("C", "d2", "铁水产量"), node("D", "d2", "港口库存")]
        catalog = {"AB": {"text": "发运下降导致到港下降"}, "CD": {"text": "铁水下降导致库存下降"},
                   "BC": {"text": "到港下降导致铁水下降"}}
        relations = [
            {"head": "A", "tail": "B", "type": "导致", "relation": "导致",
             "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["AB"]},
            {"head": "B", "tail": "C", "type": "导致", "relation": "导致",
             "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["BC"]},
            {"head": "C", "tail": "D", "type": "导致", "relation": "导致",
             "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["CD"]},
        ]
        bridge = [{"head": "B", "tail": "C", "type": "事态延续", "change": "延续"}]
        direct = [item for item in relations if item["head"] != "B"]
        mixed = [item for item in explain_paths(events, direct, bridge, catalog)["MIXED_EXPLANATION"]
                 if item["nodes"] == ("A", "B", "C", "D")]
        self.assertTrue(mixed)
        self.assertTrue(mixed[0]["causal_edges_verified"])
        self.assertFalse(mixed[0]["all_steps_verified"])
        self.assertFalse(mixed[0]["confirmed"])
        chains, _ = reasoning_chains(events, relations, [], evidence_index=catalog)
        classified = {}
        bare, _ = reasoning_chains(events, relations, [], classified=classified)
        self.assertTrue(chains)
        self.assertGreater(chains[0]["score"], 0)
        self.assertTrue(all(state == "VERIFIED" for state in chains[0]["evidence_status"]))
        self.assertEqual(bare, [])
        self.assertEqual(classified["candidate"][0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
