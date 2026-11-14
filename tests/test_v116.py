# -*- coding: utf-8 -*-
"""v1.1.6: one identity rule, one relation check, and word-level retrieval."""
import unittest

from atlas_rag.kg_construction.commodity import canonical_mention
from atlas_rag.kg_construction.event_ontology import normalize_event_relation
from eventkg.graph_induction import _mentions_for, retrieve_neighborhood
from eventkg.identity import can_merge_event_instances, period_key, same_instance
from eventkg.link import continuity, merge
from eventkg.paths import allows_causal_walk, evidence_state
from eventkg.retrieval import _ranks, _rrf, tokenize
from eventkg.support import verify_relation
from eventkg.trends import interpret_trend
from eventkg.validate import clean_event, normalize_dim, normalize_period, time_bucket


def _event(event_id, indicator, direction, **extra):
    item = {
        "id": event_id, "indicator": indicator, "dir": direction, "commodity": extra.pop("commodity", "铁矿石"),
        "scope": extra.pop("scope", ""), "subject": extra.pop("subject", ""), "time": extra.pop("time", ""),
        "event_time": extra.pop("event_time", ""), "doc_id": extra.pop("doc_id", "d"), "status": "已发生",
    }
    item.update(extra)
    return item


def _check(head, tail, relation, text):
    edge = {
        "head": head["id"], "tail": tail["id"], "relation": relation, "type": relation,
        "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["e1"],
    }
    return evidence_state(edge, {"e1": {"text": text}}, {head["id"]: head, tail["id"]: tail})


class IdentityGateTest(unittest.TestCase):
    def test_outline_cannot_bypass_a_rejected_instance(self):
        short = {"id": "a", "doc_id": "d", "commodity": "铁矿石", "indicator": "市场综述", "scope": "",
                 "dir": "下降", "status": "已发生", "time": "本周", "subject": "", "evidence": "库存下降"}
        long = dict(short, id="b", indicator="港口库存", scope="47港", evidence="47港港口库存下降")
        self.assertFalse(same_instance(short, long))
        self.assertFalse(can_merge_event_instances(short, long, "same_document"))

    def test_dates_only_in_the_mention_do_not_merge(self):
        def build(event_id, day):
            text = "%s47港铁矿石港口库存下降。" % day
            raw = {"id": event_id, "name": "本周47港铁矿石港口库存下降", "commodity": "铁矿石",
                   "indicator": "港口库存", "scope": "47港", "dim": "库存", "dir": "下降",
                   "value": "", "change": "", "time": "本周", "period": "当期", "status": "已发生",
                   "subject": "", "evidence": text, "salience": "一般"}
            item = clean_event(raw, text, [text], ["铁矿石"])
            item.update(id=event_id, doc_id="d", evidences=[text], sections=["s"], names=[item["name"]],
                        members=[event_id], var="铁矿石|港口库存|47港|当期")
            return item
        first, second = build("E1", "4月17日"), build("E2", "4月24日")
        self.assertNotEqual(period_key(first), period_key(second))
        self.assertFalse(same_instance(first, second))
        _merged, _alias, stats = merge([first, second], [], [])
        self.assertEqual(stats["events_after"], 2)

    def test_empty_time_is_unknown_and_a_day_is_not_a_month(self):
        self.assertEqual(time_bucket(""), "UNKNOWN")
        self.assertEqual(normalize_period("", "4月17日"), "当期")
        self.assertEqual(normalize_period("", "2026年4月17日"), "当期")
        self.assertEqual(normalize_dim("", "没有这个指标", "也没有"), "UNKNOWN")

    def test_grades_and_scopes_stay_apart(self):
        left = _event("a", "价格", "上升", scope="62%澳粉")
        right = _event("b", "价格", "上升", scope="65%精粉")
        left["name"], right["name"] = "62%澳粉价格上升", "65%精粉价格上升"
        self.assertFalse(same_instance(left, right))
        one = _event("c", "港口库存", "下降", scope="47港")
        other = _event("d", "港口库存", "下降", scope="45港")
        self.assertFalse(same_instance(one, other))


class RelationContractTest(unittest.TestCase):
    def test_another_stock_and_another_day_are_not_this_cause(self):
        head = _event("P", "港口库存", "下降", scope="47港")
        tail = _event("R", "价格", "上升")
        self.assertEqual(_check(head, tail, "导致", "47港社会库存下降导致价格上升。"), "UNVERIFIED")
        dated = _event("S", "发运量", "下降", event_time="2026-04-24", time="2026-04-24")
        arrive = _event("A", "到港量", "下降", event_time="2026-04-24", time="2026-04-24")
        self.assertEqual(_check(dated, arrive, "导致", "4月17日发运量下降导致到港量下降。"), "UNVERIFIED")
        self.assertEqual(_check(dated, arrive, "导致", "发运量下降导致到港量下降。"), "UNVERIFIED")

    def test_a_forecast_result_is_not_an_asserted_cause(self):
        head = _event("S", "发运量", "下降", time="本周")
        tail = _event("A", "到港量", "下降", time="下周")
        self.assertEqual(_check(head, tail, "导致", "发运下降导致预计下周到港下降。"), "FORECASTS")

    def test_support_and_denial_stay_disputed(self):
        head = _event("S", "发运量", "下降")
        tail = _event("A", "到港量", "下降")
        edge = {"head": "S", "tail": "A", "relation": "导致", "type": "导致",
                "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["e1", "e2"],
                "counter_evidence_ids": ["e2"]}
        catalog = {"e1": {"text": "发运量下降导致到港量下降。"}, "e2": {"text": "发运量下降并未导致到港量下降。"}}
        self.assertEqual(evidence_state(edge, catalog, {"S": head, "A": tail}), "DISPUTED")

    def test_discourse_words_and_cross_clauses_do_not_borrow_a_trigger(self):
        head = _event("S", "发运量", "下降")
        tail = _event("A", "到港量", "下降")
        self.assertNotEqual(_check(head, tail, "导致", "随着发运下降，到港下降。"), "VERIFIED")
        text = "发运下降导致到港下降，铁水上升推高矿价上涨。"
        self.assertIsNone(normalize_event_relation("发运下降", "导致", "矿价上涨", text))
        self.assertEqual(normalize_event_relation("发运下降", "导致", "到港下降", text), "导致")
        self.assertIsNone(normalize_event_relation("铁水上升", "导致", "矿价上涨", text))
        self.assertEqual(normalize_event_relation("铁水上升", "促进", "矿价上涨", text), "促进")
        self.assertIsNone(normalize_event_relation("发运下降", "促进", "到港下降", text))

    def test_necessity_is_not_an_ordinary_cause(self):
        self.assertFalse(allows_causal_walk({"type": "必要条件", "relation": "必要条件"}))
        self.assertFalse(allows_causal_walk({"head": "a", "tail": "b", "type": "促进", "relation": "必要条件"}))

    def test_a_missing_week_is_not_the_next_period(self):
        def ev(event_id, doc, day):
            return {"id": event_id, "doc_id": doc, "status": "已发生", "var": "铁矿石|港口库存|47港|当期",
                    "dir": "下降", "comparison_basis": "环比", "name": "港口库存下降", "evidence": "港口库存下降",
                    "change": "", "salience": "一般"}
        docs = [{"id": "d1", "report_date": "2026-04-17", "report_series_id": "series:iron"},
                {"id": "d2", "report_date": "2026-05-08", "report_series_id": "series:iron"}]
        links = continuity([ev("a", "d1", 17), ev("b", "d2", 8)], docs)
        self.assertEqual(links[0]["period_relation"], "NEXT_AVAILABLE_REPORT")


class TrendAndRetrievalTest(unittest.TestCase):
    def test_narrowing_rise_stays_a_rise(self):
        reading = interpret_trend("本周价格上涨，但涨幅收窄", indicator="价格", when="本周")
        self.assertEqual(reading["level_direction"], "上升")
        self.assertEqual(reading["momentum"], "收窄")
        self.assertNotEqual(canonical_mention("价格上涨但涨幅收窄"), "价格下降")
        cost = interpret_trend("成本预期松动造成价格下降", indicator="价格")
        self.assertFalse(cost["forecast"])

    def test_zero_score_does_not_enter_rrf_and_words_are_not_characters(self):
        ranks = _ranks([0.0, 0.2, 0.0])
        self.assertEqual(ranks[0], 0)
        self.assertEqual(ranks[1], 1)
        fused = _rrf([ranks, [0, 0, 0]])
        self.assertEqual(fused[0], 0.0)
        self.assertGreater(fused[1], 0.0)
        tokens = tokenize("澳洲发运下降导致港口库存下降")
        self.assertIn("港口库存", tokens)
        self.assertNotIn("港", tokens)

    def test_wrong_port_is_not_linked_and_old_events_stay_recallable(self):
        event = _event("E47", "港口库存", "下降", scope="47港", doc_id="d1")
        mentions = _mentions_for(
            {"evidence_id": "ev1", "document_id": "d1", "text": "本周45港港口库存下降。", "report_date": "2026-04-17"},
            [event], [{"id": "d1", "report_date": "2026-04-17"}])
        self.assertFalse(any(item.get("linked_event_id") == "E47" and item.get("alignment_confidence", 0) >= 0.75
                             for item in mentions))
        old = _event("old", "发运量", "下降", doc_id="d3", event_time="2026-01-01", var="铁矿石|发运量||当期")
        target = _event("now", "价格", "上升", doc_id="d2", event_time="2026-05-15", var="铁矿石|价格||当期")
        neighborhood = retrieve_neighborhood(target, [old, target], [{"id": "d2"}, {"id": "d3"}], [])
        self.assertIn("old", [item["id"] for item in neighborhood["candidates"]])


if __name__ == "__main__":
    unittest.main()
