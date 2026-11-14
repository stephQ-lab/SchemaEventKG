# -*- coding: utf-8 -*-
"""v1.1.5: instance mentions, direction disputes, ISO dates, and intra-document evidence ids."""
import unittest

from eventkg.graph_induction import index_evidence
from eventkg.identity import period_key, same_instance
from eventkg.link import merge
from eventkg.paths import evidence_state, explain_paths
from eventkg.support import verify_relation
from eventkg.trends import interpret_trend, rewrite_market_label
from eventkg.validate import clean_event


def _event(event_id, indicator, direction, commodity="", scope="", subject="", when=""):
    return {
        "id": event_id, "indicator": indicator, "dir": direction, "commodity": commodity,
        "scope": scope, "subject": subject, "time": when, "event_time": when,
    }


def _check(head, tail, relation, texts):
    edge = {
        "head": head["id"], "tail": tail["id"], "relation": relation, "type": relation,
        "relation_class": "ASSERTED_CAUSE",
        "evidence_ids": ["e%d" % index for index in range(len(texts))],
    }
    catalog = {"e%d" % index: {"text": text} for index, text in enumerate(texts)}
    status = evidence_state(edge, catalog, {head["id"]: head, tail["id"]: tail})
    return status, edge


class InstanceEvidenceTest(unittest.TestCase):
    def test_another_commodity_is_not_the_same_mention(self):
        head = _event("E1", "港口库存", "下降", "PVC", "华东")
        tail = _event("E2", "钢厂利润", "下降", "铁矿石")
        status, _edge = _check(head, tail, "导致", ["铁矿石港口库存下降导致钢厂利润下降。"])
        self.assertEqual(status, "UNVERIFIED")

    def test_bare_stock_does_not_confirm_a_scoped_instance(self):
        head = _event("E1", "港口库存", "下降", "铁矿石", "47港")
        tail = _event("E2", "价格", "上升", "铁矿石")
        status, _edge = _check(head, tail, "导致", ["库存下降导致价格上升。"])
        self.assertEqual(status, "UNVERIFIED")

    def test_opposite_direction_is_not_support(self):
        head = _event("ship", "发运量", "下降", "铁矿石")
        tail = _event("arrive", "到港量", "下降", "铁矿石")
        status, _edge = _check(head, tail, "导致", ["发运量上升导致到港量上升。"])
        self.assertEqual(status, "UNVERIFIED")

    def test_effect_side_negation_blocks_confirmation(self):
        head = _event("ship", "发运量", "下降", "铁矿石")
        tail = _event("arrive", "到港量", "下降", "铁矿石")
        status, _edge = _check(head, tail, "导致", ["发运量下降导致到港量没有下降。"])
        self.assertNotEqual(status, "VERIFIED")

    def test_support_plus_same_period_negation_is_disputed(self):
        head = _event("ship", "发运量", "下降", "铁矿石")
        tail = _event("arrive", "到港量", "下降", "铁矿石")
        status, edge = _check(head, tail, "导致", [
            "发运量下降导致到港量下降。", "发运量下降并未导致到港量下降。"])
        self.assertEqual(status, "DISPUTED")
        self.assertIn("e1", edge["verification"]["counter_evidence_ids"])
        self.assertIn("发运量下降导致到港量下降", edge["verification"]["text"])

    def test_a_later_period_does_not_dispute_this_period(self):
        head = _event("ship", "发运量", "下降", "铁矿石", when="本周")
        tail = _event("arrive", "到港量", "下降", "铁矿石", when="本周")
        status, _edge = _check(head, tail, "导致", [
            "本周发运量下降导致到港量下降。", "下周发运量下降并未导致到港量下降。"])
        self.assertEqual(status, "VERIFIED")

    def test_comma_before_trigger_keeps_the_cause(self):
        head = _event("ship", "发运量", "下降", "铁矿石")
        tail = _event("arrive", "到港量", "下降", "铁矿石")
        status, _edge = _check(head, tail, "导致", ["发运量下降，导致到港量下降。"])
        self.assertEqual(status, "VERIFIED")

    def test_promote_is_not_the_same_as_cause(self):
        head = _event("ship", "发运量", "下降", "铁矿石")
        tail = _event("arrive", "到港量", "下降", "铁矿石")
        denied, _edge = _check(head, tail, "促进", ["发运量下降导致到港量下降。"])
        allowed, _edge = _check(head, tail, "促进", ["发运量下降促进到港量下降。"])
        self.assertEqual(denied, "UNVERIFIED")
        self.assertEqual(allowed, "VERIFIED")


class DateAndTrendTest(unittest.TestCase):
    def _clean(self, event_id, when, text):
        raw = {
            "id": event_id, "name": "%s本周47港铁矿石港口库存下降" % when,
            "commodity": "铁矿石", "indicator": "港口库存", "scope": "47港", "dim": "库存",
            "dir": "下降", "value": "", "change": "", "time": when, "period": "当期",
            "status": "已发生", "subject": "", "evidence": text, "salience": "一般",
        }
        item = clean_event(raw, text, [text], ["铁矿石"])
        item["id"] = item["local_id"]
        item.update(doc_id="d", evidences=[item["evidence"]], sections=["s"],
                    names=[item["name"]], members=[item["id"]],
                    var="铁矿石|港口库存|47港|当期")
        return item

    def test_two_iso_dates_do_not_merge(self):
        first = self._clean("E1", "2026-04-17", "2026-04-17本周47港铁矿石港口库存下降。")
        second = self._clean("E2", "2026-04-24", "2026-04-24本周47港铁矿石港口库存下降。")
        self.assertEqual(period_key(first), ("event_time", "2026-04-17"))
        self.assertEqual(period_key(second), ("event_time", "2026-04-24"))
        self.assertFalse(same_instance(first, second))
        _merged, _alias, stats = merge([first, second], [], [])
        self.assertEqual(stats["events_after"], 2)

    def test_iso_and_chinese_date_are_the_same_day(self):
        left = self._clean("E1", "2026-04-17", "2026-04-17本周47港铁矿石港口库存下降。")
        right = self._clean("E2", "2026年4月17日", "2026年4月17日本周47港铁矿石港口库存下降。")
        self.assertEqual(left["event_time"], "2026-04-17")
        self.assertEqual(right["event_time"], "2026-04-17")
        self.assertTrue(same_instance(left, right))

    def test_comparison_base_is_not_the_observation_period(self):
        reading = interpret_trend("本周库存较上周下降", indicator="库存", when="本周")
        self.assertEqual(reading["level_direction"], "下降")
        self.assertEqual(reading["comparison_basis"], "环比")
        self.assertFalse(reading["forecast"])

    def test_iron_concentrate_grade_stays(self):
        self.assertIn("65%铁精粉", rewrite_market_label("65%铁精粉价格上涨1.75美元/吨"))


class CatalogPathTest(unittest.TestCase):
    def test_intra_document_cause_enters_the_shared_catalog(self):
        ship = dict(_event("A", "发运量", "下降", "铁矿石"), doc_id="d", time="本周", salience="一般")
        arrive = dict(_event("B", "到港量", "下降", "铁矿石"), doc_id="d", time="本周", salience="一般")
        relation = {"head": "A", "tail": "B", "type": "导致", "evidence": "发运量下降导致到港量下降。", "basis": "原文明示"}
        items = index_evidence([ship, arrive], [{"id": "d"}], [relation])
        self.assertEqual(relation["evidence_ids"], ["ev:rel:A:B:0"])
        self.assertEqual(relation["relation_class"], "ASSERTED_CAUSE")
        catalog = {item["evidence_id"]: item for item in items}
        self.assertEqual(evidence_state(relation, catalog, {"A": ship, "B": arrive}), "VERIFIED")
        paths = explain_paths([ship, arrive], [relation], evidence_index=catalog)
        self.assertTrue(paths["PURE_CAUSAL"])
        self.assertTrue(paths["PURE_CAUSAL"][0]["confirmed"])

    def test_inferred_basis_is_not_promoted_to_an_asserted_cause(self):
        relation = {"head": "A", "tail": "B", "type": "导致", "evidence": "发运量下降导致到港量下降。", "basis": "综合推断"}
        index_evidence([], [], [relation])
        self.assertEqual(relation.get("evidence_ids"), ["ev:rel:A:B:0"])
        self.assertNotEqual(relation.get("relation_class"), "ASSERTED_CAUSE")

    def test_direct_text_without_a_saved_record_uses_the_catalog(self):
        edge = {"head": "A", "tail": "B", "type": "导致", "evidence": "发运量下降导致到港量下降。"}
        self.assertEqual(evidence_state(edge, {"ev:rel:A:B:0": {"text": edge["evidence"]}}), "MISSING")
        index_evidence([], [], [edge])
        record = verify_relation(edge, {"A": _event("A", "发运量", "下降"), "B": _event("B", "到港量", "下降")},
                                 {"ev:rel:A:B:0": {"text": edge["evidence"]}})
        self.assertEqual(record["relation_evidence_ids"], ["ev:rel:A:B:0"])
        self.assertEqual(record["verification_status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
