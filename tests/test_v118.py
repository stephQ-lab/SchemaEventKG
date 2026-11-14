# -*- coding: utf-8 -*-
"""v1.1.8: unknown time, instance evidence, one relation table, and report series."""
import unittest

from atlas_rag.kg_construction.event_ontology import normalize_event_relation
from eventkg.evidence_index import index_corpus
from eventkg.graph_induction import _relation_label, retrieve_neighborhood
from eventkg.identity import can_merge_event_instances
from eventkg.link import _observation_supports, continuity, expectation
from eventkg.support import verify_relation
from eventkg.validate import clean_relation, time_bucket


def _event(event_id, indicator, direction, **extra):
    item = {
        "id": event_id, "commodity": extra.pop("commodity", "铁矿石"), "indicator": indicator,
        "dir": direction, "level_direction": direction, "status": extra.pop("status", "已发生"),
        "scope": extra.pop("scope", ""), "subject": extra.pop("subject", ""),
        "time": extra.pop("time", "本周"), "event_time": extra.pop("event_time", ""),
        "doc_id": extra.pop("doc_id", "d1"), "name": extra.pop("name", indicator + direction),
        "var": extra.pop("var", "铁矿石|%s||当期" % indicator),
        "evidence": extra.pop("evidence", indicator + direction),
    }
    item.update(extra)
    return item


class TimeIdentityTest(unittest.TestCase):
    def test_unknown_wording_is_not_this_period(self):
        self.assertEqual(time_bucket(""), "UNKNOWN")
        self.assertEqual(time_bucket("本周"), "本期")
        self.assertEqual(time_bucket("上周"), "上期")
        self.assertEqual(time_bucket("下周"), "下期")
        self.assertEqual(time_bucket("近期"), "UNKNOWN")
        self.assertEqual(time_bucket("过去几周"), "UNKNOWN")
        self.assertEqual(time_bucket("下月"), "下月")
        self.assertEqual(time_bucket("上月"), "上月")
        self.assertEqual(time_bucket("本季度"), "本季度")
        self.assertEqual(time_bucket("年初以来"), "年初以来")
        self.assertEqual(time_bucket("近三年同期"), "同期")
        self.assertNotEqual(time_bucket("预计下月发运下降"), "本期")

    def test_recent_and_last_month_do_not_merge_into_this_week(self):
        current = _event("a", "港口库存", "下降", time="本周", scope="47港")
        recent = _event("b", "港口库存", "下降", time="近期", scope="47港")
        last_month = _event("c", "港口库存", "下降", time="上月", scope="47港")
        self.assertFalse(can_merge_event_instances(current, recent))
        self.assertFalse(can_merge_event_instances(current, last_month))


class InstanceEvidenceTest(unittest.TestCase):
    def test_a_directionless_mechanism_does_not_confirm_two_declines(self):
        head = _event("A", "发运量", "下降")
        tail = _event("B", "到港量", "下降")
        catalog = {"e1": {"text": "发运量导致到港量。", "document_id": "d1"}}
        verdict = verify_relation(
            {"head": "A", "tail": "B", "relation": "导致", "evidence_ids": ["e1"]},
            {"A": head, "B": tail}, catalog)
        self.assertNotEqual(verdict["verification_status"], "VERIFIED")

    def test_this_period_does_not_bind_an_earlier_date(self):
        head = _event("A", "发运量", "下降", time="本周", report_date="2026-04-24")
        tail = _event("B", "到港量", "下降", time="本周", report_date="2026-04-24")
        catalog = {"e1": {"text": "2026-04-17发运下降导致2026-04-17到港下降。", "document_id": "d1"}}
        verdict = verify_relation(
            {"head": "A", "tail": "B", "relation": "导致", "evidence_ids": ["e1"]},
            {"A": head, "B": tail}, catalog)
        self.assertNotEqual(verdict["verification_status"], "VERIFIED")

    def test_support_and_denial_are_judged_inside_one_period(self):
        head = _event("A", "发运量", "下降", time="2026-04-24", event_time="2026-04-24")
        tail = _event("B", "到港量", "下降", time="2026-04-24", event_time="2026-04-24")
        catalog = {
            "e1": {"text": "2026-04-17发运下降导致到港下降。", "document_id": "d1"},
            "e2": {"text": "2026-04-24发运下降导致到港下降。", "document_id": "d2"},
            "e3": {"text": "2026-04-24发运下降并未导致到港下降。", "document_id": "d2"},
        }
        verdict = verify_relation(
            {"head": "A", "tail": "B", "relation": "导致", "evidence_ids": ["e1", "e2"],
             "counter_evidence_ids": ["e3"]},
            {"A": head, "B": tail}, catalog)
        self.assertEqual(verdict["verification_status"], "DISPUTED")

    def test_social_stock_matches_social_stock_text(self):
        head = _event("A", "社会库存", "下降", commodity="")
        tail = _event("B", "利润", "下降", commodity="钢厂")
        catalog = {"e1": {"text": "社会库存下降导致利润下降。", "document_id": "d1"}}
        verdict = verify_relation(
            {"head": "A", "tail": "B", "relation": "导致", "evidence_ids": ["e1"]},
            {"A": head, "B": tail}, catalog)
        self.assertEqual(verdict["verification_status"], "VERIFIED")
        port = _event("C", "港口库存", "下降")
        rejected = verify_relation(
            {"head": "C", "tail": "B", "relation": "导致", "evidence_ids": ["e1"]},
            {"C": port, "B": tail}, catalog)
        self.assertNotEqual(rejected["verification_status"], "VERIFIED")


class RelationRegistryTest(unittest.TestCase):
    def test_drag_and_push_keep_their_relation(self):
        quote = "焦炭上涨拖累钢厂利润下降"
        events = {
            "A": {"name": "焦炭上涨", "indicator": "价格", "commodity": "焦炭", "evidence": "焦炭上涨",
                  "status": "已发生", "scope": "", "subject": ""},
            "B": {"name": "钢厂利润下降", "indicator": "利润", "commodity": "", "evidence": "钢厂利润下降",
                  "status": "已发生", "scope": "", "subject": ""},
        }
        kept = clean_relation({"head": "A", "tail": "B", "type": "拖累", "evidence": quote}, events, quote, None)
        self.assertIsNotNone(kept)
        self.assertEqual(kept["type"], "拖累")
        self.assertEqual(normalize_event_relation("焦炭上涨", "拖累", "钢厂利润下降", quote), "拖累")
        self.assertEqual(_relation_label({"polarity": "拖累", "trigger": "拖累"}), "拖累")
        pushed = "铁水上升推高矿价"
        self.assertEqual(normalize_event_relation("铁水上升", "促进", "矿价", pushed), "促进")
        self.assertEqual(_relation_label({"polarity": "促进", "trigger": "推高"}), "促进")
        self.assertNotEqual(_relation_label({"polarity": "拖累", "trigger": "拖累"}), "导致")
        self.assertNotEqual(_relation_label({"polarity": "促进", "trigger": "推高"}), "导致")


class SeriesAndObservationTest(unittest.TestCase):
    def _doc(self, doc_id, day, series):
        return {"id": doc_id, "report_date": "2026-04-%02d" % day, "report_series_id": series}

    def test_another_series_does_not_continue_or_confirm_a_forecast(self):
        documents = [self._doc("d1", 17, "series:A"), self._doc("d2", 24, "series:B")]
        happened = [
            _event("d1:a", "发运量", "下降", doc_id="d1", var="铁矿石|发运量||当期"),
            _event("d2:a", "发运量", "下降", doc_id="d2", var="铁矿石|发运量||当期"),
        ]
        self.assertEqual(continuity(happened, documents), [])
        forecast = _event("d1:f", "发运量", "下降", doc_id="d1", status="预测中", time="下期",
                          var="铁矿石|发运量||当期")
        self.assertEqual(expectation(happened + [forecast], documents), [])

    def test_a_table_without_a_period_does_not_bind_this_week(self):
        event = _event("now", "港口库存", "下降", scope="47港", time="本周", comparison_basis="环比")
        measure = {
            "status": "confirmed", "indicator_id": "港口库存", "row_header": "港口库存",
            "scope": "47港", "direction": "下降", "measure_kind": "change",
            "comparison_basis": "环比", "observation_period": "UNKNOWN", "commodity_id": "铁矿石",
        }
        self.assertFalse(_observation_supports(event, measure, "2026-04-24"))
        self.assertEqual(measure["observation_link_status"], "AMBIGUOUS")


class RecallAndSourceTest(unittest.TestCase):
    def test_same_sentence_keeps_each_report(self):
        text = "发运下降导致到港下降。"
        documents = [
            {"id": "d1", "report_date": "2026-04-17", "windows": [{"id": "w1", "text": text}]},
            {"id": "d2", "report_date": "2026-04-24", "windows": [{"id": "w2", "text": text}]},
        ]
        items = index_corpus(documents)
        self.assertEqual(len(items), 1)
        sources = items[0]["sources"]
        self.assertEqual({item["document_id"] for item in sources}, {"d1", "d2"})
        self.assertEqual({item["report_date"] for item in sources}, {"2026-04-17", "2026-04-24"})
        self.assertEqual(len({item["evidence_id"] for item in sources}), 2)

    def test_a_local_relation_is_not_dropped_for_a_weak_lexical_hit(self):
        target = _event("T", "价格", "上升", doc_id="d1", evidence="港口库存下降", var="铁矿石|价格||当期")
        local = _event("L", "检修", "中断", doc_id="d2", evidence="装置停车检修完毕",
                       var="铁矿石|装置检修||当期", dir="中断")
        noise = []
        for index in range(4):
            noise.append(_event(
                "N%d" % index, "价格", "上升", doc_id="d3",
                evidence="港口库存下降并且价格上涨", var="焦炭|价格||当期", commodity="焦炭"))
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-04-17"},
            {"id": "d3", "report_date": "2026-04-17"},
        ]
        relations = [{"head": "L", "tail": "T", "type": "导致"}]
        found = retrieve_neighborhood(
            target, [target, local] + noise, documents, relations, top_k=2, retrieval_method="bm25")
        self.assertIn("L", [item["id"] for item in found["candidates"]])


if __name__ == "__main__":
    unittest.main()
