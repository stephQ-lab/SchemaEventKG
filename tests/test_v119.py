# -*- coding: utf-8 -*-
"""v1.1.9: mention time, scope identity, evidence sources, retrieval and export."""
import os
import unittest

from eventkg.export import Builder, _confirmed_label, _mention_rows, _resolve_category, _trend_rows
from eventkg.gold import empty_gold_report
from eventkg.graph_induction import report_series_id
from eventkg.identity import can_merge_event_instances
from eventkg.link import continuity
from eventkg.llm import cache_key
from eventkg.ontology import raw_key
from eventkg.paths import allows_causal_walk
from eventkg.retrieval import event_signature, rank_events, tokenize
from eventkg.retrieve import Index
from eventkg.support import _evidence_rows, instance_fingerprint, verify_relation
from eventkg.validate import clean_event


def _event(event_id, indicator, direction, **extra):
    item = {
        "id": event_id, "commodity": extra.pop("commodity", "铁矿石"), "indicator": indicator,
        "dir": direction, "status": extra.pop("status", "已发生"),
        "scope": extra.pop("scope", ""), "subject": extra.pop("subject", ""),
        "time": extra.pop("time", "本周"), "event_time": extra.pop("event_time", ""),
        "doc_id": extra.pop("doc_id", "d1"), "name": extra.pop("name", indicator + direction),
        "var": extra.pop("var", "铁矿石|%s|%s|当期" % (indicator, extra.get("scope", ""))),
        "evidence": extra.pop("evidence", indicator + direction),
    }
    item.update(extra)
    return item


class MentionTimeTest(unittest.TestCase):
    def test_another_clause_does_not_lend_its_forecast(self):
        evidence = "钢厂利润下降，预计下周铁矿石价格上涨。"
        profit = clean_event(
            {"name": "钢厂利润下降", "indicator": "钢厂利润", "dir": "下降", "evidence": evidence},
            evidence, None, ["铁矿石"])
        price = clean_event(
            {"name": "铁矿石价格上涨", "indicator": "价格", "commodity": "铁矿石", "dir": "上升",
             "evidence": evidence},
            evidence, None, ["铁矿石"])
        self.assertEqual(profit["status"], "已发生")
        self.assertNotEqual(profit["time"], "下周")
        self.assertNotEqual(time_bucket_of(profit), "下期")
        self.assertEqual(price["status"], "预测中")
        self.assertEqual(price["time"], "下周")


def time_bucket_of(event):
    from eventkg.validate import time_bucket
    return time_bucket(event.get("time") or "")


class IdentityAndSeriesTest(unittest.TestCase):
    def test_scopes_subjects_and_dates_stay_distinct(self):
        week = _event("a", "港口库存", "下降", time="本周", scope="47港")
        recent = _event("b", "港口库存", "下降", time="近期", scope="47港")
        self.assertFalse(can_merge_event_instances(week, recent))
        dated = _event("c", "港口库存", "下降", time="2026-04-17", event_time="2026-04-17", scope="47港")
        later = _event("d", "港口库存", "下降", time="2026-04-24", event_time="2026-04-24", scope="47港")
        self.assertFalse(can_merge_event_instances(dated, later))
        named = _event("e", "发运量", "下降", subject="澳洲矿山", scope="澳洲")
        blank = _event("f", "发运量", "下降", subject="", scope="澳洲", evidence="另一句发运下降")
        self.assertFalse(can_merge_event_instances(named, blank))
        mills = {"commodity": "钢厂", "indicator": "盈利率", "scope": "247家钢厂", "period": "当期"}
        unknown = {"commodity": "钢厂", "indicator": "盈利率", "scope": "", "period": "当期"}
        other = {"commodity": "钢厂", "indicator": "盈利率", "scope": "250家钢厂", "period": "当期"}
        self.assertNotEqual(raw_key(mills), raw_key(unknown))
        self.assertNotEqual(raw_key(mills), raw_key(other))
        spot = _event("g", "现货价格", "上升", name="现货价格上升")
        future = _event("h", "期货价格", "上升", name="期货价格上升")
        self.assertFalse(can_merge_event_instances(spot, future))
        self.assertEqual(report_series_id({"title": "铁矿石周报"}), "")
        self.assertTrue(report_series_id({"title": "Mysteel铁矿石周报"}).startswith("mysteel:"))

    def test_another_subject_does_not_continue(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17", "report_series_id": "series:iron"},
            {"id": "d2", "report_date": "2026-04-24", "report_series_id": "series:iron"},
        ]
        events = [
            _event("a", "发运量", "下降", doc_id="d1", subject="澳洲矿山", var="铁矿石|发运量||当期"),
            _event("b", "发运量", "下降", doc_id="d2", subject="巴西矿山", var="铁矿石|发运量||当期"),
        ]
        self.assertEqual(continuity(events, documents), [])


class EvidenceRecordTest(unittest.TestCase):
    def test_another_document_without_a_date_does_not_bind(self):
        head = _event("A", "发运量", "下降", doc_id="d1")
        tail = _event("B", "到港量", "下降", doc_id="d1")
        catalog = {"e1": {"text": "发运下降导致到港下降。", "document_id": "d2", "report_date": "2026-04-24"}}
        verdict = verify_relation(
            {"head": "A", "tail": "B", "relation": "导致", "evidence_ids": ["e1"]},
            {"A": head, "B": tail}, catalog)
        self.assertNotEqual(verdict["verification_status"], "VERIFIED")
        self.assertEqual(verdict["edge_verification_status"], verdict["verification_status"])

    def test_mention_and_role_changes_invalidate_the_record(self):
        head = _event("A", "发运量", "下降")
        one = {"e1": {"text": "发运下降导致到港下降。", "document_id": "d1", "mentions": []}}
        two = {"e1": {"text": "发运下降导致到港下降。", "document_id": "d1",
                      "mentions": [{"linked_event_id": "A", "alignment_confidence": 0.9}]}}
        self.assertNotEqual(_evidence_rows(one, ["e1"])[0], _evidence_rows(two, ["e1"])[0])
        before = instance_fingerprint(head)
        head["roles"] = [{"role": "起点", "entity_id": "port-1", "value": "澳港"}]
        self.assertNotEqual(before, instance_fingerprint(head))

    def test_legacy_candidate_is_not_a_causal_walk(self):
        self.assertFalse(allows_causal_walk(
            {"type": "跨期传导", "relation": "导致", "relation_class": "LEGACY_CANDIDATE"}))
        self.assertFalse(allows_causal_walk({"type": "必要条件", "relation": "必要条件"}))


class RetrievalAndExportTest(unittest.TestCase):
    def test_signature_drops_prices_and_keeps_identity_numbers(self):
        event = _event("p", "价格", "上升", scope="47港", commodity="62%澳粉",
                       evidence="62%澳粉47港价格上涨1.75美元/吨至108.1美元/吨")
        signature = event_signature(event)
        self.assertIn("47港", signature)
        self.assertIn("62%", signature)
        self.assertNotIn("108.1", signature)
        tokens = tokenize("港口库存下降")
        self.assertTrue(tokens)
        self.assertFalse(any(len(token) == 1 for token in tokens))

    def test_hybrid_uses_rrf_and_kgqa_follows_the_configured_method(self):
        class Encoder:
            model_name = "fake-4"
            calls = 0

            def encode(self, texts):
                self.calls += 1
                vectors = []
                for text in texts:
                    hit = 1.0 if "价格" in (text or "") else 0.0
                    vectors.append([hit, 0.0, 0.0, 0.0])
                return vectors

        target = _event("T", "价格", "上升", evidence="铁矿石价格上升")
        near = _event("N", "价格", "上升", evidence="铁矿石价格上涨")
        far = _event("F", "检修", "中断", evidence="装置停车", dir="中断", commodity="焦炭",
                     status="预测中", time="", event_time="")
        ranked = rank_events(target, [target, near, far], method="hybrid", encoder=Encoder(), top_k=5)
        self.assertEqual(ranked["method"], "hybrid")
        self.assertEqual(len(Encoder().encode(["x"])[0]), 4)
        self.assertIn("N", [row["id"] for row in ranked["candidates"]])
        self.assertNotIn("F", [row["id"] for row in ranked["candidates"]])
        graph = {"retrieval_method": "hybrid", "events": [target, near], "documents": [],
                 "relations": [], "continuity": [], "expectations": [], "cross": []}
        index = Index(graph)
        with self.assertRaises(RuntimeError):
            index.seeds("为什么价格上涨")
        found = index.seeds("为什么价格上涨", encoder=Encoder())
        self.assertTrue(found)

    def test_endpoint_and_unconfirmed_export(self):
        left = cache_key("m", False, "s", "u", 8, endpoint="http://one")
        right = cache_key("m", False, "s", "u", 8, endpoint="http://two")
        self.assertNotEqual(left, right)
        self.assertEqual(_confirmed_label({"verification_status": "UNVERIFIED"}, "导致"), "候选关系")
        self.assertEqual(_confirmed_label({"relation_class": "LEGACY_CANDIDATE"}, "导致"), "候选关系")
        self.assertEqual(_confirmed_label({}, "导致"), "导致")
        builder = Builder()
        builder.edge("a", "导致", "b", {"证据编号": ["e1"]})
        builder.edge("a", "导致", "b", {"证据编号": ["e2"]})
        self.assertEqual(builder.edges[("a", "导致", "b")]["证据编号"], ["e1", "e2"])
        self.assertEqual(_resolve_category("宏观", ["铁矿石"]), "")
        rows = _trend_rows({"events": [_event("e", "价格", "上升", doc_id="d1")]})
        self.assertEqual(rows[0]["direction"], "上升")
        self.assertNotIn("108.1", str(rows))
        links = _mention_rows({"events": [_event("e", "价格", "上升")]})
        self.assertEqual(links[0]["alignment_status"], "UNIQUE")
        self.assertEqual(links[0]["event_instance_id"], "e")
        report = empty_gold_report()
        self.assertIsNone(report["relation"]["f1"])
        self.assertEqual(report["qwen"], "NOT_RUN")
        self.assertEqual(report["dense_hybrid"], "NOT_RUN")
        os.environ.pop("EVENTKG_EMBEDDING", None)


if __name__ == "__main__":
    unittest.main()
