# -*- coding: utf-8 -*-
"""v1.1.7: a verified relation has to stay verified, and an unverified path cannot look confirmed."""
import unittest

from atlas_rag.kg_construction.event_ontology import normalize_event_relation
from eventkg.export import document_successors
from eventkg.extract import parse_document
from eventkg.graph_induction import _lookup_place
from eventkg.identity import can_merge_event_instances
from eventkg.link import _observation_supports, abstract_layer
from eventkg.paths import evidence_state, reasoning_chains
from eventkg.retrieve import Index
from eventkg.roles import event_kind
from eventkg.validate import clean_event


def _event(event_id, indicator, direction, **extra):
    item = {
        "id": event_id, "commodity": extra.pop("commodity", "铁矿石"), "indicator": indicator,
        "dir": direction, "level_direction": direction, "status": extra.pop("status", "已发生"),
        "scope": extra.pop("scope", ""), "subject": extra.pop("subject", ""),
        "time": extra.pop("time", "本周"), "event_time": extra.pop("event_time", ""),
        "doc_id": extra.pop("doc_id", "d1"), "name": indicator + direction,
        "var": "铁矿石|%s||当期" % indicator,
        "abstract": "铁矿石|%s|%s" % (indicator, direction),
    }
    item.update(extra)
    return item


class VerificationContractTest(unittest.TestCase):
    def test_changed_direction_invalidates_an_old_verified_record(self):
        head = _event("A", "发运量", "下降")
        tail = _event("B", "到港量", "下降")
        catalog = {"e1": {"text": "发运量下降导致到港量下降", "document_id": "d1"}}
        edge = {"head": "A", "tail": "B", "relation": "导致", "type": "导致",
                "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["e1"]}
        events = {"A": head, "B": tail}
        self.assertEqual(evidence_state(edge, catalog, events), "VERIFIED")
        tail["dir"] = "上升"
        tail["level_direction"] = "上升"
        self.assertNotEqual(evidence_state(edge, catalog, events), "VERIFIED")

    def test_counter_evidence_is_part_of_the_same_verification(self):
        head = _event("A", "发运量", "下降")
        tail = _event("B", "到港量", "下降")
        catalog = {
            "e1": {"text": "本周发运量下降导致到港量下降", "document_id": "d1"},
            "e2": {"text": "本周发运量下降并未导致到港量下降", "document_id": "d2"},
        }
        edge = {"head": "A", "tail": "B", "relation": "导致", "type": "导致",
                "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["e1"],
                "counter_evidence_ids": ["e2"]}
        self.assertEqual(evidence_state(edge, catalog, {"A": head, "B": tail}), "DISPUTED")

    def test_forecast_event_is_not_an_occurred_cause(self):
        head = _event("A", "发运量", "下降")
        tail = _event("B", "到港量", "下降", status="预测中")
        catalog = {"e1": {"text": "发运量下降导致到港量下降"}}
        edge = {"head": "A", "tail": "B", "relation": "导致", "type": "导致",
                "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["e1"]}
        self.assertEqual(evidence_state(edge, catalog, {"A": head, "B": tail}), "FORECASTS")


class ChainAndAbstractTest(unittest.TestCase):
    def test_a_path_without_evidence_is_only_a_candidate(self):
        events = [_event("A", "发运量", "下降", doc_id="d1"), _event("B", "到港量", "下降", doc_id="d1"),
                  _event("C", "港口库存", "下降", doc_id="d2"), _event("D", "价格", "下降", doc_id="d2")]
        for event in events:
            event["salience"] = "一般"
            event["sections"] = ["正文"]
        relations = [{"head": left, "tail": right, "type": "导致", "confidence": 0.9}
                     for left, right in (("A", "B"), ("B", "C"), ("C", "D"))]
        classified = {}
        confirmed, _local = reasoning_chains(events, relations, [], classified=classified)
        self.assertEqual(confirmed, [])
        self.assertTrue(classified["candidate"])
        self.assertTrue(all(item["chain_status"] == "candidate" for item in classified["candidate"]))

    def test_verified_steps_stay_verified_on_the_chain(self):
        events = [_event("A", "发运量", "下降", doc_id="d1", salience="核心", sections=["甲"]),
                  _event("B", "到港量", "下降", doc_id="d2", salience="核心", sections=["乙"]),
                  _event("C", "港口库存", "下降", doc_id="d3", salience="核心", sections=["丙"]),
                  _event("D", "价格", "上升", doc_id="d4", salience="核心", sections=["丁"])]
        catalog = {
            "ab": {"text": "发运量下降导致到港量下降"},
            "bc": {"text": "到港量下降导致港口库存下降"},
            "cd": {"text": "港口库存下降导致价格上升"},
        }
        relations = []
        for left, right, evidence_id in (("A", "B", "ab"), ("B", "C", "bc"), ("C", "D", "cd")):
            relations.append({"head": left, "tail": right, "type": "导致", "relation": "导致",
                              "relation_class": "ASSERTED_CAUSE", "confidence": 0.9,
                              "evidence_ids": [evidence_id]})
        confirmed, _local = reasoning_chains(events, relations, [], evidence_index=catalog)
        self.assertTrue(confirmed)
        for chain in confirmed:
            self.assertTrue(chain["evidence_status"])
            self.assertTrue(all(state == "VERIFIED" for state in chain["evidence_status"]))

    def test_abstract_layer_verifies_when_status_is_missing_and_counts_evidence_docs(self):
        ship = _event("A", "发运量", "下降", doc_id="d1", scope="澳洲")
        arrive = _event("B", "到港量", "下降", doc_id="d2", scope="47港")
        relation = {"head": "A", "tail": "B", "type": "导致", "relation": "导致", "confidence": 0.9,
                    "doc_id": "d2", "evidence_ids": ["e1"],
                    "evidence": "澳洲发运量下降导致47港到港量下降"}
        catalog = {"e1": {"evidence_id": "e1", "document_id": "d2",
                          "text": "澳洲发运量下降导致47港到港量下降"}}
        _abstracts, edges = abstract_layer([ship, arrive], [relation], [], {}, evidence_index=catalog)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["docs"], ["d2"])
        self.assertEqual(edges[0]["support"], 1)


class SharedRelationTypeTest(unittest.TestCase):
    def test_base_relation_type_follows_the_trigger(self):
        self.assertIsNone(normalize_event_relation("发运下降", "导致", "到港下降", "发运下降抑制到港下降"))
        self.assertIsNone(normalize_event_relation("发运下降", "促进", "到港下降", "发运下降导致到港下降"))
        self.assertEqual(normalize_event_relation("发运下降", "抑制", "到港下降", "发运下降抑制到港下降"), "抑制")
        self.assertEqual(normalize_event_relation("发运下降", "导致", "到港下降", "发运下降导致到港下降"), "导致")

    def test_document_synthesis_keeps_the_reverse_and_a_different_type(self):
        quote_back = "到港量下降导致发运量下降。"
        quote_help = "发运量下降促进到港量下降。"
        source = quote_back + quote_help
        events = [_event("A", "发运量", "下降"), _event("B", "到港量", "下降")]
        for event in events:
            event["status"] = "已发生"
            event["sections"] = ["正文"]
        existing = [{"head": "A", "tail": "B", "type": "导致", "evidence": "发运量下降导致到港量下降。"}]
        parsed = {"relations": [
            {"head": "B", "tail": "A", "type": "导致", "evidence": quote_back, "basis": "原文明示"},
            {"head": "A", "tail": "B", "type": "促进", "evidence": quote_help, "basis": "原文明示"},
        ]}
        doc = {"id": "d", "narrative": source}
        added, _groups, _lines, failed, _reasons = parse_document(
            parsed, doc, events, {"A": "A", "B": "B"}, existing)
        self.assertEqual(failed, 0)
        kept = {(item["head"], item["type"], item["tail"]) for item in added}
        self.assertIn(("B", "导致", "A"), kept)
        self.assertIn(("A", "促进", "B"), kept)


class BoundaryTest(unittest.TestCase):
    def test_macro_event_does_not_inherit_the_report_category(self):
        raw = {"name": "美国6月ISM服务业PMI指数下降", "commodity": "", "indicator": "ISM服务业PMI指数",
               "scope": "", "dir": "下降", "time": "6月", "status": "已发生", "subject": "",
               "evidence": "美国6月ISM服务业PMI指数下降", "dim": "宏观金融", "salience": "一般"}
        text = raw["evidence"]
        cleaned = clean_event(raw, text, [text], ["铁矿石"])
        self.assertEqual(cleaned["commodity"], "")

    def test_shared_comparison_basis_conflict_blocks_a_merge(self):
        left = _event("A", "价格", "上升", comparison_basis="环比；同比", trend_signals=[
            {"comparison_basis": "环比", "level_direction": "上升"},
            {"comparison_basis": "同比", "level_direction": "下降"},
        ])
        right = _event("B", "价格", "上升", comparison_basis="环比；同比", trend_signals=[
            {"comparison_basis": "环比", "level_direction": "上升"},
            {"comparison_basis": "同比", "level_direction": "上升"},
        ])
        self.assertFalse(can_merge_event_instances(left, right, "same_document"))

    def test_roles_separate_a_mill_decision_from_maintenance_and_policy(self):
        self.assertEqual(event_kind({"name": "钢厂因亏损主动停产"}), "PRODUCTION_CHANGE")
        self.assertEqual(event_kind({"name": "钢厂主动限产"}), "PRODUCTION_CHANGE")
        self.assertEqual(event_kind({"name": "装置计划检修"}), "MAINTENANCE")
        self.assertEqual(event_kind({"name": "政府发布环保限产政策"}), "POLICY_CHANGE")

    def test_last_period_observation_does_not_support_this_week(self):
        event = _event("A", "港口库存", "下降", time="本周")
        measure = {"status": "confirmed", "indicator_id": "港口库存", "row_header": "港口库存",
                   "scope": "", "measure_kind": "change", "level_direction": "下降",
                   "comparison_basis": "", "observation_period": "上期"}
        self.assertFalse(_observation_supports(event, measure, "2026-04-24"))

    def test_a_skipped_week_is_not_exported_as_the_next_period(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-24", "title": "Mysteel铁矿石周报", "file_name": "", "categories": ["铁矿石"]},
            {"id": "d2", "report_date": "2026-05-15", "title": "Mysteel铁矿石周报", "file_name": "", "categories": ["铁矿石"]},
        ]
        edges = document_successors(documents)
        self.assertEqual(edges, [("d1", "下一份已收录报告", "d2")])

    def test_repeated_quote_is_not_given_the_first_offset(self):
        sentence = "发运下降导致到港下降。"
        documents = [{"id": "d", "windows": [{"id": "w", "text": sentence + sentence, "char_start": 0}]}]
        place = _lookup_place(sentence, {"doc_id": "d"}, documents)
        self.assertEqual(place["place_status"], "AMBIGUOUS")
        self.assertIsNone(place["char_start"])

    def test_question_seed_uses_word_retrieval_for_ore_price(self):
        graph = {"events": [_event("P", "价格", "上升", doc_id="d", evidence="铁矿石价格上涨")],
                 "documents": [{"id": "d", "report_date": "2026-04-17"}],
                 "relations": [], "continuity": [], "expectations": [], "cross": []}
        found = Index(graph).seeds("为什么矿价上涨？")
        self.assertIn("P", found)


if __name__ == "__main__":
    unittest.main()
