# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from eventkg.experiment_sheet import EXPLICIT, IMPLICIT, SEGMENTED, UNCLASSIFIED, edge_kind, sheet_for_method
from eventkg.link import verify_cross
from eventkg.snapshot import snapshot_sha256, write_event_snapshot


class OrderPrepTest(unittest.TestCase):
    def test_snapshot_sha_covers_events_and_relations_only(self):
        events = [{"id": "e1", "name": "发运增加"}]
        relations = [{"head": "e1", "tail": "e1", "type": "同时"}]
        with tempfile.TemporaryDirectory() as folder:
            info = write_event_snapshot(Path(folder) / "event_snapshot.json", events, relations,
                                        [{"id": "d1", "title": "周报", "file_name": "a.pdf", "report_date": ""}])
        self.assertEqual(info["snapshot_sha256"], snapshot_sha256(events, relations))
        self.assertEqual(len(info["snapshot_sha256"]), 64)

    def test_edge_kinds_follow_the_existing_path_label(self):
        self.assertEqual(edge_kind({"relation_class": "ASSERTED_CAUSE",
                                    "components": {"inference_path": "asserted"}}), EXPLICIT)
        self.assertEqual(edge_kind({"relation_class": "INFERRED_SUPPORTED",
                                    "components": {"inference_path": "explicit_third_document"}}), EXPLICIT)
        self.assertEqual(edge_kind({"relation_class": "ASSERTED_CAUSE",
                                    "components": {"inference_path": "implicit_segment", "segment": 1}}), SEGMENTED)
        self.assertEqual(edge_kind({"relation_class": "INFERRED_SUPPORTED",
                                    "components": {"inference_path": "implicit_multi_document"}}), IMPLICIT)
        self.assertEqual(edge_kind({"relation": "导致", "rule_evidence": "历史规律"}), UNCLASSIFIED)

    def test_recall_stays_empty_without_human_labels(self):
        sheet = sheet_for_method(
            "abductive", {"snapshot_sha256": "abc"}, ["t1", "t2"],
            [{"target": "t1", "candidates": ["c1"]}, {"target": "t2", "candidates": []}],
            [{"head": "c1", "tail": "t1", "relation": "导致", "relation_class": "ASSERTED_CAUSE",
              "components": {"inference_path": "asserted"}, "evidence_ids": ["ev:1"]}],
            [{"head": "c2", "tail": "t1", "relation": "导致",
              "components": {"inference_path": "implicit_segment"}}],
            [{"head": "c3", "tail": "t2", "relation": "导致", "rejection_reason": "没有证据"}],
        )
        self.assertIsNone(sheet["model_accuracy"])
        self.assertFalse(sheet["constructed_slice_f1_used"])
        self.assertIsNone(sheet["candidate_recall_at_k"]["recall"])
        self.assertEqual(sheet["target_coverage"]["targets"], 2)
        self.assertEqual(sheet["target_coverage"]["with_candidates"], 1)
        self.assertEqual(sheet["confirmed"][0]["kind"], EXPLICIT)
        self.assertEqual(sheet["uncertain"][0]["kind"], SEGMENTED)
        self.assertEqual(sheet["rejected"][0]["kind"], UNCLASSIFIED)
        self.assertEqual(sheet["confirmed"][0]["evidence"]["evidence_ids"], ["ev:1"])

    def test_legacy_records_rejected_without_changing_the_accept_rule(self):
        class Chat:
            def complete_json(self, system, user, max_tokens=4096, tag=""):
                return [{"id": "p1", "verdict": "成立", "reason": "原文支持"},
                        {"id": "p2", "verdict": "不成立", "reason": "只是同向"}]

        events = [{"id": "a", "name": "发运", "status": "已发生", "evidence": "发运下降", "doc_id": "d1"},
                  {"id": "b", "name": "价格", "status": "已发生", "evidence": "价格上涨", "doc_id": "d2"},
                  {"id": "c", "name": "库存", "status": "已发生", "evidence": "库存上升", "doc_id": "d2"}]
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        candidates = [
            {"head": "a", "tail": "b", "type": "导致", "gap_days": 7, "rule": {"evidence": "规律", "support": 2}},
            {"head": "a", "tail": "c", "type": "导致", "gap_days": 7, "rule": {"evidence": "规律", "support": 2}},
        ]
        recorded = []
        accepted = verify_cross(Chat(), candidates, events, documents, 1, verdicts_out=recorded)
        self.assertEqual([(item["head"], item["tail"]) for item in accepted], [("a", "b")])
        self.assertEqual([item["status"] for item in recorded], ["candidate", "rejected"])
        self.assertEqual(accepted[0]["relation_class"], "LEGACY_CANDIDATE")
        self.assertEqual(accepted[0]["verification_status"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
