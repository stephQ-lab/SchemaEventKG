# -*- coding: utf-8 -*-
import inspect
import unittest

from eventkg.evidence_index import index_corpus
from eventkg.graph_induction import induce, index_evidence, retrieve_neighborhood
from eventkg.identity import event_type_id
from eventkg.link import merge
from eventkg.retrieval import character_jaccard, compare_methods, event_signature, rank_events


def event(event_id, doc, var, direction, name, **extra):
    row = {"id": event_id, "doc_id": doc, "var": var, "abstract": var + "|" + direction, "dir": direction,
           "status": "已发生", "name": name, "evidence": extra.pop("evidence", name), "evidences": [name],
           "sections": ["s"], "salience": "核心", "change": "", "value": "", "subject": "", "scope": "",
           "commodity": "铁矿石", "indicator": "价格", "period": "当期", "time": ""}
    row["evidences"] = [row["evidence"]]
    row.update(extra)
    return row


class ScriptedEncoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.model_name = "scripted-test"
        self.model_version = "test"
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return [self.vectors.get(text, [0.0, 1.0]) for text in texts]


class IdentityRetrievalTest(unittest.TestCase):
    def test_different_weeks_stay_separate(self):
        last_week = event("d1:a", "d1", "铁矿石|港口库存|47港|当期", "下降", "上周库存下降", time="上周", scope="47港")
        this_week = event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "下降", "本周库存下降", time="本周", scope="47港")
        merged, _, stats = merge([last_week, this_week], [], [])
        self.assertEqual(stats["events_after"], 2)
        self.assertEqual(event_type_id(merged[0]), event_type_id(merged[1]))
        self.assertNotEqual(merged[0]["event_instance_id"], merged[1]["event_instance_id"])

    def test_cross_document_coreference_keeps_both_mentions(self):
        text = "2026年4月17日澳洲矿山发运下降。"
        fields = dict(event_time="2026-04-17", time="2026-04-17", scope="澳洲", subject="澳洲矿山",
                      indicator="发运量", evidence=text, commodity="铁矿石")
        first = event("d1:a", "d1", "铁矿石|发运量|澳洲|当期", "下降", "澳洲发运下降", **fields)
        second = event("d2:a", "d2", "铁矿石|发运量|澳洲|当期", "下降", "澳洲发运下降", **fields)
        merged, _, stats = merge([first, second], [], [])
        self.assertEqual(stats["events_after"], 1)
        self.assertEqual(stats["cross_document_links"], 1)
        self.assertEqual(sorted(merged[0]["source_document_ids"]), ["d1", "d2"])
        self.assertEqual(sorted(item["mention_id"] for item in merged[0]["mentions"]), ["men:d1:a", "men:d2:a"])
        self.assertTrue(all(item["char_start"] is None for item in merged[0]["mentions"]))

    def test_unknown_time_does_not_merge(self):
        first = event("d1:a", "d1", "铁矿石|港口库存|47港|当期", "下降", "库存下降", scope="47港")
        second = event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "下降", "港口库存下降", scope="47港")
        merged, _, stats = merge([first, second], [], [[first["id"], second["id"]]])
        self.assertEqual(stats["events_after"], 2)
        self.assertEqual(stats["llm_merges"], 0)

    def test_dense_recall_links_a_paraphrase_and_ignores_region(self):
        price = event("d1:price", "d1", "铁矿石|价格||当期", "上升", "铁矿石价格上涨",
                      evidence="铁矿石价格上涨。")
        wording = event("d2:word", "d2", "铁矿石|市场情绪||当期", "上升", "矿价走强",
                        indicator="市场情绪", evidence="矿价走强。")
        arrival = event("d2:port", "d2", "铁矿石|到港量|中国|当期", "下降", "中国到港下降",
                        indicator="到港量", scope="中国", evidence="中国港口到港下降。")
        ship = event("d1:ship", "d1", "铁矿石|发运量|澳洲|当期", "下降", "澳洲发运下降",
                     indicator="发运量", scope="澳洲", evidence="澳洲发运下降。")
        events = [price, wording, arrival, ship]
        vectors = {
            event_signature(price): [1.0, 0.0],
            event_signature(wording): [0.96, 0.05],
            event_signature(arrival): [0.2, 0.8],
            event_signature(ship): [0.1, 0.9],
        }
        encoder = ScriptedEncoder(vectors)
        neighborhood = retrieve_neighborhood(price, events, [{"id": "d1"}, {"id": "d2"}], [],
                                            retrieval_method="dense", encoder=encoder)
        wording_row = next(item for item in neighborhood["candidates"] if item["id"] == "d2:word")
        self.assertTrue(any(item["source"] == "语义检索" for item in wording_row["recall"]))
        self.assertIn("d2:port", {item["id"] for item in neighborhood["candidates"]})
        self.assertIn("d1:ship", {item["id"] for item in neighborhood["candidates"]})
        self.assertNotIn("超出范围", " ".join(neighborhood["stats"]["skip_reasons"]))

    def test_three_methods_return_candidates_and_jaccard_stays_offline(self):
        price = event("d1:price", "d1", "铁矿石|价格||当期", "上升", "铁矿石价格上涨")
        other = event("d2:word", "d2", "铁矿石|价格||当期", "上升", "铁矿石价格走强")
        vectors = {event_signature(price): [1.0, 0.0], event_signature(other): [0.9, 0.1]}
        encoder = ScriptedEncoder(vectors)
        events = [price, other]
        for method in ("bm25", "dense", "hybrid"):
            ranked = rank_events(price, events, method=method, encoder=encoder, top_k=10)
            self.assertEqual(ranked["candidates"][0]["id"], "d2:word")
            self.assertNotEqual(ranked["candidates"][0]["source"], "字符Jaccard")
        offline = rank_events(price, events, method="jaccard", top_k=10)
        self.assertEqual(offline["method"], "jaccard")
        self.assertGreater(character_jaccard(event_signature(price), event_signature(other)), 0)
        with self.assertRaises(RuntimeError):
            retrieve_neighborhood(price, events, [{"id": "d1"}, {"id": "d2"}], [], retrieval_method="jaccard")
        source = inspect.getsource(retrieve_neighborhood)
        self.assertNotIn("_similarity", source)
        self.assertNotIn("_grams", source)
        report = compare_methods(events, [{"head": "d2:word", "tail": "d1:price", "label": 1}], encoder, ks=(10,))
        self.assertIsNone(report["model_accuracy"])
        self.assertEqual(report["methods"]["dense"]["recall"]["10"], 1.0)
        self.assertIn("不能据此宣称", report["note"])

    def test_uncited_paragraph_is_retrievable_once(self):
        paragraph = "运输受阻发生在4月，与本期发运不是同一句话。"
        documents = [{
            "id": "d1", "narrative": paragraph,
            "windows": [{"id": "d1:w1", "sections": ["物流"], "text": paragraph + "\n\n" + paragraph}],
            "observations": [{"id": "d1:obs:1", "doc_id": "d1", "section": "表", "page": 3, "text": "表：到港量 100"}],
        }]
        passages = index_corpus(documents)
        copies = [item for item in passages if item["text"] == paragraph]
        self.assertEqual(len(copies), 1)
        self.assertEqual(len(copies[0]["sources"]), 2)
        self.assertTrue(copies[0]["counted"])
        found = next(item for item in passages if "运输受阻" in item["text"])
        self.assertEqual(found["paragraph_id"], "d1:w1")
        self.assertIsInstance(found["char_start"], int)
        table = next(item for item in passages if item["source_type"] == "observation")
        self.assertEqual(table["page"], 3)
        cited = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "发运下降", evidence="澳洲发运下降。")
        catalog = index_evidence([cited], documents, [{
            "head": "d1:ship", "tail": "d1:ship", "type": "导致", "evidence": "澳洲发运下降。",
        }])
        counted = [item for item in catalog if item.get("counted") and item["text"] == "澳洲发运下降。"]
        self.assertEqual(len(counted), 1)

    def test_needs_retrieval_runs_one_bounded_lookup(self):
        paragraph = "运输受阻发生在4月，需要单独核对时期。"
        documents = [{
            "id": "d1", "narrative": paragraph, "observations": [],
            "windows": [{"id": "d1:w1", "sections": ["物流"], "text": paragraph}],
        }, {"id": "d2", "narrative": "", "windows": [], "observations": []}]
        ship = event("d1:ship", "d1", "铁矿石|发运量|澳洲|当期", "下降", "澳洲发运下降",
                     indicator="发运量", scope="澳洲", evidence="澳洲发运下降。", salience="一般")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "铁矿石价格上涨",
                      evidence="铁矿石价格上涨。")

        class Chat:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, user, max_tokens=4096, tag=""):
                self.calls += 1
                if "运输受阻发生在4月" in user:
                    return {"hypotheses": [{"id": "h1", "kind": "证据不足", "needs_retrieval": False,
                                            "why": "已经看到补充段落", "missing": "", "edges": []}]}
                return {"hypotheses": [{"id": "h1", "kind": "证据不足", "needs_retrieval": True,
                                        "retrieval_query": "运输受阻发生在4月", "why": "缺少时期",
                                        "missing": "运输受阻发生在4月", "edges": []}]}

        chat = Chat()
        confirmed, payload = induce(
            chat, [ship, price], [], documents, workers=1, retrieval_method="bm25", retrieval_rounds=1)
        self.assertEqual(chat.calls, 2)
        self.assertTrue(any("运输受阻发生在4月" in (item.get("text") or "") for item in payload["evidence"]))


if __name__ == "__main__":
    unittest.main()
