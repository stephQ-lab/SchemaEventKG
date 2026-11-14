# -*- coding: utf-8 -*-
import json
import shutil
import unittest

from atlas_rag.document_input.structure import contents_pages, make_chunks
from atlas_rag.kg_construction.commodity import canonical_mention, collapse_mentions
from atlas_rag.kg_construction.document_linking import collapse_document_labels, contained_duplicate
from atlas_rag.kg_construction.event_ontology import normalize_event_relation
from eventkg.link import align_polarity, continuity, expectation, merge
from eventkg.ontology import _accept, raw_key
from eventkg.paths import reasoning_chains
from eventkg.pipeline import run
from eventkg.validate import clean_event, clean_relation, repair_quote
from run_unified import ROOT
from tests.fake_llm import FakeChat

DATA = ROOT / "example" / "example_data" / "unified10"
SOURCE = "本期47港铁矿石库存环比去库75万吨。受高炉检修影响，日均铁水产量下降2.1万吨。"


def event(event_id, doc, var, direction, status="已发生", name=None):
    return {"id": event_id, "doc_id": doc, "var": var, "abstract": var + "|" + direction, "dir": direction,
            "status": status, "name": name or event_id + "事件描述", "evidence": "证据" + event_id,
            "evidences": ["证据" + event_id], "sections": ["s"], "salience": "一般", "change": "", "value": "",
            "subject": "", "scope": "", "commodity": "铁矿石", "indicator": "港口库存"}


class ValidateTest(unittest.TestCase):
    def test_quote_is_repaired_to_the_source_sentence(self):
        self.assertEqual(repair_quote("47港铁矿石库存环比去库75万吨", SOURCE), "47港铁矿石库存环比去库75万吨")
        self.assertEqual(repair_quote("本期47港铁矿库存环比去库75万吨。", SOURCE), "本期47港铁矿石库存环比去库75万吨。")
        self.assertEqual(repair_quote("美联储宣布降息", SOURCE), "")

    def test_event_numbers_must_be_in_the_evidence(self):
        raw = {"id": "E1", "name": "47港铁矿石库存环比去库80万吨", "commodity": "铁矿石", "indicator": "港口库存",
               "scope": "47港", "dir": "去库", "evidence": "本期47港铁矿石库存环比去库75万吨。"}
        self.assertIsNone(clean_event(raw, SOURCE, None, ["铁矿石"]))
        raw["name"] = "47港铁矿石库存环比去库75万吨"
        cleaned = clean_event(raw, SOURCE, None, ["铁矿石"])
        self.assertEqual(cleaned["dir"], "下降")
        self.assertEqual(cleaned["dim"], "库存")
        self.assertEqual(cleaned["scope"], "47港")
        self.assertEqual(cleaned["name"], "本期47港铁矿石港口库存下降")
        self.assertNotIn("75", cleaned["name"])

    def test_causal_relation_needs_a_trigger_and_both_ends(self):
        events = {
            "a": {"name": "高炉检修", "indicator": "高炉检修", "scope": "", "subject": "", "commodity": "铁水",
                  "evidence": "受高炉检修影响"},
            "b": {"name": "日均铁水产量下降2.1万吨", "indicator": "日均铁水产量", "scope": "", "subject": "",
                  "commodity": "铁水", "evidence": "日均铁水产量下降2.1万吨"},
        }
        kept = clean_relation({"head": "a", "tail": "b", "type": "导致",
                               "evidence": "受高炉检修影响，日均铁水产量下降2.1万吨。"}, events, SOURCE, None)
        self.assertEqual(kept["basis"], "原文明示")
        dropped = clean_relation({"head": "a", "tail": "b", "type": "导致",
                                  "evidence": "本期47港铁矿石库存环比去库75万吨。"}, events, SOURCE, None)
        self.assertIsNone(dropped)


class OntologyTest(unittest.TestCase):
    def test_scope_and_commodity_are_never_merged_away(self):
        raw = "铁矿石|港口库存|47港|当期"
        self.assertIsNone(_accept(raw, {"key": "铁矿石|港口库存|45港|当期"}, "库存"))
        self.assertIsNone(_accept(raw, {"key": "PB粉|港口库存|47港|当期"}, "库存"))
        self.assertIsNone(_accept(raw, {"key": "铁矿石|港口库存||当期"}, "库存"))
        self.assertEqual(_accept(raw, {"key": "铁矿石|港口库存|47港|当期"}, "库存")["key"], raw)

    def test_downstream_demand_is_scoped_by_commodity(self):
        coke = {"commodity": "焦炭", "indicator": "下游需求", "scope": "", "period": "当期"}
        ore = {"commodity": "铁矿石", "indicator": "下游需求", "scope": "", "period": "当期"}
        self.assertNotEqual(raw_key(coke), raw_key(ore))

    def test_scope_spellings_and_mill_measures_are_unified(self):
        first = {"commodity": "螺纹钢", "indicator": "钢厂盈利率", "scope": "247家样本钢厂", "period": "当期"}
        second = {"commodity": "钢材", "indicator": "钢厂盈利率", "scope": "247家钢厂", "period": "当期"}
        self.assertEqual(raw_key(first), raw_key(second))
        hot_metal = {"commodity": "铁矿石", "indicator": "日均铁水产量", "scope": "全国", "period": "当期"}
        self.assertEqual(raw_key(hot_metal), "铁水|日均铁水产量|全国|当期")
        self.assertIsNotNone(_accept("钢厂|盈利率|247家钢厂|当期", {"key": "钢厂|钢厂盈利率|247家样本钢厂|当期"}, "利润"))

    def test_indicator_prefix_is_tidied(self):
        item = {"commodity": "铁矿石", "indicator": "47港铁矿石库存", "scope": "47港", "period": "当期"}
        self.assertEqual(raw_key(item), "铁矿石|港口库存|47港|当期")


class LinkTest(unittest.TestCase):
    def test_same_wording_and_a_truncated_cut_become_one_trend(self):
        long = "Mysteel62%澳粉远期价格指数周环比上涨1.75美元/吨至108.1美元/吨"
        short = "62%澳粉远期价格指数上涨"
        other = "螺纹钢价格周环比上涨47元/吨至3217元/吨"
        self.assertEqual(canonical_mention(long), "62%澳粉远期价格指数上升")
        self.assertEqual(canonical_mention("上涨"), "")
        mapping = collapse_mentions([long, short, other])
        self.assertEqual(mapping[long], mapping[short])
        self.assertNotEqual(mapping[other], mapping[long])
        self.assertNotRegex(mapping[long], r"\d+\.\d+")
        record = {"entity_relation_dict": [{"Head": long, "Relation": "推升", "Tail": other}],
                  "event_entity_dict": [{"Event": short, "Entity": [long, "108.1美元/吨"]}],
                  "event_relation_dict": []}
        self.assertGreater(collapse_document_labels([record]), 0)
        self.assertEqual(record["event_entity_dict"][0]["Event"], mapping[short])
        self.assertNotIn("108.1美元/吨", record["event_entity_dict"][0]["Entity"])

    def test_same_fact_with_different_wording_merges(self):
        first = event("d1:a", "d1", "铁矿石|港口库存总量|47港|当期", "下降", name="47港库存总量环比去库75万吨")
        second = event("d1:b", "d1", "铁矿石|港存|47港|当期", "下降", name="47港港存下降")
        for item, indicator, when in ((first, "港口库存总量", "本周"), (second, "港存", "本期")):
            item.update(commodity="铁矿石", indicator=indicator, scope="47港", period="当期", time=when)
        merged, _, stats = merge([first, second], [], [])
        self.assertEqual(stats["events_after"], 1)
        self.assertEqual(merged[0]["name"], "本期47港铁矿石港口库存下降")
        self.assertNotIn("75", merged[0]["name"])

    def test_merge_stays_inside_one_document(self):
        shared = "本周47港铁矿石港口库存下降。"
        first = event("d1:a", "d1", "铁矿石|港口库存|47港|当期", "下降", name="本周库存下降")
        second = event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "下降", name="本周港口库存下降")
        third = event("d2:a", "d2", "铁矿石|港口库存|47港|当期", "下降", name="本周库存下降")
        for item in (first, second):
            item.update(time="本周", scope="47港", indicator="港口库存", period="当期",
                        evidence=shared, evidences=[shared])
        third.update(time="本周", scope="47港", indicator="港口库存", period="当期",
                     evidence="另一篇没有写同一句。", evidences=["另一篇没有写同一句。"])
        merged, _, stats = merge([first, second, third], [], [])
        self.assertEqual(stats["events_after"], 2)
        self.assertEqual(sorted(item["doc_id"] for item in merged), ["d1", "d2"])

    def test_continuity_and_expectation_cross_documents(self):
        documents = [{"id": "d1", "report_date": "2026-05-22", "report_series_id": "series:iron"},
                     {"id": "d2", "report_date": "2026-05-29", "report_series_id": "series:iron"}]
        events = [event("d1:a", "d1", "v", "下降"), event("d1:f", "d1", "v", "上升", status="预测中"),
                  event("d2:a", "d2", "v", "上升")]
        links = continuity(events, documents)
        self.assertEqual([(link["head"], link["tail"], link["change"]) for link in links], [("d1:a", "d2:a", "反转")])
        checked = expectation(events, documents)
        self.assertEqual([(link["head"], link["outcome"]) for link in checked], [("d1:f", "兑现")])

    def test_undated_documents_are_not_ordered(self):
        documents = [{"id": "p1", "report_date": ""}, {"id": "p2", "report_date": ""}]
        events = [event("p1:a", "p1", "v", "下降"), event("p2:a", "p2", "v", "上升")]
        self.assertEqual(continuity(events, documents), [])


class ChainTest(unittest.TestCase):
    def test_chain_jumps_over_reports_without_a_step(self):
        events = [event("d1:a", "d1", "供应", "上升"), event("d1:b", "d1", "到港", "上升"),
                  event("d2:b", "d2", "到港", "上升"), event("d3:b", "d3", "到港", "上升"),
                  event("d3:c", "d3", "库存", "上升"), event("d3:d", "d3", "价格", "下降")]
        relations = [
            {"head": "d1:a", "tail": "d1:b", "type": "导致", "confidence": 0.9},
            {"head": "d3:b", "tail": "d3:c", "type": "导致", "confidence": 0.9},
            {"head": "d3:c", "tail": "d3:d", "type": "抑制", "confidence": 0.9},
        ]
        links = [{"head": "d1:b", "tail": "d2:b", "type": "事态延续", "change": "延续"},
                 {"head": "d2:b", "tail": "d3:b", "type": "事态延续", "change": "延续"}]
        classified = {}
        cross, _ = reasoning_chains(events, relations, links, splice_continuity=True, classified=classified)
        self.assertEqual(cross, [])
        wanted = ("d1:a", "d1:b", "d2:b", "d3:b", "d3:c", "d3:d")
        matched = [item for item in classified["candidate"] if item["nodes"] == wanted]
        self.assertEqual(matched[0]["causal_edges"], 3)

    def test_reversal_does_not_carry_a_chain(self):
        events = [event("d1:a", "d1", "供应", "上升"), event("d1:b", "d1", "到港", "上升"),
                  event("d2:b", "d2", "到港", "下降"), event("d2:c", "d2", "库存", "下降"),
                  event("d2:d", "d2", "价格", "上升")]
        relations = [{"head": "d1:a", "tail": "d1:b", "type": "导致", "confidence": 0.9},
                     {"head": "d2:b", "tail": "d2:c", "type": "导致", "confidence": 0.9},
                     {"head": "d2:c", "tail": "d2:d", "type": "导致", "confidence": 0.9}]
        links = [{"head": "d1:b", "tail": "d2:b", "type": "事态延续", "change": "反转"}]
        cross, _ = reasoning_chains(events, relations, links, splice_continuity=True)
        self.assertEqual(cross, [])

    def test_nothing_propagates_past_a_counteracting_factor(self):
        events = [event("d1:a", "d1", "铁水", "上升"), event("d1:b", "d1", "库存", "上升"),
                  event("d1:c", "d1", "价格", "下降"), event("d1:d", "d1", "利润", "上升")]
        relations = [{"head": "d1:a", "tail": "d1:b", "type": "抑制", "confidence": 0.9},
                     {"head": "d1:b", "tail": "d1:c", "type": "导致", "confidence": 0.9},
                     {"head": "d1:c", "tail": "d1:d", "type": "导致", "confidence": 0.9}]
        cross, local = reasoning_chains(events, relations, [])
        self.assertTrue(all(chain["nodes"][0] != "d1:a" for chain in cross + local))


class PolarityTest(unittest.TestCase):
    def test_summary_factor_against_the_known_sign_is_relabelled(self):
        events = [event("d1:a", "d1", "铁水|日均铁水产量||当期", "上升"),
                  event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "上升"),
                  event("d1:c", "d1", "铁矿石|到港量|47港|当期", "上升")]
        relations = [{"head": "d1:a", "tail": "d1:b", "type": "促进", "confidence": 0.9},
                     {"head": "d1:c", "tail": "d1:b", "type": "促进", "confidence": 0.9}]
        self.assertEqual(align_polarity(events, relations), 1)
        self.assertEqual([item["type"] for item in relations], ["促进", "促进"])
        self.assertEqual(relations[0]["polarity"], "POLARITY_CONFLICT")

    def test_abductive_confirmed_relation_is_marked_not_rewritten(self):
        events = [event("d1:a", "d1", "铁水|日均铁水产量||当期", "上升"),
                  event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "上升")]
        confirmed = [{"head": "d1:a", "tail": "d1:b", "type": "跨期传导", "relation": "导致",
                      "confidence": 0.89, "evidence": "铁水上升带动港口库存上升", "evidence_ids": ["ev:d1:b"]}]
        self.assertEqual(align_polarity(events, confirmed, rewrite=False), 1)
        self.assertEqual(confirmed[0]["relation"], "导致")
        self.assertEqual(confirmed[0]["evidence"], "铁水上升带动港口库存上升")
        self.assertEqual(confirmed[0]["confidence"], 0.89)
        self.assertEqual(confirmed[0]["polarity"], "POLARITY_CONFLICT")

    def test_grade_named_in_scope_becomes_the_commodity(self):
        item = {"commodity": "铁矿石", "indicator": "到港量", "scope": "47港/澳矿", "period": "当期"}
        self.assertEqual(raw_key(item), "澳矿|到港量|47港|当期")


class ExplicitLinkTest(unittest.TestCase):
    def test_order_needs_one_sentence_and_a_cue(self):
        source = "发运增加之后到港增加。库存则另行统计。"
        self.assertEqual(normalize_event_relation("发运增加", "之前", "到港增加", source), "之前")
        self.assertIsNone(normalize_event_relation("发运增加", "之后", "库存下降", source))
        self.assertIsNone(normalize_event_relation("发运增加", "之前", "到港增加", "发运增加。到港增加。"))

    def test_order_cue_stays_inside_one_sentence_and_causal_checks_stay(self):
        source = "发运增加之后到港增加。库存则另行统计。"
        self.assertEqual(normalize_event_relation("发运增加", "之后", "到港增加", source), "之后")
        self.assertIsNone(normalize_event_relation("发运增加", "之后", "到港增加", "发运增加。到港增加。之后统计库存。"))
        self.assertEqual(normalize_event_relation("发运增加", "导致", "到港增加", "发运增加导致到港增加。"), "导致")
        self.assertIsNone(normalize_event_relation("发运增加", "导致", "库存下降", source))
        self.assertEqual(normalize_event_relation("发运增加", "同时", "到港增加", "发运增加。到港增加同时发生"), "同时")

    def test_contents_page_is_not_a_second_copy_of_the_summary(self):
        units = [
            {"page": 2, "kind": "section_header", "text": "目录 CONTENTS"},
            {"page": 2, "kind": "text", "text": "库存：47港铁矿石库存环比去库"},
            {"page": 3, "kind": "text", "text": "中国47港铁矿石库存环比去库228万吨。"},
        ]
        self.assertEqual(contents_pages(units), {2})
        chunks = make_chunks({"id": "d", "units": units, "text_sha256": "x"}, limit=5000)
        joined = "\n".join(chunk["text"] for chunk in chunks)
        self.assertNotIn("目录", joined)
        self.assertIn("228万吨", joined)

    def test_outline_title_merges_into_the_body_sentence(self):
        title = "库存：47港铁矿石库存环比去库"
        body = "中国47港铁矿石库存环比去库228万吨，较年初累库744万吨。"
        self.assertTrue(contained_duplicate(title, body))
        self.assertFalse(contained_duplicate("45港铁矿石库存环比去库", body))
        short = event("d1:a", "d1", "铁矿石|市场综述||当期", "下降", name=title)
        long = event("d1:b", "d1", "铁矿石|港口库存|47港|当期", "下降", name="47港库存下降")
        short.update(commodity="铁矿石", indicator="市场综述", scope="", period="当期", evidence=title, evidences=[title])
        long.update(commodity="铁矿石", indicator="港口库存", scope="47港", period="当期", evidence=body, evidences=[body])
        merged, _, stats = merge([short, long], [], [])
        self.assertEqual(stats["events_after"], 2)
        self.assertEqual(stats["outline_merges"], 0)


class EndToEndTest(unittest.TestCase):
    def test_ten_documents_offline(self):
        output = ROOT / "example" / "generated" / "_eventkg_test"
        shutil.rmtree(output, ignore_errors=True)
        result, report = run(DATA, output, workers=4, chat=FakeChat(), expected_docs=10)
        self.assertGreater(report["instance_events"], 50)
        self.assertGreater(report["cross_doc_edges"], 0)
        for chain in result["reasoning_chains"]:
            self.assertGreaterEqual(len(chain["docs"]), 2)
            self.assertGreaterEqual(len(chain["edges"]), 3)
            self.assertTrue(chain["evidence_status"])
            self.assertTrue(all(state in ("VERIFIED", "TEMPORAL_VERIFIED") for state in chain["evidence_status"]))
        cypher = (output / "event_layer.cypher").read_text(encoding="utf-8")
        statements = [line for line in cypher.splitlines() if line and not line.startswith("//")]
        self.assertTrue(all(line.endswith(";") for line in statements))
        self.assertIn("MERGE (n:`实例事件` {uid: row.uid}) SET n += row.props SET n:`事件`", cypher)
        graph = json.loads((output / "graph.json").read_text(encoding="utf-8"))
        from eventkg.retrieve import search
        found = search(graph, "铁矿石港口库存为什么下降")
        self.assertTrue(found["seeds"])


if __name__ == "__main__":
    unittest.main()
