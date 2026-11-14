# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

from eventkg.evaluate_cross import compare_slice, iron_ore_slice, pending_reviews, run_ablation
from eventkg.export import write_all
from eventkg.graph_induction import (
    _evidence_role,
    _relation_pairs,
    attach_mentions,
    find_indirect_paths,
    important_targets,
    index_evidence,
    resolve_event_time,
    propose_graphs,
    report_series_id,
    retrieve_neighborhood,
    select_graph,
    select_joint,
)
from eventkg.link import abstract_rules
from eventkg.paths import allows_causal_walk, reasoning_chains


def event(event_id, doc, var, direction, evidence, salience="核心", commodity="铁矿石", indicator=""):
    commodity = commodity or var.split("|")[0]
    indicator = indicator or (var.split("|")[1] if "|" in var else "")
    return {"id": event_id, "doc_id": doc, "var": var, "abstract": var + "|" + direction, "dir": direction,
            "status": "已发生", "name": event_id, "evidence": evidence, "salience": salience,
            "commodity": commodity, "indicator": indicator, "sections": ["s"], "page": 1, "scope": "", "dim": ""}


class EvidenceTest(unittest.TestCase):
    def test_event_facts_without_a_relation_sentence_are_not_a_cause(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        cause = event("d1:a", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般", commodity="铁水", indicator="日均铁水产量")
        effect = event("d2:b", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨")
        evidence = index_evidence([cause, effect], documents, [])
        choice = select_graph(effect, [{
            "id": "h1", "kind": "单一直接原因", "why": "两件事都发生了", "missing": "", "needs_retrieval": False,
            "edges": [{"head": "d1:a", "tail": "d2:b", "relation": "导致",
                       "evidence_ids": ["ev:d1:a", "ev:d2:b"], "counter_evidence_ids": [], "why": "各有原文"}],
        }], [{"id": "d1:a"}], [cause, effect], documents, evidence)
        self.assertEqual(choice["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertEqual(choice["edges"], [])
        self.assertIn("缺少能同时覆盖两端且已对齐到事件实例的关系支持证据", "；".join(choice["hypotheses"][0]["reasons"]))

    def test_a_quote_without_an_evidence_id_is_not_accepted(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        cause = event("d1:a", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般", commodity="铁水", indicator="日均铁水产量")
        effect = event("d2:b", "d2", "铁矿石|价格||当期", "上升", "日均铁水产量上升带动矿价上涨")
        evidence = index_evidence([cause, effect], documents, [])
        choice = select_graph(effect, [{
            "id": "h1", "kind": "单一直接原因", "why": "只贴了一段原文", "edges": [
                {"head": "d1:a", "tail": "d2:b", "relation": "导致", "evidence": "日均铁水产量上升带动矿价上涨"}],
        }], [{"id": "d1:a"}], [cause, effect], documents, evidence)
        self.assertEqual(choice["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertIn("没有证据ID", choice["hypotheses"][0]["reasons"][0])

    def test_explicit_link_is_confirmed_and_a_reversal_is_not(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-04-24"},
            {"id": "d4", "report_date": "2026-05-22"},
        ]
        hot = event("d1:hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般", commodity="铁水", indicator="日均铁水产量")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "2026-04-17日均铁水产量上升带动本期矿价上涨")
        margin_up = event("d2:margin", "d2", "钢厂|盈利率||当期", "上升", "钢厂盈利率回升至64%", salience="一般", commodity="钢厂", indicator="盈利率")
        margin_down = event("d4:margin", "d4", "钢厂|盈利率||当期", "下降", "矿价上涨挤压钢厂利润，盈利率下降", commodity="钢厂", indicator="盈利率")
        early = event("d1:price", "d1", "铁矿石|价格||当期", "上升", "本期矿价上涨", salience="一般")
        early["event_time"] = "2026-04-17"
        margin_up["event_time"] = "2026-04-24"
        margin_down["event_time"] = "2026-05-22"
        events = [hot, price, margin_up, margin_down, early]
        evidence = index_evidence(events, documents, [])
        kept = select_graph(price, [{
            "id": "h1", "kind": "单一直接原因", "why": "后文写明带动", "edges": [
                {"head": "d1:hot", "tail": "d2:price", "relation": "促进", "evidence_ids": ["ev:d2:price"],
                 "counter_evidence_ids": [], "why": "点名铁水和矿价"}],
        }], [{"id": "d1:hot"}], events, documents, evidence)
        self.assertEqual(kept["decision"], "h1")
        self.assertEqual(kept["edges"][0]["relation_class"], "ASSERTED_CAUSE")
        self.assertEqual(kept["edges"][0]["evidence_ids"], ["ev:d2:price"])
        self.assertLess(kept["edges"][0]["model_score"], 1)
        self.assertEqual(kept["edges"][0]["score_kind"], "model_score")
        blocked = select_graph(margin_down, [{
            "id": "h2", "kind": "单一直接原因", "why": "后文写挤压", "edges": [
                {"head": "d1:price", "tail": "d4:margin", "relation": "抑制", "evidence_ids": ["ev:d4:margin"],
                 "counter_evidence_ids": [], "why": "点名矿价和利润"}],
        }], [{"id": "d1:price"}], events, documents, evidence)
        self.assertEqual(blocked["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertIn("中间出现反向观测", "；".join(blocked["hypotheses"][0]["reasons"]))

    def test_same_indicator_and_direction_do_not_pick_another_weeks_instance(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"},
                     {"id": "d0", "report_date": "2026-04-10"}]
        earlier = event("d0:hot", "d0", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般",
                        commodity="铁水", indicator="日均铁水产量")
        hot = event("d1:hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般",
                    commodity="铁水", indicator="日均铁水产量")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "日均铁水产量上升带动本期矿价上涨")
        events = [earlier, hot, price]
        evidence = index_evidence(events, documents, [])
        choice = select_graph(price, [{
            "id": "h-wrong", "kind": "单一直接原因", "why": "指标和方向一样", "edges": [
                {"head": "d0:hot", "tail": "d2:price", "relation": "导致", "evidence_ids": ["ev:d2:price"],
                 "why": "没有日期"}],
        }], [{"id": "d0:hot"}, {"id": "d1:hot"}], events, documents, evidence)
        self.assertEqual(choice["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertIn("同一次具体事件", "；".join(choice["hypotheses"][0]["reasons"]))
        kept = select_graph(price, [{
            "id": "h-dated", "kind": "单一直接原因", "why": "句子写了日期", "edges": [
                {"head": "d1:hot", "tail": "d2:price", "relation": "导致",
                 "evidence_ids": ["ev:d2:price"], "why": "4月17日的铁水"}],
        }], [{"id": "d1:hot"}], events, documents, evidence)
        self.assertEqual(kept["decision"], "NO_SUPPORTED_EXPLANATION")
        dated = event("d2:dated", "d2", "铁矿石|价格||当期", "上升", "2026-04-17日均铁水产量上升带动本期矿价上涨")
        aligned_events = [earlier, hot, dated]
        evidence = index_evidence(aligned_events, documents, [])
        aligned = select_graph(dated, [{
            "id": "h-ok", "kind": "单一直接原因", "why": "日期对准4月17日", "edges": [
                {"head": "d1:hot", "tail": "d2:dated", "relation": "促进", "evidence_ids": ["ev:d2:dated"],
                 "why": "日期锁定"}],
        }], [{"id": "d1:hot"}], aligned_events, documents, evidence)
        self.assertEqual(aligned["edges"][0]["relation_class"], "ASSERTED_CAUSE")
        self.assertEqual(aligned["edges"][0]["evidence_bundle"]["cause_instance"][0], "ev:d2:dated")

    def test_two_causal_clauses_do_not_share_a_trigger(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        quote = "2026-04-17发运下降导致2026-04-24到港下降；2026-04-17铁水上升推高2026-04-24矿价。"
        ship = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", salience="一般", indicator="发运量")
        arrive = event("d2:arrive", "d2", "铁矿石|到港量||当期", "下降", quote, indicator="到港量")
        hot = event("d1:hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般",
                    commodity="铁水", indicator="日均铁水产量")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨", salience="一般")
        events = [ship, arrive, hot, price]
        evidence = index_evidence(events, documents, [])

        def choose(head, tail, target, relation="导致"):
            return select_graph(target, [{
                "id": "h", "kind": "单一直接原因", "why": "整句里有触发词", "edges": [
                    {"head": head, "tail": tail, "relation": relation, "evidence_ids": ["ev:d2:arrive"], "why": "同段"}],
            }], [{"id": head}], events, documents, evidence)

        crossed = choose("d1:ship", "d2:price", price)
        self.assertEqual(crossed["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertIn("同一关系分句", "；".join(crossed["hypotheses"][0]["reasons"]))
        self.assertEqual(choose("d1:ship", "d2:arrive", arrive)["edges"][0]["relation_class"], "ASSERTED_CAUSE")
        self.assertEqual(choose("d1:hot", "d2:price", price, "促进")["edges"][0]["relation_class"], "ASSERTED_CAUSE")

    def test_repeated_indicator_uses_each_date(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        early = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        later = event("d2:ship", "d2", "铁矿石|发运量||当期", "下降",
                      "2026-04-17发运下降，2026-04-24发运下降", indicator="发运量")
        catalog = index_evidence([early, later], documents, [])
        attach_mentions(catalog, [early, later], documents)
        mentions = [mention for item in catalog if item["evidence_id"] == "ev:d2:ship"
                    for mention in item["mentions"] if mention["indicator"] == "发运量"]
        self.assertEqual(len(mentions), 2)
        self.assertEqual(mentions[0]["linked_event_id"], "d1:ship")
        self.assertEqual(mentions[1]["linked_event_id"], "d2:ship")
        self.assertLess(mentions[0]["char_start"], mentions[1]["char_start"])

    def test_previous_period_does_not_bind_the_current_shipment(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17", "report_series_id": "mysteel:铁矿石:weekly"},
            {"id": "d2", "report_date": "2026-04-24", "report_series_id": "mysteel:铁矿石:weekly"},
        ]
        previous = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        current = event("d2:ship", "d2", "铁矿石|发运量||当期", "下降", "前期发运下降", indicator="发运量")
        catalog = index_evidence([previous, current], documents, [])
        attach_mentions(catalog, [previous, current], documents)
        mention = [item for item in catalog if item["evidence_id"] == "ev:d2:ship"][0]["mentions"][0]
        self.assertEqual(mention["period"], "前期")
        self.assertEqual(mention["observation_period"], "前期")
        self.assertEqual(mention["linked_event_id"], "d1:ship")
        self.assertNotEqual(mention["linked_event_id"], "d2:ship")
        self.assertEqual(mention["report_date"], "2026-04-24")
        self.assertEqual(mention["event_time"], "")
        self.assertNotEqual(mention["event_time"], mention["report_date"])

    def test_every_merged_evidence_can_be_retrieved_once(self):
        documents = [{"id": "d1", "report_date": "2026-04-17", "windows": [{
            "id": "d1:w1", "text": "综述\n澳洲发运下降。2026-04-17发运下降导致2026-04-24到港下降。",
            "sections": ["供应"], "char_start": 0,
        }]}]
        ship = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        ship["evidences"] = ["澳洲发运下降", "2026-04-17发运下降导致2026-04-24到港下降"]
        ship["paragraph_id"] = "d1:p3"
        ship["sentence_id"] = "d1:s2"
        ship["page"] = 4
        ship["section"] = "供应"
        relation = {"head": "d1:ship", "tail": "d2:arrive", "type": "导致", "doc_id": "d1",
                    "evidence": "2026-04-17发运下降导致2026-04-24到港下降",
                    "evidences": ["2026-04-17发运下降导致2026-04-24到港下降", "另一句到港补充"],
                    "page": 4, "section": "供应"}
        catalog = index_evidence([ship], documents, [relation])
        by_id = {item["evidence_id"]: item for item in catalog}
        self.assertEqual(by_id["ev:d1:ship"]["text"], "澳洲发运下降")
        self.assertEqual(by_id["ev:d1:ship"]["paragraph_id"], "d1:p3")
        self.assertEqual(by_id["ev:d1:ship"]["sentence_id"], "d1:s2")
        self.assertEqual(by_id["ev:d1:ship"]["page"], 4)
        shared = by_id["ev:rel:d1:ship:d2:arrive:0"]
        self.assertIn("发运下降导致", shared["text"])
        self.assertFalse(shared["counted"])
        self.assertEqual({item["kind"] for item in shared["sources"]}, {"event", "relation"})
        counted = [item for item in catalog if item["text"] == shared["text"] and item["counted"]]
        self.assertEqual(len(counted), 1)
        self.assertTrue(any(item["text"] == "另一句到港补充" for item in catalog))
        located = by_id["ev:d1:ship:1"]
        self.assertEqual(located["paragraph_id"], "d1:p3")
        self.assertIsInstance(located["char_start"], int)
        self.assertGreater(located["char_start"], 0)
        self.assertGreater(located["char_end"], located["char_start"])

    def test_forecast_result_is_not_an_occurred_cause(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        hot = event("d1:hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般",
                    commodity="铁水", indicator="日均铁水产量")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "2026-04-17日均铁水产量上升带动本期矿价上涨")
        price["status"] = "预测中"
        evidence = index_evidence([hot, price], documents, [])
        choice = select_graph(price, [{
            "id": "h", "kind": "单一直接原因", "why": "句子像已发生", "edges": [
                {"head": "d1:hot", "tail": "d2:price", "relation": "导致", "evidence_ids": ["ev:d2:price"], "why": "带动"}],
        }], [{"id": "d1:hot"}], [hot, price], documents, evidence)
        self.assertEqual(choice["edges"], [])
        self.assertEqual(choice["hypotheses"][0]["edge_scores"][0]["relation_class"], "FORECASTS")


class RecallTest(unittest.TestCase):
    def test_rules_are_a_recall_reason_and_later_events_stay_visible(self):
        secret = "库存上升导致价格下降"
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"},
                     {"id": "d3", "report_date": "2026-05-01"}]
        inv = event("d1:inv", "d1", "铁矿石|港口库存||当期", "上升", "港口库存上升", salience="一般", indicator="港口库存")
        old = event("d1:old", "d1", "铁矿石|价格||当期", "下降", secret, salience="一般")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "下降", "本期矿价下跌")
        later = event("d3:stock", "d3", "铁矿石|港口库存||当期", "下降", "到港下降带动港口库存下降", salience="一般", indicator="港口库存")
        events = [inv, old, price, later]
        relations = [{"head": "d1:inv", "tail": "d1:old", "type": "导致", "evidence": secret}]
        rules = abstract_rules(events, relations)
        neighborhood = retrieve_neighborhood(price, events, documents, relations, rules, top_k=5)
        dumped = json.dumps(neighborhood, ensure_ascii=False)
        self.assertNotIn("rule_evidence", dumped)
        inv_row = next(item for item in neighborhood["candidates"] if item["id"] == "d1:inv")
        self.assertIn("不作为本次因果证据", " ".join(item["reason"] for item in inv_row["recall"]))
        later_row = next(item for item in neighborhood["candidates"] if item["id"] == "d3:stock")
        self.assertEqual(later_row["time_relation"], "after")
        self.assertIn("不能作为该目标的原因", later_row["causal_use"])
        evidence = index_evidence(events, documents, relations)
        blocked = select_graph(event("d2:up", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨"), [{
            "id": "h-rule", "kind": "单一直接原因", "why": "套用历史句式", "edges": [
                {"head": "d1:inv", "tail": "d2:up", "relation": "抑制", "evidence_ids": ["ev:rel:d1:inv:d1:old:0"],
                 "why": "历史句式"}],
        }], neighborhood, events + [event("d2:up", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨")], documents, evidence)
        self.assertEqual(blocked["decision"], "NO_SUPPORTED_EXPLANATION")

    def test_topk_reports_every_event_it_does_not_keep(self):
        documents = [{"id": "d0", "report_date": "2026-04-17"}] + [
            {"id": "d%d" % index, "report_date": "2026-04-24"} for index in range(1, 8)]
        target = event("d0:t", "d0", "铁矿石|价格||当期", "上升", "矿价上涨")
        others = [event("d%d:e" % index, "d%d" % index, "铁矿石|港口库存||当期", "上升", "库存上升",
                        salience="一般", indicator="港口库存") for index in range(1, 8)]
        neighborhood = retrieve_neighborhood(target, [target] + others, documents, [], top_k=3)
        self.assertEqual(neighborhood["stats"]["kept"], 3)
        self.assertEqual(neighborhood["stats"]["dropped_by_topk"], neighborhood["stats"]["hits"] - 3)
        self.assertEqual(len(neighborhood["stats"]["omitted_ids"]), neighborhood["stats"]["dropped_by_topk"])

    def test_event_time_recalls_without_a_report_date(self):
        documents = [{"id": "d1"}, {"id": "d2", "report_date": "2026-04-24"}]
        cause = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        effect = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨", indicator="价格")
        cause["event_time"] = "2026-04-17"
        effect["event_time"] = "2026-04-24"
        events = [cause, effect]
        targets, stats = important_targets(events, documents)
        self.assertIn("d1:ship", {item["id"] for item in targets})
        self.assertEqual(stats["skip_reasons"].get("不是核心事件", 0), 0)
        neighborhood = retrieve_neighborhood(effect, events, documents, [])
        row = neighborhood["candidates"][0]
        self.assertEqual(row["id"], "d1:ship")
        self.assertEqual(row["time_basis"], "event_time")
        self.assertEqual(row["time_relation"], "before")
        self.assertEqual(row["report_date"], "")
        self.assertEqual(row["event_time"], "2026-04-17")

    def test_report_date_does_not_override_event_time(self):
        documents = [
            {"id": "d1", "report_date": "2026-05-20"},
            {"id": "d2", "report_date": "2026-04-01"},
        ]
        cause = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        effect = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "矿价上涨", indicator="价格")
        cause["event_time"] = "2026-04-17"
        effect["event_time"] = "2026-05-15"
        neighborhood = retrieve_neighborhood(effect, [cause, effect], documents, [])
        row = neighborhood["candidates"][0]
        self.assertTrue(row["time_conflict"])
        self.assertEqual(row["time_basis"], "event_time")
        self.assertEqual(row["time_relation"], "before")
        self.assertEqual(row["report_date"], "2026-05-20")
        self.assertNotEqual(row["event_time"], row["report_date"])
        cause["event_time"] = "2026-05-20"
        effect["event_time"] = "2026-04-17"
        effect["evidence"] = "发运下降导致矿价上涨"
        evidence = index_evidence([cause, effect], documents, [])
        choice = select_graph(effect, [{
            "id": "h", "kind": "单一直接原因", "why": "报告更早", "edges": [{
                "head": "d1:ship", "tail": "d2:price", "relation": "导致",
                "evidence_ids": ["ev:d2:price"], "why": "发布时间相反"}],
        }], [{"id": "d1:ship"}], [cause, effect], documents, evidence)
        scored = choice["hypotheses"][0]["edge_scores"][0]
        self.assertNotEqual(scored["relation_class"], "ASSERTED_CAUSE")
        self.assertNotEqual(scored["relation_class"], "INFERRED_SUPPORTED")
        self.assertIn("时间不兼容", "；".join(scored["reasons"]))

    def test_unknown_time_keeps_the_candidate_but_not_the_direction(self):
        documents = [{"id": "d1"}, {"id": "d2"}]
        cause = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        effect = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "矿价上涨", indicator="价格")
        neighborhood = retrieve_neighborhood(effect, [cause, effect], documents, [])
        self.assertEqual(neighborhood["stats"]["skipped"], 0)
        row = neighborhood["candidates"][0]
        self.assertEqual(row["time_basis"], "TEMPORAL_UNKNOWN")
        self.assertEqual(row["time_relation"], "TEMPORAL_UNKNOWN")
        self.assertIn("不能确认因果方向", row["causal_use"])
        evidence = index_evidence([cause, effect], documents, [])
        choice = select_graph(effect, [{
            "id": "h", "kind": "单一直接原因", "why": "没有时间", "edges": [{
                "head": "d1:ship", "tail": "d2:price", "relation": "导致",
                "evidence_ids": ["ev:d1:ship", "ev:d2:price"], "why": "只有事实"}],
        }], neighborhood, [cause, effect], documents, evidence)
        scored = choice["hypotheses"][0]["edge_scores"][0]
        self.assertNotIn(scored["relation_class"], ("ASSERTED_CAUSE", "INFERRED_SUPPORTED"))
        self.assertEqual(choice["edges"], [])
        outside = event("d3:old", "d3", "铁矿石|港口库存||当期", "上升", "库存上升", indicator="港口库存")
        outside["event_time"] = "2026-01-01"
        effect["event_time"] = "2026-05-15"
        wide = retrieve_neighborhood(effect, [cause, effect, outside], documents + [{"id": "d3"}], [])
        self.assertNotIn("d3:old", wide["stats"]["skipped_ids"])
        self.assertIn("d3:old", [item["id"] for item in wide["candidates"]])


class GraphScoreTest(unittest.TestCase):
    def test_three_supported_edges_are_not_rejected_for_being_long(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-20"},
                     {"id": "d3", "report_date": "2026-04-24"}, {"id": "d4", "report_date": "2026-04-27"}]
        quote = "2026-04-17发运下降导致2026-04-20到港下降，2026-04-20到港下降带动2026-04-24库存下降，2026-04-24库存下降推高2026-04-27矿价"
        nodes = [
            event("d1:a", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", salience="一般", indicator="发运量"),
            event("d2:b", "d2", "铁矿石|到港量||当期", "下降", quote, salience="一般", indicator="到港量"),
            event("d3:c", "d3", "铁矿石|港口库存||当期", "下降", quote, salience="一般", indicator="港口库存"),
            event("d4:d", "d4", "铁矿石|价格||当期", "上升", quote, indicator="价格"),
        ]
        evidence = index_evidence(nodes, documents, [])
        choice = select_graph(nodes[-1], [{
            "id": "h-long", "kind": "间接影响", "why": "每一步原文都写了带动", "edges": [
                {"head": "d1:a", "tail": "d2:b", "relation": "导致", "evidence_ids": ["ev:d2:b"], "why": "发运到到港"},
                {"head": "d2:b", "tail": "d3:c", "relation": "促进", "evidence_ids": ["ev:d3:c"], "why": "到港到库存"},
                {"head": "d3:c", "tail": "d4:d", "relation": "促进", "evidence_ids": ["ev:d4:d"], "why": "库存到矿价"},
            ],
        }], [{"id": "d1:a"}, {"id": "d2:b"}, {"id": "d3:c"}], nodes, documents, evidence)
        self.assertNotEqual(choice["decision"], "NO_SUPPORTED_EXPLANATION")
        self.assertEqual(len(choice["edges"]), 3)
        self.assertEqual(choice["hypotheses"][0]["graph_scores"]["parsimony"], 0)


class ClosedSetTest(unittest.TestCase):
    def test_scripted_comparison_is_not_reported_as_model_accuracy(self):
        scores = compare_slice()
        self.assertEqual(scores["report_type"], "functional_test")
        self.assertIsNone(scores["model_accuracy"])
        self.assertTrue(scores["abductive"]["complete"])
        self.assertGreater(scores["abductive"]["f1"], scores["legacy"]["f1"])
        self.assertEqual(scores["abductive"]["fp"], 0)
        self.assertEqual(scores["candidate_recall_at_8"]["recall"], 1)
        self.assertEqual(scores["evidence_supported_edge_precision"]["precision"], 1)
        self.assertEqual({(edge["head"], edge["tail"]) for edge in scores["edges"]}, {
            ("r1:hot", "r2:price"), ("r1:ship", "r2:arrive"), ("r2:arrive", "r3:stock"),
            ("r1:price", "r4:cost"), ("r1:coke", "r4:cost"),
        })
        pending = pending_reviews([("r9:new", "r9:other")], [])
        self.assertEqual(pending[0]["reviewed"], False)

    def test_unreviewed_predictions_block_the_headline_f1(self):
        from eventkg.evaluate_cross import score_relations
        gold = [{"head": "a", "tail": "b", "label": 1, "reviewed": True, "scope": "cross"}]
        scored = score_relations([("a", "b"), ("c", "d")], gold)
        self.assertFalse(scored["complete"])
        self.assertIsNone(scored["f1"])
        self.assertEqual(scored["pending"][0]["head"], "c")


class JointTest(unittest.TestCase):
    def test_global_selection_keeps_an_edge_from_a_nonwinning_hypothesis(self):
        documents, events, _relations = iron_ore_slice()
        from eventkg.graph_induction import induce

        class Chat:
            def complete_json(self, system, user, max_tokens=4096, tag=""):
                target = json.loads(user)["target"]["id"]
                if target != "r4:cost":
                    return {"hypotheses": [{"id": "h0", "kind": "证据不足", "why": "其他目标不参与", "edges": []}]}
                return {"hypotheses": [
                    {"id": "h-short", "kind": "单一直接原因", "why": "只留矿价", "edges": [
                        {"head": "r1:price", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                         "why": "矿价"}]},
                    {"id": "h-both", "kind": "多个共同原因", "why": "矿价和焦炭", "edges": [
                        {"head": "r1:price", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                         "why": "矿价"},
                        {"head": "r1:coke", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                         "why": "焦炭"}]},
                ]}

        local, local_payload = induce(Chat(), events, _relations, documents, workers=1)
        joint, joint_payload = induce(Chat(), events, _relations, documents, workers=1, joint=True)
        self.assertIn(("r1:price", "r4:cost"), {(edge["head"], edge["tail"]) for edge in local})
        self.assertNotIn(("r1:coke", "r4:cost"), {(edge["head"], edge["tail"]) for edge in local})
        self.assertIn(("r1:coke", "r4:cost"), {(edge["head"], edge["tail"]) for edge in joint})
        coke = [edge for edge in joint if edge["head"] == "r1:coke"][0]
        self.assertIn("h-both", coke["hypothesis_ids"])
        self.assertTrue(joint_payload["candidate_graph"])

    def test_human_review_of_real_sentences_is_not_a_method_f1(self):
        from eventkg.evaluate_cross import score_human_review

        review = score_human_review()
        self.assertIsNone(review["method_f1"])
        self.assertEqual(review["reviewed"], 4)
        self.assertTrue(all(item["agrees_with_human"] for item in review["items"]))
        mismatch = [item for item in review["items"] if item["id"] == "coke-wrong-week"][0]
        self.assertFalse(mismatch["system_confirmed"])
        self.assertIn("同一次具体事件", "；".join(mismatch["system_reasons"]))
    def test_split_evidence_needs_both_passages_and_a_trigger_on_the_cause(self):
        documents, events, _relations = iron_ore_slice(extended=True)
        evidence = index_evidence(events, documents, [])
        ship = "r1:ship"
        freight = "r3:freight"

        def record(target, evidence_ids):
            edge = {"head": ship, "tail": freight, "relation": "促进", "relation_class": "NONE",
                    "evidence_ids": evidence_ids, "support_ids": [], "counter_evidence_ids": [],
                    "components": {}, "edge_score": 0, "reasons": [], "confirmed": False, "why": "",
                    "missing": "", "gap_days": 28}
            return {"target": target, "candidates": [{"id": ship}, {"id": freight}], "hypotheses": [{
                "id": "h", "kind": "单一直接原因", "why": "", "missing": "", "needs_retrieval": False,
                "nodes": [], "edge_scores": [edge],
            }]}

        separate = select_joint([
            record("r3:freight", ["ev:r3:freight"]),
            record("r4:push", ["ev:r4:push"]),
        ], evidence, events, documents, multi_source=False)
        self.assertNotIn((ship, freight), {(edge["head"], edge["tail"]) for edge in separate["confirmed"]})
        pooled = select_joint([
            record("r3:freight", ["ev:r3:freight"]),
            record("r4:push", ["ev:r4:push"]),
        ], evidence, events, documents, multi_source=True)
        kept = [edge for edge in pooled["confirmed"] if (edge["head"], edge["tail"]) == (ship, freight)]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["relation_class"], "INFERRED_SUPPORTED")
        self.assertEqual(kept[0]["components"]["independent_passages"], 2)
        copied = dict(evidence[0])
        for item in evidence:
            if item["evidence_id"] == "ev:r4:push":
                copied = dict(item, evidence_id="ev:r4:push-copy")
        evidence.append(copied)
        duplicated = select_joint([
            record("r3:freight", ["ev:r3:freight"]),
            record("r4:push", ["ev:r4:push", "ev:r4:push-copy"]),
        ], evidence, events, documents, multi_source=True)
        again = [edge for edge in duplicated["confirmed"] if edge["tail"] == freight][0]
        self.assertEqual(again["components"]["independent_passages"], 2)
        counted = [item for item in again["evidence_sources"] if item["counted"]]
        self.assertEqual(len(counted), 2)

    def test_chain_and_time_do_not_confirm_a_cause(self):
        documents, events, _relations = iron_ore_slice(extended=True)
        evidence = index_evidence(events, documents, [])
        edge = {"head": "r1:hot", "tail": "r4:cost", "relation": "导致", "relation_class": "NONE",
                "evidence_ids": ["ev:r1:hot", "ev:r4:cost"], "support_ids": [], "counter_evidence_ids": [],
                "components": {}, "edge_score": 0, "reasons": [], "confirmed": False, "why": "上游",
                "missing": "", "gap_days": 35}
        choice = select_joint([{
            "target": "r4:cost", "candidates": [], "hypotheses": [{
                "id": "h-chain", "kind": "间接影响", "why": "产业链", "missing": "", "needs_retrieval": False,
                "nodes": [], "edge_scores": [edge]}],
        }], evidence, events, documents, multi_source=True)
        self.assertFalse(any(item["head"] == "r1:hot" and item["tail"] == "r4:cost" for item in choice["confirmed"]))
        rejected = [item for item in choice["possible"] + choice["rejected"]
                    if item["head"] == "r1:hot" and item["tail"] == "r4:cost"]
        self.assertTrue(rejected)
        self.assertIn("不能靠时间或产业链", rejected[0]["rejection_reason"])

    def test_ablation_is_measured_and_not_reported_as_model_accuracy(self):
        report = run_ablation()
        self.assertIsNone(report["model_accuracy"])
        modes = report["modes"]
        pairs = {name: {(item["head"], item["tail"]) for item in modes[name]["confirmed"]} for name in "ABCD"}
        self.assertNotIn(("r1:coke", "r4:cost"), pairs["A"])
        self.assertNotIn(("r1:ship", "r3:freight"), pairs["A"])
        self.assertIn(("r1:ship", "r3:freight"), pairs["B"])
        self.assertNotIn(("r1:coke", "r4:cost"), pairs["B"])
        self.assertIn(("r1:coke", "r4:cost"), pairs["C"])
        self.assertNotIn(("r1:ship", "r3:freight"), pairs["C"])
        self.assertIn(("r1:ship", "r3:freight"), pairs["D"])
        self.assertIn(("r1:coke", "r4:cost"), pairs["D"])
        for name in "ABCD":
            self.assertNotIn(("r1:hot", "r4:cost"), pairs[name])
            self.assertNotIn(("r1:price", "r2:price"), pairs[name])
            self.assertNotIn(("r1:inv", "r2:price"), pairs[name])
            self.assertEqual(modes[name]["path_precision"]["precision"], modes["A"]["path_precision"]["precision"])
            self.assertTrue(modes[name]["relation"]["complete"])
        freight = [item for item in modes["D"]["edges"] if item["head"] == "r1:ship" and item["tail"] == "r3:freight"]
        self.assertEqual(freight[0]["relation_class"], "INFERRED_SUPPORTED")
        self.assertTrue(freight[0]["evidence_sources"])
        self.assertTrue(freight[0]["components"])
        self.assertTrue(freight[0]["alternatives"] or freight[0]["sources"])
        cost = [item for item in modes["D"]["edges"] if item["tail"] == "r4:cost" and item["status"] == "confirmed"]
        self.assertGreaterEqual(len(cost), 2)

    def test_multi_cause_hypothesis_is_not_treated_like_rival_single_causes(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        quote = "2026-04-17日均铁水产量上升带动本期矿价上涨；2026-04-17发运下降导致本期矿价上涨。"
        hot = event("d1:hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量上升", salience="一般",
                    commodity="铁水", indicator="日均铁水产量")
        ship = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", salience="一般", indicator="发运量")
        price = event("d2:price", "d2", "铁矿石|价格||当期", "上升", quote)
        events = [hot, ship, price]
        evidence = index_evidence(events, documents, [])
        edges = [
            {"head": "d1:hot", "tail": "d2:price", "relation": "促进", "evidence_ids": ["ev:d2:price"], "why": "铁水"},
            {"head": "d1:ship", "tail": "d2:price", "relation": "导致", "evidence_ids": ["ev:d2:price"], "why": "发运"},
        ]
        together = select_graph(price, [{
            "id": "h-both", "kind": "多个共同原因", "why": "两句各自写了原因", "edges": edges,
        }], [{"id": "d1:hot"}, {"id": "d1:ship"}], events, documents, evidence)
        joint = select_joint([together], evidence, events, documents)
        self.assertEqual({(item["head"], item["tail"]) for item in joint["confirmed"]},
                         {("d1:hot", "d2:price"), ("d1:ship", "d2:price")})
        multi = [item for item in joint["competitions"] if item["type"] == "MULTIPLE_CONTRIBUTORS"]
        self.assertEqual(multi[0]["hypothesis_ids"], ["h-both"])
        self.assertEqual(multi[0]["status"], "compatible")
        rivals = select_graph(price, [
            {"id": "h-hot", "kind": "单一直接原因", "why": "只认铁水", "edges": [edges[0]]},
            {"id": "h-ship", "kind": "单一直接原因", "why": "只认发运", "edges": [edges[1]]},
        ], [{"id": "d1:hot"}, {"id": "d1:ship"}], events, documents, evidence)
        exclusive = select_joint([rivals], evidence, events, documents)
        self.assertEqual({item["head"] for item in exclusive["confirmed"]}, {"d1:hot", "d1:ship"})
        self.assertEqual(exclusive["competitions"][0]["type"], "MULTIPLE_CONTRIBUTORS")
        self.assertTrue(exclusive["competitions"])
        self.assertIn("h-hot", exclusive["competitions"][0]["hypothesis_ids"])
        self.assertIn("h-ship", exclusive["competitions"][0]["hypothesis_ids"])
        self.assertFalse(exclusive["map_optimization"])
        self.assertEqual(exclusive["procedure"], "逐边联合裁决")

    def test_joint_records_relation_cycle_and_duplicate_reasons(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        nodes = [event("d1:a", "d1", "铁矿石|发运量||当期", "下降", "只是一句发运", indicator="发运量"),
                 event("d2:b", "d2", "铁矿石|到港量||当期", "下降", "只是一句到港", indicator="到港量"),
                 event("d1:c", "d1", "铁水|日均铁水产量||当期", "上升", "只是一句铁水", commodity="铁水", indicator="日均铁水产量"),
                 event("d2:d", "d2", "铁矿石|价格||当期", "上升", "只是一句矿价")]
        evidence = index_evidence(nodes, documents, [])

        def row(head, tail, relation="导致", support=None):
            return {"head": head, "tail": tail, "relation": relation, "relation_class": "ASSERTED_CAUSE",
                    "evidence_ids": support or [], "support_ids": support or [], "counter_evidence_ids": [],
                    "components": {"independent_passages": 1}, "edge_score": 0.8, "reasons": [],
                    "confirmed": True, "why": "", "gap_days": 7}

        def record(rows, hypothesis="h"):
            return {"target": rows[0]["tail"], "candidates": [], "hypotheses": [{
                "id": hypothesis, "kind": "单一直接原因", "why": "", "edge_scores": rows}]}

        conflict = select_joint([
            record([row("d1:a", "d2:b", "导致")], "h-up"),
            record([row("d1:a", "d2:b", "抑制")], "h-down"),
        ], evidence, nodes, documents)
        self.assertTrue(any(item["kind"] == "relation_conflict" for item in conflict["adjudications"]))
        cycle = select_joint([
            record([row("d1:a", "d2:b"), row("d2:b", "d1:c")]),
            record([row("d1:c", "d1:a")]),
        ], evidence, nodes, documents)
        self.assertTrue(any(item["kind"] == "causal_cycle" for item in cycle["adjudications"]))
        shared = "ev:d1:a"
        duplicate = select_joint([
            record([row("d1:a", "d2:b", support=[shared]), row("d1:c", "d2:d", support=[shared])]),
        ], evidence, nodes, documents)
        self.assertTrue(any(item["kind"] == "duplicate_evidence" for item in duplicate["adjudications"]))


class WalkTest(unittest.TestCase):
    def test_time_series_and_possible_causes_do_not_propagate(self):
        self.assertFalse(allows_causal_walk({"type": "事态延续", "change": "延续"}))
        self.assertFalse(allows_causal_walk({"type": "预期验证", "outcome": "兑现"}))
        self.assertFalse(allows_causal_walk({"type": "跨期传导", "relation": "导致", "relation_class": "POSSIBLE_CAUSE"}))
        self.assertFalse(allows_causal_walk({"type": "跨期传导", "relation": "导致", "relation_class": "NEXT_OBSERVATION"}))
        self.assertTrue(allows_causal_walk({"type": "跨期传导", "relation": "导致", "relation_class": "ASSERTED_CAUSE"}))
        self.assertTrue(allows_causal_walk({"type": "导致"}))
        events = [event("d1:a", "d1", "供应|产量||当期", "上升", "供应上升"),
                  event("d1:b", "d1", "到港|到港量||当期", "上升", "到港上升"),
                  event("d2:b", "d2", "到港|到港量||当期", "上升", "到港继续上升"),
                  event("d3:c", "d3", "库存|港口库存||当期", "上升", "库存上升")]
        relations = [{"head": "d1:a", "tail": "d1:b", "type": "导致", "confidence": 0.9}]
        links = [{"head": "d1:b", "tail": "d2:b", "type": "事态延续", "change": "延续"},
                 {"head": "d2:b", "tail": "d3:c", "type": "跨期传导", "relation": "导致", "relation_class": "POSSIBLE_CAUSE", "confidence": 0.4}]
        cross, local = reasoning_chains(events, relations, links)
        self.assertFalse(any("d2:b" in item["nodes"] or "d3:c" in item["nodes"] for item in cross + local))


class ExportTest(unittest.TestCase):
    def test_abductive_output_does_not_replace_graph_json(self):
        folder = Path(tempfile.mkdtemp())
        original = folder / "graph.json"
        original.write_text("OLD", encoding="utf-8")
        result = {
            "relation_method": "abductive", "abduction": {"targets": []},
            "documents": [{"id": "d", "title": "t", "report_date": "2026-04-17", "file_name": "a.pdf",
                           "source_file": "a.pdf", "categories": ["铁矿石"]}],
            "events": [], "relations": [], "variables": {}, "abstracts": {}, "abstract_edges": [],
            "continuity": [], "expectations": [], "cross": [], "observation_links": [], "observation_index": {},
            "reasoning_chains": [], "local_chains": [], "transmission_chains": [], "storylines": [],
        }
        write_all(result, folder)
        self.assertEqual(original.read_text(encoding="utf-8"), "OLD")
        self.assertTrue((folder / "graph_abductive.json").is_file())
        self.assertTrue((folder / "abduction.json").is_file())
        self.assertFalse((Path(__file__).resolve().parents[1] / "example" / "example_data" / "unified10" / "graph.json").exists())


class PromptTest(unittest.TestCase):
    def test_prompt_does_not_treat_a_historical_rule_as_the_current_proof(self):
        documents, events, relations = iron_ore_slice()
        from eventkg.link import abstract_rules
        target = next(item for item in events if item["id"] == "r2:price")
        neighborhood = retrieve_neighborhood(target, events, documents, relations, abstract_rules(events, relations))

        class Recorder:
            def __init__(self):
                self.user = ""
                self.system = ""

            def complete_json(self, system, user, max_tokens=4096, tag=""):
                self.system = system
                self.user = user
                return {"hypotheses": [{"id": "h0", "kind": "证据不足", "why": "只检查输入", "edges": []}]}

        chat = Recorder()
        propose_graphs(chat, target, neighborhood, relations, documents, index_evidence(events, documents, relations))
        self.assertIn("不能当作这次因果的证据", chat.system)
        self.assertNotIn("本次因果已经成立", chat.user)


def _placed(text, specs):
    mentions, cursor = [], 0
    for name, linked in specs:
        start = text.find(name, cursor)
        if start < 0:
            raise AssertionError(name)
        mentions.append({
            "mention_id": "%s@%d" % (linked, start),
            "char_start": start, "char_end": start + len(name),
            "linked_event_id": linked, "alignment_confidence": 0.9,
        })
        cursor = start + len(name)
    return mentions


def _accepted(text, specs):
    pairs = _relation_pairs(text, _placed(text, specs))
    return sorted((pair["cause"]["linked_event_id"], pair["effect"]["linked_event_id"])
                  for pair in pairs if not pair["mood"] and pair.get("role", "direct") == "direct")


def _scored(head, tail, score=0.8):
    return {"head": head, "tail": tail, "relation": "导致", "confirmed": True,
            "relation_class": "ASSERTED_CAUSE", "edge_score": score, "evidence_ids": [],
            "components": {"independent_passages": 1}, "reasons": [], "why": "测试",
            "counter_evidence_ids": []}


def _proposal(target, hypothesis_id, kind, edges, total=1.0):
    return {"target": target, "hypotheses": [{
        "id": hypothesis_id, "kind": kind, "why": "测试", "graph_scores": {"total": total},
        "edge_scores": edges,
    }]}


def _joint_signature(result):
    def rows(name):
        return sorted((item["head"], item["tail"], item["status"], item["relation_class"]) for item in result[name])

    competitions = sorted((item["type"], item["tail"], item["status"], tuple(item.get("hypothesis_keys") or []))
                          for item in result["competitions"])
    return rows("confirmed"), rows("possible"), competitions


class RelationScopeTest(unittest.TestCase):
    def test_comma_joined_clauses_do_not_cross_pair(self):
        text = "发运下降导致到港下降，铁水上升推高矿价上涨"
        specs = [("发运", "A"), ("到港", "B"), ("铁水", "C"), ("矿价", "D")]
        self.assertEqual(_accepted(text, specs), [("A", "B"), ("C", "D")])
        self.assertNotIn(("A", "C"), _accepted(text, specs))
        self.assertNotIn(("B", "D"), _accepted(text, specs))

    def test_comma_before_trigger_still_keeps_the_cause(self):
        text = "发运下降，导致到港下降"
        self.assertEqual(_accepted(text, [("发运", "A"), ("到港", "B")]), [("A", "B")])
        pushed = "成本推升，导致盈利率下降"
        self.assertEqual(_accepted(pushed, [("成本", "A"), ("盈利", "B")]), [("A", "B")])
        background = "成材价格大幅下跌，而部分钢厂切换至高价原料，成本推升，导致盈利率回落"
        self.assertEqual(_accepted(background, [("价格", "价格"), ("成本", "成本"), ("盈利", "盈利")]),
                         [("成本", "盈利")])
        chain = "发运下降导致到港下降，随后到港下降导致库存下降"
        self.assertEqual(_accepted(chain, [("发运", "A"), ("到港", "B"), ("到港", "B2"), ("库存", "C")]),
                         [("A", "B"), ("B2", "C")])

    def test_coordinated_causes_stay_with_one_trigger(self):
        text = "矿价上涨与焦炭上涨共同推高成本"
        self.assertEqual(_accepted(text, [("矿价", "A"), ("焦炭", "B"), ("成本", "C")]),
                         [("A", "C"), ("B", "C")])

    def test_reason_marker_before_the_cause(self):
        text = "由于发运下降，到港下降"
        self.assertEqual(_accepted(text, [("发运", "A"), ("到港", "B")]), [("A", "B")])

    def test_affected_by_marker(self):
        text = "受发运下降影响，到港下降"
        self.assertEqual(_accepted(text, [("发运", "A"), ("到港", "B")]), [("A", "B")])

    def test_intermediate_event_is_a_chain_not_a_shortcut(self):
        text = "受价格上涨影响，钢厂缩减采购，进而导致库存增加"
        specs = [("价格", "价格"), ("采购", "采购"), ("库存", "库存")]
        self.assertEqual(_accepted(text, specs), [("价格", "采购"), ("采购", "库存")])
        self.assertNotIn(("价格", "库存"), _accepted(text, specs))
        hidden = "受价格上涨影响，钢厂缩减采购，进而导致库存增加"
        pairs = _relation_pairs(hidden, _placed(hidden, [("价格", "价格"), ("库存", "库存")]))
        direct = [pair for pair in pairs if pair.get("role") == "direct" and not pair["mood"]]
        indirect = [pair for pair in pairs if pair.get("role") == "indirect"]
        self.assertEqual(direct, [])
        self.assertTrue(indirect)
        self.assertIn("采购", indirect[0]["missing_intermediate"])

    def test_promote_is_a_forward_cause(self):
        text = "发运下降促进到港下降"
        pairs = _relation_pairs(text, _placed(text, [("发运", "A"), ("到港", "B")]))
        self.assertEqual([(pair["cause"]["linked_event_id"], pair["effect"]["linked_event_id"], pair["trigger"])
                          for pair in pairs if not pair["mood"]], [("A", "B", "促进")])

    def test_negation_does_not_cancel_the_next_clause(self):
        text = "发运下降未导致到港下降，但铁水上升导致矿价上涨"
        specs = [("发运", "A"), ("到港", "B"), ("铁水", "C"), ("矿价", "D")]
        self.assertEqual(_accepted(text, specs), [("C", "D")])
        self.assertEqual(_evidence_role(text), "RELATION_SUPPORT")

    def test_one_direction_conflict_does_not_reject_the_other_relation(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-04-24"},
            {"id": "d3", "report_date": "2026-05-15"},
        ]
        quote = "2026-04-17发运下降导致2026-04-24到港下降，2026-04-24发运上升导致2026-05-15到港上升"
        ship_down = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        arrive_down = event("d2:arrive", "d2", "铁矿石|到港量||当期", "下降", quote, indicator="到港量")
        ship_up = event("d2:ship", "d2", "铁矿石|发运量||当期", "上升", "本期发运上升", indicator="发运量")
        arrive_up = event("d3:arrive", "d3", "铁矿石|到港量||当期", "上升", "到港上升", indicator="到港量")
        events = [ship_down, arrive_down, ship_up, arrive_up]
        evidence = index_evidence(events, documents, [])

        def choose(head, tail, target):
            return select_graph(target, [{
                "id": "h", "kind": "单一直接原因", "why": "分句各自写了方向", "edges": [
                    {"head": head, "tail": tail, "relation": "导致", "evidence_ids": ["ev:d2:arrive"], "why": "同段"}],
            }], [{"id": head}], events, documents, evidence)

        down = choose("d1:ship", "d2:arrive", arrive_down)
        up = choose("d2:ship", "d3:arrive", arrive_up)
        crossed = choose("d1:ship", "d3:arrive", arrive_up)
        self.assertEqual(down["edges"][0]["relation_class"], "ASSERTED_CAUSE")
        self.assertEqual(up["edges"][0]["relation_class"], "ASSERTED_CAUSE")
        self.assertFalse(any(edge.get("relation_class") == "ASSERTED_CAUSE" for edge in crossed.get("edges") or []))

    def test_mood_stays_inside_its_own_clause(self):
        forecast = "发运下降预计导致到港下降；铁水上升推高矿价上涨"
        specs = [("发运", "A"), ("到港", "B"), ("铁水", "C"), ("矿价", "D")]
        self.assertEqual(_accepted(forecast, specs), [("C", "D")])
        hypothesis = "如果发运下降导致到港下降。铁水上升推高矿价上涨"
        self.assertEqual(_accepted(hypothesis, specs), [("C", "D")])
        contrast = "发运下降导致到港下降，但是铁水上升未证实推高矿价上涨"
        self.assertEqual(_accepted(contrast, specs), [("A", "B")])
        self.assertEqual(_evidence_role(contrast), "RELATION_SUPPORT")

    def test_negation_and_modal_bind_only_their_proposition(self):
        self.assertEqual(_accepted("发运下降不能导致到港下降", [("发运", "A"), ("到港", "B")]), [])
        self.assertEqual(_accepted("发运下降不一定导致到港下降", [("发运", "A"), ("到港", "B")]), [])
        self.assertEqual(_accepted("发运下降未必导致到港下降", [("发运", "A"), ("到港", "B")]), [])
        self.assertEqual(_accepted("发运下降预计导致到港下降", [("发运", "A"), ("到港", "B")]), [])
        self.assertEqual(_accepted("发运下降可能导致到港下降", [("发运", "A"), ("到港", "B")]), [])
        self.assertEqual(_accepted("成本预期松动造成价格走弱", [("成本", "A"), ("价格", "B")]), [("A", "B")])
        self.assertEqual(_accepted("流动性收紧预期压制价格", [("流动性", "A"), ("价格", "B")]), [("A", "B")])
        self.assertEqual(_accepted("发运下降未导致到港下降，但铁水上升导致矿价上涨",
                                   [("发运", "A"), ("到港", "B"), ("铁水", "C"), ("矿价", "D")]), [("C", "D")])
        split = "发运下降预测到港下降，但铁水上升导致矿价上涨"
        self.assertEqual(_accepted(split, [("发运", "A"), ("到港", "B"), ("铁水", "C"), ("矿价", "D")]), [("C", "D")])


class TimeAndImplicitTest(unittest.TestCase):
    def test_publication_order_does_not_decide_causal_direction(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        cause = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        effect = event("d2:price", "d2", "铁矿石|价格||当期", "上升", "本期矿价上涨")
        evidence = index_evidence([cause, effect], documents, [])
        choice = select_graph(effect, [{
            "id": "h", "kind": "单一直接原因", "why": "报告日期更早", "edges": [
                {"head": "d1:ship", "tail": "d2:price", "relation": "导致",
                 "evidence_ids": ["ev:d1:ship", "ev:d2:price"], "why": "只有发布时间"}],
        }], [{"id": "d1:ship"}], [cause, effect], documents, evidence)
        row = choice["hypotheses"][0]["edge_scores"][0]
        self.assertNotEqual(row["relation_class"], "ASSERTED_CAUSE")
        self.assertNotEqual(row["relation_class"], "INFERRED_SUPPORTED")
        self.assertIn("报告发布时间", "；".join(row["reasons"]))
        neighborhood = retrieve_neighborhood(effect, [cause, effect], documents, [])
        row = neighborhood["candidates"][0]
        self.assertEqual(row["time_basis"], "report_date")
        self.assertEqual(row["time_relation"], "before")
        self.assertIn("不能据此判断因果方向", row["causal_use"])

    def test_unrelated_middle_event_is_not_a_causal_bridge(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-05-01"},
            {"id": "d3", "report_date": "2026-05-15"},
            {"id": "d9", "report_date": "2026-05-20"},
        ]
        ship = event("e1", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        training = event("e2", "d2", "铁矿石|员工培训||当期", "上升", "港区举行员工培训", indicator="员工培训")
        price = event("e3", "d3", "铁矿石|价格||当期", "上升", "铁矿石价格上涨", indicator="价格")
        ship.update(event_time="2026-04-17", subject="澳洲矿山")
        training.update(event_time="2026-05-01", subject="港区")
        price.update(event_time="2026-05-15", subject="铁矿石市场")
        events = [ship, training, price]
        relations = [{"head": "e1", "tail": "e3", "type": "导致", "doc_id": "d9",
                      "evidence": "历史抽象规律认为发运下降导致价格上涨"}]
        evidence = index_evidence(events, documents, relations)
        choice = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "三次观测时间相连", "edges": [{
                "head": "e1", "tail": "e3", "relation": "导致",
                "evidence_ids": ["ev:e1", "ev:e2", "ev:e3", "ev:rel:e1:e3:0"],
                "why": "同属铁矿石"}],
        }], [{"id": "e1"}, {"id": "e2"}], events, documents, evidence)
        self.assertEqual(choice["edges"], [])
        row = choice["hypotheses"][0]["edge_scores"][0]
        self.assertEqual(row["relation_class"], "POSSIBLE_INDIRECT_CAUSE")
        self.assertFalse(row["confirmed"])
        text = "；".join(row["reasons"])
        self.assertIn("缺少桥接证据", text)
        self.assertIn("历史抽象规律不能作为中间关系的证据", text)
        self.assertNotIn(("e1", "e3"), {(edge["head"], edge["tail"]) for edge in choice["edges"]})

    def test_two_evidenced_hops_stay_and_do_not_add_a_direct_edge(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-05-01"},
            {"id": "d3", "report_date": "2026-05-15"},
            {"id": "d4", "report_date": "2026-05-08"},
            {"id": "d5", "report_date": "2026-05-16"},
        ]
        ship = event("e1", "d1", "铁矿石|发运量||当期", "下降", "2026-04-17澳洲发运下降", indicator="发运量")
        stock = event("e2", "d2", "铁矿石|港口库存||当期", "下降", "2026-05-01港口库存下降", indicator="港口库存")
        price = event("e3", "d3", "铁矿石|价格||当期", "上升", "2026-05-15铁矿石价格上涨", indicator="价格")
        ship["event_time"] = "2026-04-17"
        stock["event_time"] = "2026-05-01"
        price["event_time"] = "2026-05-15"
        events = [ship, stock, price]
        relations = [
            {"head": "e1", "tail": "e2", "type": "导致", "doc_id": "d4",
             "evidence": "2026-04-17澳洲发运下降导致2026-05-01港口库存下降"},
            {"head": "e2", "tail": "e3", "type": "导致", "doc_id": "d5",
             "evidence": "2026-05-01港口库存下降导致2026-05-15铁矿石价格上涨"},
        ]
        evidence = index_evidence(events, documents, relations)
        choice = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "两段各自有原文", "edges": [{
                "head": "e1", "tail": "e3", "relation": "导致",
                "evidence_ids": ["ev:e1", "ev:e2", "ev:e3", "ev:rel:e1:e2:0", "ev:rel:e2:e3:1"],
                "why": "不要合成直接边"}],
        }], [{"id": "e1"}, {"id": "e2"}], events, documents, evidence)
        kept = [(edge["head"], edge["tail"]) for edge in choice["edges"]]
        self.assertEqual(kept, [("e1", "e2"), ("e2", "e3")])
        self.assertNotIn(("e1", "e3"), kept)
        self.assertTrue(all(edge["relation_class"] == "INFERRED_SUPPORTED" for edge in choice["edges"]))
        self.assertTrue(all(edge["components"]["inference_path"] == "implicit_segment" for edge in choice["edges"]))
        self.assertTrue(allows_causal_walk(choice["edges"][0]))
        self.assertFalse(allows_causal_walk({"relation_class": "POSSIBLE_INDIRECT_CAUSE", "type": "跨期传导"}))

    def test_time_order_alone_does_not_confirm_a_cause(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-05-01"},
            {"id": "d3", "report_date": "2026-05-15"},
        ]
        ship = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        freight = event("d2:freight", "d2", "铁矿石|海运费||当期", "上升", "海运费上涨", indicator="海运费")
        price = event("d3:price", "d3", "铁矿石|价格||当期", "上升", "矿价上涨", indicator="价格")
        ship["event_time"] = "2026-04-17"
        freight["event_time"] = "2026-05-01"
        price["event_time"] = "2026-05-15"
        events = [ship, freight, price]
        evidence = index_evidence(events, documents, [])
        choice = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "只有时间先后", "edges": [{
                "head": "d1:ship", "tail": "d3:price", "relation": "导致",
                "evidence_ids": ["ev:d1:ship", "ev:d2:freight", "ev:d3:price"], "why": "中间只有海运费"}],
        }], [{"id": "d1:ship"}, {"id": "d2:freight"}], events, documents, evidence)
        self.assertEqual(choice["edges"], [])
        row = choice["hypotheses"][0]["edge_scores"][0]
        self.assertEqual(row["relation_class"], "POSSIBLE_INDIRECT_CAUSE")
        self.assertIn("缺少桥接证据", "；".join(row["reasons"]))

    def test_mixed_polarity_hops_keep_their_own_relations(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-05-01"},
            {"id": "d3", "report_date": "2026-05-15"},
            {"id": "d4", "report_date": "2026-05-08"},
            {"id": "d5", "report_date": "2026-05-16"},
        ]
        ship = event("e1", "d1", "铁矿石|发运量||当期", "下降", "2026-04-17澳洲发运下降", indicator="发运量")
        stock = event("e2", "d2", "铁矿石|港口库存||当期", "下降", "2026-05-01港口库存下降", indicator="港口库存")
        price = event("e3", "d3", "铁矿石|价格||当期", "上升", "2026-05-15铁矿石价格上涨", indicator="价格")
        ship["event_time"] = "2026-04-17"
        stock["event_time"] = "2026-05-01"
        price["event_time"] = "2026-05-15"
        events = [ship, stock, price]
        relations = [
            {"head": "e1", "tail": "e2", "type": "导致", "doc_id": "d4",
             "evidence": "2026-04-17澳洲发运下降导致2026-05-01港口库存下降"},
            {"head": "e2", "tail": "e3", "type": "抑制", "doc_id": "d5",
             "evidence": "2026-05-01港口库存下降抑制2026-05-15铁矿石价格上涨"},
        ]
        evidence = index_evidence(events, documents, relations)
        choice = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "后一段是抑制", "edges": [{
                "head": "e1", "tail": "e3", "relation": "导致",
                "evidence_ids": ["ev:e1", "ev:e2", "ev:e3", "ev:rel:e1:e2:0", "ev:rel:e2:e3:1"],
                "why": "原始边写成导致"}],
        }], [{"id": "e1"}, {"id": "e2"}], events, documents, evidence)
        kept = [(edge["head"], edge["relation"], edge["tail"]) for edge in choice["edges"]]
        self.assertEqual(kept, [("e1", "导致", "e2"), ("e2", "抑制", "e3")])
        self.assertNotIn(("e1", "e3"), {(edge["head"], edge["tail"]) for edge in choice["edges"]})
        segments = choice["edges"][1]["components"]["segment"]
        self.assertEqual(segments["relation"], "抑制")
        self.assertTrue(segments["relation_support"])
        self.assertTrue(segments["temporal_compatibility"])

    def test_mention_time_matches_on_direct_and_indirect_paths(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-05-01"},
            {"id": "d3", "report_date": "2026-05-15"},
            {"id": "d4", "report_date": "2026-05-08"},
            {"id": "d5", "report_date": "2026-05-16"},
        ]
        ship = event("e1", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        stock = event("e2", "d2", "铁矿石|港口库存||当期", "下降", "港口库存下降", indicator="港口库存")
        price = event("e3", "d3", "铁矿石|价格||当期", "上升", "铁矿石价格上涨", indicator="价格")
        ship["evidences"] = ["2026-04-17澳洲发运下降。"]
        stock["evidences"] = ["2026-05-01港口库存下降。"]
        price["evidences"] = ["2026-05-15铁矿石价格上涨。"]
        events = [ship, stock, price]
        relations = [
            {"head": "e1", "tail": "e2", "type": "导致", "doc_id": "d4",
             "evidence": "2026-04-17澳洲发运下降导致2026-05-01港口库存下降"},
            {"head": "e2", "tail": "e3", "type": "导致", "doc_id": "d5",
             "evidence": "2026-05-01港口库存下降导致2026-05-15铁矿石价格上涨"},
        ]
        evidence = index_evidence(events, documents, relations)
        direct = select_graph(stock, [{
            "id": "h", "kind": "单一直接原因", "why": "提及里有时间", "edges": [{
                "head": "e1", "tail": "e2", "relation": "导致",
                "evidence_ids": ["ev:rel:e1:e2:0", "ev:e1:1"], "why": "直接边"}],
        }], [{"id": "e1"}], events, documents, evidence)
        indirect = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "同一批提及时间", "edges": [{
                "head": "e1", "tail": "e3", "relation": "导致",
                "evidence_ids": ["ev:e1:1", "ev:e2:1", "ev:e3:1", "ev:rel:e1:e2:0", "ev:rel:e2:e3:1"],
                "why": "间接边"}],
        }], [{"id": "e1"}, {"id": "e2"}], events, documents, evidence)
        direct_row = direct["hypotheses"][0]["edge_scores"][0]
        hop = next(edge for edge in indirect["edges"] if edge["head"] == "e1" and edge["tail"] == "e2")
        self.assertEqual(direct_row["components"]["time_basis"], "event_time")
        self.assertEqual(hop["components"]["time_basis"], "event_time")
        self.assertEqual(direct_row["gap_days"], hop["gap_days"])
        self.assertNotIn("报告发布时间", "；".join(direct_row["reasons"]))
        self.assertNotIn("报告发布时间", hop.get("reason") or "")
        catalog = {item["evidence_id"]: item for item in evidence}
        self.assertEqual(resolve_event_time(ship, catalog).isoformat(), "2026-04-17")
        self.assertEqual(resolve_event_time(stock, catalog).isoformat(), "2026-05-01")
        self.assertFalse(ship.get("event_time"))

    def test_default_search_does_not_expand_past_one_middle(self):
        documents = [
            {"id": "d1", "report_date": "2026-04-17"},
            {"id": "d2", "report_date": "2026-04-24"},
            {"id": "d3", "report_date": "2026-05-01"},
            {"id": "d4", "report_date": "2026-05-15"},
            {"id": "d5", "report_date": "2026-05-08"},
            {"id": "d6", "report_date": "2026-05-10"},
            {"id": "d7", "report_date": "2026-05-18"},
        ]
        ship = event("e1", "d1", "铁矿石|发运量||当期", "下降", "2026-04-17澳洲发运下降", indicator="发运量")
        arrive = event("m1", "d2", "铁矿石|到港量||当期", "下降", "2026-04-24到港下降", indicator="到港量")
        stock = event("m2", "d3", "铁矿石|港口库存||当期", "下降", "2026-05-01港口库存下降", indicator="港口库存")
        price = event("e4", "d4", "铁矿石|价格||当期", "上升", "2026-05-15铁矿石价格上涨", indicator="价格")
        for item, when in ((ship, "2026-04-17"), (arrive, "2026-04-24"), (stock, "2026-05-01"), (price, "2026-05-15")):
            item["event_time"] = when
        events = [ship, arrive, stock, price]
        relations = [
            {"head": "e1", "tail": "m1", "type": "导致", "doc_id": "d5",
             "evidence": "2026-04-17澳洲发运下降导致2026-04-24到港下降"},
            {"head": "m1", "tail": "m2", "type": "抑制", "doc_id": "d6",
             "evidence": "2026-04-24到港下降抑制2026-05-01港口库存下降"},
            {"head": "m2", "tail": "e4", "type": "导致", "doc_id": "d7",
             "evidence": "2026-05-01港口库存下降导致2026-05-15铁矿石价格上涨"},
        ]
        evidence = index_evidence(events, documents, relations)
        attach_mentions(evidence, events, documents)
        catalog = {item["evidence_id"]: item for item in evidence}
        cited = [catalog[item] for item in (
            "ev:e1", "ev:m1", "ev:m2", "ev:e4", "ev:rel:e1:m1:0", "ev:rel:m1:m2:1", "ev:rel:m2:e4:2")]
        self.assertEqual(find_indirect_paths(cited, ship, price, events, catalog), [])
        longer = find_indirect_paths(cited, ship, price, events, catalog, max_intermediate_nodes=2)
        self.assertEqual(len(longer), 1)
        self.assertEqual(longer[0]["middles"], ["m1", "m2"])
        self.assertEqual([segment["relation"] for segment in longer[0]["segments"]], ["导致", "抑制", "导致"])
        choice = select_graph(price, [{
            "id": "h", "kind": "间接影响", "why": "默认只看一个中间事件", "edges": [{
                "head": "e1", "tail": "e4", "relation": "导致",
                "evidence_ids": [item["evidence_id"] for item in cited], "why": "三段"}],
        }], [{"id": "e1"}, {"id": "m1"}, {"id": "m2"}], events, documents, evidence)
        self.assertEqual(choice["edges"], [])
        self.assertNotIn(("e1", "e4"), {(edge["head"], edge["tail"]) for edge in choice["edges"]})


class HypothesisIdentityTest(unittest.TestCase):
    def test_same_hypothesis_id_on_two_targets_is_not_one_hypothesis(self):
        multi = _proposal("target-cost", "h1", "多个共同原因",
                          [_scored("cause-a", "result"), _scored("cause-b", "result")], total=2)
        single = _proposal("target-price", "h1", "单一直接原因",
                           [_scored("cause-c", "result", 0.99)], total=9)
        result = select_joint([multi, single], [], [], [])
        confirmed = {item["head"] for item in result["confirmed"]}
        self.assertEqual(confirmed, {"cause-a", "cause-b", "cause-c"})
        compatible = [item for item in result["competitions"] if item["type"] == "MULTIPLE_CONTRIBUTORS"]
        self.assertIn("target-cost/h1", compatible[0]["hypothesis_keys"])
        self.assertIn("target-price/h1", compatible[0]["hypothesis_keys"])
        self.assertNotEqual(compatible[0]["hypothesis_keys"], ["h1"])
        keys = {source["hypothesis_key"] for edge in result["edges"] for source in edge["sources"]}
        self.assertEqual(keys, {"target-cost/h1", "target-price/h1"})

    def test_single_cause_is_not_absorbed_by_another_multi_cause_hypothesis(self):
        multi = _proposal("target-cost", "h-both", "多个共同原因",
                          [_scored("cause-a", "result"), _scored("cause-b", "result")])
        single = _proposal("target-cost", "h-only", "单一直接原因", [_scored("cause-c", "result", 0.99)])
        result = select_joint([multi, single], [], [], [])
        self.assertEqual({item["head"] for item in result["confirmed"]}, {"cause-a", "cause-b", "cause-c"})
        compatible = [item for item in result["competitions"] if item["status"] == "compatible"]
        self.assertEqual(compatible[0]["heads"], ["cause-a", "cause-b", "cause-c"])
        only = [edge for edge in result["edges"] if edge["head"] == "cause-c"][0]
        self.assertEqual(only["sources"][0]["hypothesis"], "h-only")

    def test_adjudication_does_not_depend_on_proposal_order(self):
        multi = _proposal("target-cost", "h1", "多个共同原因",
                          [_scored("cause-a", "result", 0.7), _scored("cause-b", "result", 0.6)])
        single = _proposal("target-price", "h1", "单一直接原因", [_scored("cause-c", "result", 0.95)])
        rivals = [
            _proposal("target-a", "h1", "单一直接原因", [_scored("left", "other", 0.5)]),
            _proposal("target-b", "h1", "单一直接原因", [_scored("right", "other", 0.5)]),
        ]
        forward = select_joint([multi, single] + rivals, [], [], [])
        backward = select_joint(list(reversed(rivals + [single, multi])), [], [], [])
        swapped = _proposal("target-cost", "h1", "多个共同原因",
                            [_scored("cause-b", "result", 0.6), _scored("cause-a", "result", 0.7)])
        reordered = select_joint([rivals[1], swapped, rivals[0], single], [], [], [])
        self.assertEqual(_joint_signature(forward), _joint_signature(backward))
        self.assertEqual(_joint_signature(forward), _joint_signature(reordered))
        self.assertEqual({item["head"] for item in forward["confirmed"]},
                         {"cause-a", "cause-b", "cause-c", "left", "right"})


class ReportSeriesTest(unittest.TestCase):
    def test_unknown_series_does_not_borrow_another_report(self):
        documents = [{"id": "d1", "report_date": "2026-04-17"}, {"id": "d2", "report_date": "2026-04-24"}]
        previous = event("d1:ship", "d1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        current = event("d2:ship", "d2", "铁矿石|发运量||当期", "下降", "上期发运下降", indicator="发运量")
        catalog = index_evidence([previous, current], documents, [])
        attach_mentions(catalog, [previous, current], documents)
        mention = [item for item in catalog if item["evidence_id"] == "ev:d2:ship"][0]["mentions"][0]
        self.assertEqual(mention["report_series_id"], "")
        self.assertEqual(mention["linked_event_id"], "")
        self.assertEqual(mention["event_time"], "")

    def test_relative_period_stays_inside_one_report_series(self):
        documents = [
            {"id": "iron1", "report_date": "2026-04-17", "title": "Mysteel铁矿石周刊20260417"},
            {"id": "pvc", "report_date": "2026-04-20", "title": "隆众资讯PVC日报20260420"},
            {"id": "iron2", "report_date": "2026-04-24", "title": "Mysteel铁矿石周刊20260424"},
        ]
        self.assertEqual(report_series_id(documents[0]), "mysteel:铁矿石:weekly")
        self.assertEqual(report_series_id(documents[1]), "longzhong:PVC:daily")
        self.assertNotEqual(report_series_id(documents[0]), report_series_id(documents[1]))
        iron_prev = event("iron1:ship", "iron1", "铁矿石|发运量||当期", "下降", "澳洲发运下降", indicator="发运量")
        pvc_ship = event("pvc:ship", "pvc", "PVC|发运量||当期", "下降", "PVC发运下降", commodity="PVC", indicator="发运量")
        iron_now = event("iron2:ship", "iron2", "铁矿石|发运量||当期", "下降", "上期发运下降", indicator="发运量")
        catalog = index_evidence([iron_prev, pvc_ship, iron_now], documents, [])
        attach_mentions(catalog, [iron_prev, pvc_ship, iron_now], documents)
        mention = [item for item in catalog if item["evidence_id"] == "ev:iron2:ship"][0]["mentions"][0]
        self.assertEqual(mention["linked_event_id"], "iron1:ship")
        self.assertNotEqual(mention["linked_event_id"], "pvc:ship")
        self.assertEqual(mention["event_time"], "")
        self.assertEqual(mention["report_date"], "2026-04-24")

    def test_missing_weekly_issue_does_not_become_the_next_period(self):
        documents = [
            {"id": "w0612", "report_date": "2026-06-12", "title": "Mysteel铁矿石周刊20260612"},
            {"id": "w0626", "report_date": "2026-06-26", "title": "Mysteel铁矿石周刊20260626"},
        ]
        current = event("w0612:ship", "w0612", "铁矿石|发运量||当期", "上升", "下期发运增加", indicator="发运量")
        later = event("w0626:ship", "w0626", "铁矿石|发运量||当期", "上升", "发运增加", indicator="发运量")
        catalog = index_evidence([current, later], documents, [])
        attach_mentions(catalog, [current, later], documents)
        mention = [item for item in catalog if item["evidence_id"] == "ev:w0612:ship"][0]["mentions"][0]
        self.assertEqual(mention["linked_event_id"], "")
        self.assertEqual(mention["event_time"], "")
        skipped = event("w0612:skip", "w0612", "铁矿石|发运量||当期", "上升", "因假期顺延，下期发运增加", indicator="发运量")
        catalog = index_evidence([skipped, later], documents, [])
        attach_mentions(catalog, [skipped, later], documents)
        bound = [item for item in catalog if item["evidence_id"] == "ev:w0612:skip"][0]["mentions"][0]
        self.assertEqual(bound["linked_event_id"], "w0626:ship")


if __name__ == "__main__":
    unittest.main()
