# -*- coding: utf-8 -*-
import unittest

from eventkg.graph_induction import _cause_compatibility, _pair_issues, _resolve_cause_competition, _score_edge
from eventkg.link import attach_observations
from eventkg.observations import parse_signed_number, parse_table_measures
from eventkg.paths import explain_paths
from eventkg.retrieval import VectorCache, event_signature, rank_events
from eventkg.retrieve import citation_bundle, search
from eventkg.roles import assign_roles


PVC_ROW = (
    "[表格：装置检修]"
    "原表第4行：列1[指标（每周五更新）]=下游开工（%）；列2[2026/9/30]=40.22%；"
    "列3[2026/10/8]=22.29%；列4[列4（表头未确认）]=（空白）；列5[涨跌]=-17.93%；列6[环比]=-44.58%"
)


def event(event_id, doc, var, direction, indicator):
    return {"id": event_id, "doc_id": doc, "var": var, "dir": direction, "status": "已发生",
            "name": indicator + direction, "evidence": indicator + direction, "salience": "核心",
            "commodity": var.split("|")[0], "indicator": indicator, "scope": "", "subject": "",
            "sections": ["s"], "event_instance_id": event_id, "event_time": ""}


class RoleTest(unittest.TestCase):
    def test_maintenance_roles_come_only_from_known_fields(self):
        item = assign_roles({
            "id": "e1", "name": "装置检修", "indicator": "检修", "subject": "某检修企业",
            "commodity": "PVC", "scope": "华东", "event_time": "2026-09-30",
            "roles": {"facility": "某装置", "status": "进行中"},
            "evidence": "某检修企业对某装置进行检修",
        })
        self.assertEqual(item["roles"]["agent"], "某检修企业")
        self.assertEqual(item["roles"]["facility"], "某装置")
        self.assertEqual(item["roles"]["product"], "PVC")
        self.assertEqual(item["roles"]["location"], "华东")
        self.assertEqual(item["roles"]["start_time"], "2026-09-30")
        self.assertEqual(item["roles"]["status"], "进行中")
        self.assertNotIn("expected_recovery_time", item["roles"])
        self.assertNotIn("actual_recovery_time", item["roles"])
        self.assertIn("expected_recovery_time", item["missing_roles"])
        agent = [row for row in item["role_bindings"] if row["role"] == "agent"][0]
        self.assertEqual(agent["name"], "某检修企业")
        self.assertTrue(agent["entity_id"].startswith("ent_"))
        self.assertNotEqual(agent["entity_id"], "agent")

    def test_policy_roles_keep_the_object_and_effective_time(self):
        item = assign_roles({
            "event_kind": "政策调整", "subject": "某部委", "scope": "华东",
            "event_time": "2026-10-01", "commodity": "PVC",
            "roles": {"policy_measure": "出口退税调整", "affected_entity": "PVC企业", "modality": "已发布"},
            "evidence": "某部委发布出口退税调整",
        })
        self.assertEqual(item["roles"]["publisher"], "某部委")
        self.assertEqual(item["roles"]["affected_region"], "华东")
        self.assertEqual(item["roles"]["affected_entity"], "PVC企业")
        self.assertEqual(item["roles"]["effective_time"], "2026-10-01")
        self.assertEqual(item["roles"]["modality"], "已发布")
        self.assertEqual(item["evidence"], "某部委发布出口退税调整")
        self.assertEqual(item["event_kind"], "POLICY_CHANGE")
        plain = assign_roles({"event_kind": "POLICY_CHANGE", "commodity": "PVC", "subject": "某部委",
                              "roles": {}, "evidence": "某部委调整政策"})
        self.assertNotIn("affected_entity", plain["roles"])
        arrival = assign_roles({"name": "本期铁矿石到港量上升", "indicator": "到港量", "commodity": "铁矿石", "dir": "上升"})
        self.assertEqual(arrival["event_kind"], "LOGISTICS_CHANGE")
        self.assertNotIn("affected_cargo", arrival["roles"])
        stopped = assign_roles({"name": "港口封航", "indicator": "封航"})
        self.assertEqual(stopped["event_kind"], "LOGISTICS_DISRUPTION")


class ObservationTest(unittest.TestCase):
    def test_pvc_numbers_follow_column_headers(self):
        measures = parse_table_measures(PVC_ROW, document_id="doc", source_table_id="doc:obs:1")
        by_header = {item["column_header"]: item for item in measures}
        earlier = by_header["2026/9/30"]
        later = by_header["2026/10/8"]
        ratio = by_header["环比"]
        delta = by_header["涨跌"]
        self.assertEqual(earlier["value_raw"], "40.22%")
        self.assertEqual(earlier["value_numeric"], 40.22)
        self.assertEqual(earlier["observation_period"], "2026-09-30")
        self.assertIsNone(earlier["change_numeric"])
        self.assertEqual(later["value_raw"], "22.29%")
        self.assertEqual(later["observation_period"], "2026-10-08")
        self.assertEqual(ratio["value_raw"], "-44.58%")
        self.assertEqual(ratio["change_numeric"], -44.58)
        self.assertEqual(ratio["observation_period"], "2026-10-08")
        self.assertEqual(ratio["comparison_period"], "2026-09-30")
        self.assertEqual(ratio["direction"], "下降")
        self.assertNotEqual(ratio["measure_kind"], "level")
        self.assertEqual(delta["value_raw"], "-17.93%")
        self.assertNotEqual(delta["value_raw"], ratio["value_raw"])
        self.assertEqual(parse_signed_number("-44.58%")["value_raw"], "-44.58%")

    def test_numbers_without_headers_stay_ambiguous(self):
        measures = parse_table_measures("40.22%、22.29%、-44.58%", document_id="doc", source_table_id="loose")
        self.assertEqual(len(measures), 1)
        self.assertEqual(measures[0]["status"], "AMBIGUOUS")
        self.assertEqual(measures[0]["observation_period"], "")
        self.assertIsNone(measures[0]["change_numeric"])
        documents = [{"id": "doc", "observations": [{"id": "loose", "text": "40.22%、22.29%、-44.58%", "page": 1, "section": ""}]}]
        events = [{"id": "e", "doc_id": "doc", "indicator": "下游开工", "scope": "", "dir": "下降"}]
        observations, links = attach_observations(events, documents)
        self.assertEqual(links, [])
        self.assertEqual(observations[0]["status"], "AMBIGUOUS")

    def test_reversed_date_columns_follow_the_arithmetic(self):
        text = ("原表第1行：列1[指标]=Mysteel62%澳粉指数（美元/干吨）；列2[2026/4/17]=108.10；"
                "列3[2026/4/10]=106.35；列4[环比]=+1.75")
        measures = parse_table_measures(text, document_id="doc", source_table_id="prices")
        by_header = {item["column_header"]: item for item in measures}
        self.assertEqual(by_header["2026/4/17"]["indicator_id"], "Mysteel62%澳粉指数（美元/干吨）")
        self.assertEqual(by_header["2026/4/17"]["value_numeric"], 108.10)
        self.assertNotEqual(by_header["2026/4/17"]["unit"], "%")
        change = by_header["环比"]
        self.assertEqual(change["observation_period"], "2026-04-17")
        self.assertEqual(change["comparison_period"], "2026-04-10")
        self.assertEqual(change["change_unit"], "美元/干吨")
        self.assertIsNone(parse_signed_number("47港库存"))
        stock = parse_table_measures(
            "原表第2行：列1[指标]=47港库存；列2[2026/4/17]=100；列3[2026/4/10]=90；列4[环比]=+10",
            document_id="doc", source_table_id="stock")
        self.assertTrue(all(item["indicator_id"] == "47港库存" for item in stock if item["status"] == "confirmed"))
        pvc = {item["column_header"]: item for item in parse_table_measures(PVC_ROW, source_table_id="pvc")}
        self.assertEqual(pvc["涨跌"]["change_unit"], "百分点")
        self.assertEqual(pvc["环比"]["change_unit"], "%")
        self.assertEqual(pvc["环比"]["observation_period"], "2026-10-08")

    def test_observation_link_requires_period_scope_and_direction(self):
        text = ("原表第1行：列1[指标]=到港量；列2[2026/4/17]=80；列3[2026/4/10]=100；列4[环比]=-20%")
        documents = [{"id": "doc", "observations": [{"id": "t1", "text": text, "page": 1, "section": "数据"}]}]
        down = {"id": "down", "doc_id": "doc", "indicator": "到港量", "scope": "", "dir": "下降",
                "event_time": "2026-04-17"}
        up = {"id": "up", "doc_id": "doc", "indicator": "到港量", "scope": "", "dir": "上升",
              "event_time": "2026-04-17"}
        earlier = {"id": "earlier", "doc_id": "doc", "indicator": "到港量", "scope": "", "dir": "下降",
                   "event_time": "2026-04-10"}
        _, links = attach_observations([down, up, earlier], documents)
        self.assertEqual({item["head"] for item in links}, {"down"})


class CauseAndRegionTest(unittest.TestCase):
    def test_separate_supported_causes_can_coexist(self):
        left = {"head": "arrival", "tail": "stock", "relation": "导致", "confirmed": True,
                "evidence_ids": ["EV-A"], "support_ids": ["EV-A"], "counter_evidence_ids": [],
                "components": {"independent_passages": 1}, "edge_score": 0.8, "reasons": [],
                "why": "低到港导致库存下降", "sources": [{"target": "stock", "hypothesis": "h-a", "kind": "单一直接原因"}]}
        right = {"head": "clearance", "tail": "stock", "relation": "导致", "confirmed": True,
                 "evidence_ids": ["EV-B"], "support_ids": ["EV-B"], "counter_evidence_ids": [],
                 "components": {"independent_passages": 1}, "edge_score": 0.7, "reasons": [],
                 "why": "铁水复产推动疏港增加", "sources": [{"target": "stock", "hypothesis": "h-b", "kind": "单一直接原因"}]}
        kept, competitions = _resolve_cause_competition([left, right])
        self.assertEqual({item["head"] for item in kept}, {"arrival", "clearance"})
        self.assertEqual(competitions[0]["type"], "MULTIPLE_CONTRIBUTORS")
        blocked = dict(right, why="不能同时成立")
        dropped, exclusive = _resolve_cause_competition([left, blocked])
        self.assertEqual(dropped, [])
        self.assertEqual(exclusive[0]["type"], "MUTUALLY_EXCLUSIVE")
        self.assertEqual(_cause_compatibility([dict(left, evidence_ids=[], support_ids=[], components={})]),
                         "UNKNOWN_COMPATIBILITY")
        countered = dict(right, counter_evidence_ids=["EV-against"])
        kept_one, competitions = _resolve_cause_competition([left, countered])
        self.assertEqual([item["head"] for item in kept_one], ["arrival"])
        self.assertNotEqual(competitions[0]["type"], "MUTUALLY_EXCLUSIVE")
        self.assertEqual(competitions[0]["compatibility_status"], "NOT_GROUP_EXCLUSION")
        self.assertEqual(countered["edge_support_status"], "contradicted")
        self.assertEqual(left["final_decision"], "confirmed")

    def test_transport_may_cross_regions_and_a_bare_region_gap_may_not(self):
        ship = {"id": "ship", "doc_id": "d1", "commodity": "铁矿石", "indicator": "发运量", "name": "澳洲发运下降",
                "var": "铁矿石|发运量|澳洲|当期", "scope": "澳洲", "subject": "", "dir": "下降",
                "roles": {"origin_region": "澳洲", "destination_region": "中国"}}
        arrival = {"id": "arrival", "doc_id": "d2", "commodity": "铁矿石", "indicator": "到港量", "name": "中国到港下降",
                   "var": "铁矿石|到港量|中国|当期", "scope": "中国", "subject": "", "dir": "下降",
                   "roles": {"destination_region": "中国"}}
        price = {"id": "price", "doc_id": "d2", "commodity": "铁矿石", "indicator": "价格", "name": "中国价格下降",
                 "var": "铁矿石|价格|中国|当期", "scope": "中国", "subject": "", "dir": "下降", "roles": {}}
        australia_price = dict(price, id="au", scope="澳洲", var="铁矿石|价格|澳洲|当期", name="澳洲价格下降")
        self.assertNotIn("不同地区且没有运输机制", _pair_issues(ship, arrival))
        self.assertIn("不同地区且没有运输机制", _pair_issues(australia_price, price))
        documents = [{"id": "d1", "report_date": "2026-04-10"}, {"id": "d2", "report_date": "2026-04-17"}]
        evidence = [{"evidence_id": "EV-route", "text": "澳洲发运下降，随后中国到港下降", "document_id": "d1"}]
        by_evidence = {item["evidence_id"]: item for item in evidence}
        by_id = {item["id"]: item for item in (ship, arrival, australia_price, price)}
        allowed = set(by_id)
        dates = {"d1": __import__("datetime").date(2026, 4, 10), "d2": __import__("datetime").date(2026, 4, 17)}
        linked = _score_edge({"head": "ship", "tail": "arrival", "relation": "导致",
                              "evidence_ids": ["EV-route"], "why": "运输"},
                             {"id": "arrival"}, allowed, by_id, by_evidence, dates, list(by_id.values()))
        blocked = _score_edge({"head": "au", "tail": "price", "relation": "导致",
                               "evidence_ids": ["EV-route"], "why": "两地价格"},
                              {"id": "price"}, allowed, by_id, by_evidence, dates, list(by_id.values()))
        self.assertEqual(linked["components"]["entity"], 1)
        self.assertIn("运输关系支持跨地区联系", "；".join(linked["reasons"]))
        self.assertEqual(blocked["components"]["entity"], 0)
        self.assertIn("不同地区且没有运输机制", blocked["reasons"])


class PathAndAnswerTest(unittest.TestCase):
    def test_inhibition_and_repeated_indicator_stay_on_typed_paths(self):
        events = [event("hot", "d1", "铁水|日均铁水产量||当期", "上升", "日均铁水产量"),
                  event("stock", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存"),
                  event("price", "d1", "铁矿石|价格|中国|当期", "上升", "价格")]
        relations = [
            {"head": "hot", "tail": "stock", "type": "抑制", "relation": "抑制", "evidence_ids": ["EV-inh"]},
            {"head": "stock", "tail": "price", "type": "导致", "relation": "导致", "evidence_ids": ["EV-price"]},
        ]
        causal = explain_paths(events, relations)["PURE_CAUSAL"]
        bare = [item for item in causal if item["nodes"] == ("hot", "stock", "price")]
        self.assertTrue(bare)
        self.assertFalse(bare[0]["confirmed"])
        self.assertIn("MISSING", bare[0]["evidence_status"])
        verified = explain_paths(events, [
            dict(relations[0], relation_class="ASSERTED_CAUSE"),
            dict(relations[1], relation_class="ASSERTED_CAUSE"),
        ], evidence_index={"EV-inh": {"text": "铁水上升抑制库存"}, "EV-price": {"text": "库存下降导致价格上升"}})
        self.assertTrue(any(item["nodes"] == ("hot", "stock", "price") and item["confirmed"]
                            for item in verified["PURE_CAUSAL"]))
        later = event("stock-2", "d2", "铁矿石|港口库存|中国|当期", "上升", "港口库存")
        later["event_time"] = "2026-04-24"
        events[1]["event_time"] = "2026-04-10"
        repeated = explain_paths(events + [later], relations + [
            {"head": "price", "tail": "stock-2", "type": "导致", "relation": "导致", "evidence_ids": ["EV-later"]},
        ])
        self.assertTrue(any("stock" in item["nodes"] and "stock-2" in item["nodes"]
                            for item in repeated["PURE_CAUSAL"]))
        part = event("part", "d1", "PVC|电石法||当期", "上升", "电石法")
        whole = event("whole", "d1", "PVC|产量||当期", "上升", "产量")
        composed = explain_paths([part, whole], [{"head": "part", "tail": "whole", "type": "构成", "evidence_ids": ["EV-part"]}])
        self.assertEqual(composed["PURE_CAUSAL"], [])
        self.assertTrue(composed["COMPOSITION"])
        self.assertEqual(composed["COMPOSITION"][0]["path_type"], "COMPOSITION")
        events[1].pop("event_time", None)
        weak = explain_paths(events, [
            {"head": "hot", "tail": "stock", "type": "导致", "relation_class": "ASSERTED_CAUSE",
             "evidence_ids": ["EV-strong"]},
            {"head": "stock", "tail": "price", "type": "导致", "evidence_ids": ["EV_DOES_NOT_EXIST"]},
            {"head": "hot", "tail": "price", "type": "导致"},
        ], evidence_index={"EV-strong": {"text": "铁水导致库存下降"}})
        self.assertEqual(weak["PURE_CAUSAL"][0]["nodes"], ("hot", "stock"))
        self.assertTrue(weak["PURE_CAUSAL"][0]["confirmed"])
        self.assertTrue(all(not item["confirmed"] for item in weak["PURE_CAUSAL"] if "EV_DOES_NOT_EXIST" in str(item["edges"])))
        temporal = explain_paths(
            [event("A", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存"),
             event("B", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存")],
            [], [{"head": "A", "tail": "B", "type": "事态延续", "change": "延续"}])
        self.assertTrue(any(item["nodes"] == ("A", "B") for item in temporal["TEMPORAL"]))
        self.assertEqual(temporal["PURE_CAUSAL"], [])

    def test_answer_keeps_each_relation_evidence_separate(self):
        cause = event("E1", "d1", "铁矿石|到港量|中国|当期", "下降", "到港量")
        effect = event("E2", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存")
        cause["evidence"] = "到港量下降"
        effect["evidence"] = "港口库存下降"
        graph = {"events": [cause, effect],
                 "documents": [{"id": "d1", "report_date": "2026-04-17"}],
                 "relations": [{"head": "E1", "tail": "E2", "type": "导致", "relation": "导致",
                                "relation_class": "ASSERTED_CAUSE", "evidence": "到港下降导致库存下降",
                                "evidence_ids": ["EV12"], "head_fact_ids": ["EV10"], "tail_fact_ids": ["EV15"]}],
                 "evidence": [{"evidence_id": "EV12", "text": "到港下降导致库存下降"}],
                 "continuity": [], "expectations": [], "cross": [], "abstract_edges": []}
        found = search(graph, "港口库存为什么下降")
        bundle = found["citation_bundles"][0]
        self.assertEqual(bundle["relation_evidence_ids"], ["EV12"])
        self.assertEqual(bundle["head_fact_ids"], ["EV10"])
        self.assertEqual(bundle["tail_fact_ids"], ["EV15"])
        self.assertNotEqual(bundle["relation_evidence_ids"], bundle["head_fact_ids"] + bundle["tail_fact_ids"])
        self.assertTrue(found["confirmed_causal"])
        graph["relations"] = [{"head": "E1", "tail": "E2", "type": "导致", "relation": "导致",
                               "relation_class": "ASSERTED_CAUSE", "evidence_ids": ["EV_DOES_NOT_EXIST"]}]
        missing = search(graph, "港口库存为什么下降")
        self.assertFalse(missing["confirmed_causal"])
        self.assertEqual(missing["uncertain_steps"][0]["evidence_status"], "MISSING")
        self.assertFalse(missing["uncertain_steps"][0]["certain"])
        self.assertIn("证据不足", " ".join(missing["paths"]))
        self.assertEqual(citation_bundle(graph["relations"][0], {item["id"]: item for item in graph["events"]})["certain"], False)
        stock = event("stock", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存")
        asked = search({"events": [stock], "documents": [{"id": "d1", "report_date": "2026-04-17"}],
                        "relations": [], "continuity": [], "expectations": [], "cross": [], "abstract_edges": []},
                       "港存为什么去化")
        self.assertTrue(asked["seeds"])

    def test_dense_encoder_cache_is_reused_across_targets(self):
        class Encoder:
            def __init__(self):
                self.seen = []

            def encode(self, texts):
                self.seen.extend(texts)
                return [[float(len(text)), 1.0] for text in texts]

        encoder = Encoder()
        cache = VectorCache(encoder)
        first = event("t1", "d1", "铁矿石|港口库存|中国|当期", "下降", "港口库存")
        second = event("t2", "d1", "铁矿石|到港量|中国|当期", "下降", "到港量")
        other = event("other", "d1", "铁矿石|发运量|澳洲|当期", "下降", "发运量")
        rank_events(first, [first, second, other], method="dense", encoder=cache)
        rank_events(second, [first, second, other], method="dense", encoder=cache)
        self.assertEqual(encoder.seen.count(event_signature(other)), 1)
