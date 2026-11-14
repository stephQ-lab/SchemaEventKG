# -*- coding: utf-8 -*-
import unittest

from atlas_rag.kg_construction.neo4j_export import CypherGraph, _uid as triple_uid
from eventkg.export import Builder, build
from run_unified import ROOT, check_input, merge_cypher


class UnifiedJoinTest(unittest.TestCase):
    def test_category_document_and_event_type_join_the_triple_half(self):
        builder = Builder()
        self.assertEqual(builder.node("品类", "", "铁矿石", {}), triple_uid("品类", "", "铁矿石"))
        self.assertEqual(builder.node("文档", "", "report_a", {}), triple_uid("文档", "", "report_a"))
        self.assertEqual(builder.node("事件类型", "", "供应事件", {}), triple_uid("事件类型", "", "供应事件"))

    def test_runner_uses_the_new_event_layer(self):
        text = (ROOT / "run_unified.py").read_text(encoding="utf-8")
        self.assertNotIn("event_chains", text)
        self.assertIn("eventkg.pipeline", text)
        self.assertIn("enable_thinking", text)

    def test_preflight_merges_sections_into_fewer_chunks(self):
        plan = check_input(ROOT / "example" / "example_data" / "unified10", 3000)
        self.assertEqual(plan["documents"], 10)
        self.assertLess(plan["chunks"], 300)
        self.assertGreater(plan["event_windows"], 10)
        self.assertEqual(plan["triple_requests_at_least"], plan["chunks"] * 3)

    def test_triple_writer_merges_on_uid_with_labels(self):
        graph = CypherGraph()
        graph.add_edge("PB粉", "commodity", "", "属于", "铁矿石", "commodity", "")
        folder = ROOT / "example" / "generated" / "_merge_test"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "triples.cypher"
        graph.write(str(path))
        text = path.read_text(encoding="utf-8")
        self.assertIn("MERGE (n:`品类` {uid: row.uid}) SET n += row.props", text)
        self.assertIn("MATCH (a:`品类` {uid: row.a})", text)
        self.assertIn("CREATE INDEX IF NOT EXISTS", text)

    def test_body_grades_are_not_categories(self):
        result = _mini_result()
        builder = build(result)
        categories = _names(builder, "品类")
        entities = _names(builder, "实体")
        self.assertEqual(categories, {"铁矿石", "PVC"})
        self.assertIn("PB粉", entities)
        self.assertIn("铁水", entities)
        self.assertIn("螺纹钢", entities)
        self.assertIn("电石法PVC", entities)
        self.assertNotIn("PB粉", categories)
        self.assertNotIn("铁水", categories)
        self.assertNotIn("螺纹钢", categories)
        self.assertNotIn("钢材", categories)
        self.assertIn("钢材", entities)
        edges = set(builder.edges)
        ore = triple_uid("品类", "", "铁矿石")
        pvc = triple_uid("品类", "", "PVC")
        pb = triple_uid("实体", "", "PB粉")
        rebar = triple_uid("实体", "", "螺纹钢")
        hot = triple_uid("实体", "", "铁水")
        carbide = triple_uid("实体", "", "电石法PVC")
        self.assertIn((pb, "属于", ore), edges)
        self.assertIn((hot, "属于", ore), edges)
        self.assertIn((carbide, "属于", pvc), edges)
        self.assertNotIn((rebar, "属于", ore), edges)
        self.assertIn((_event_uid("ore-doc", "e-pb"), "涉及品类", ore), edges)
        self.assertIn((_event_uid("ore-doc", "e-rebar"), "涉及品类", ore), edges)
        self.assertIn((_event_uid("ore-doc", "e-ore"), "涉及品类", ore), edges)
        self.assertNotIn((_event_uid("ore-doc", "e-ore"), "品种", triple_uid("实体", "", "铁矿石")), edges)
        self.assertIn((_event_uid("pvc-doc", "e-pvc"), "涉及品类", pvc), edges)

    def test_merge_keeps_both_halves(self):
        folder = ROOT / "example" / "generated" / "_merge_test"
        folder.mkdir(parents=True, exist_ok=True)
        triple = folder / "triples.cypher"
        event = folder / "events.cypher"
        triple.write_text("MERGE (n:品类 {uid: 'aaa'});\n", encoding="utf-8")
        event.write_text("MERGE (n:事件 {uid: 'bbb'});\n", encoding="utf-8")
        dest = merge_cypher(triple, event, folder / "full.cypher")
        text = dest.read_text(encoding="utf-8")
        self.assertIn("品类", text)
        self.assertIn("事件", text)


def _names(builder, label):
    return {node["props"]["名称"] for node in builder.nodes.values() if label in node["labels"]}


def _event_uid(doc_id, event_id):
    return triple_uid("实例事件", doc_id, event_id)


def _event(event_id, doc_id, commodity, var):
    return {
        "id": event_id, "doc_id": doc_id, "name": commodity + "变化",
        "commodity": commodity, "indicator": "价格", "scope": "", "dim": "价格",
        "dir": "上升", "value": "", "change": "", "time": "本期", "period": "当期",
        "status": "已发生", "subject": "", "evidence": "原文", "evidences": [],
        "names": [], "sections": [], "salience": "核心", "var": var,
        "abstract": var + "|上升", "members": [],
    }


def _mini_result():
    specs = [
        ("e-pb", "ore-doc", "PB粉"),
        ("e-hot", "ore-doc", "铁水"),
        ("e-rebar", "ore-doc", "螺纹钢"),
        ("e-steel", "ore-doc", "钢材"),
        ("e-ore", "ore-doc", "铁矿石"),
        ("e-pvc", "pvc-doc", "电石法PVC"),
    ]
    events, variables = [], {}
    for event_id, doc_id, commodity in specs:
        key = commodity + "|价格||当期"
        events.append(_event(event_id, doc_id, commodity, key))
        variables[key] = {
            "commodity": commodity, "indicator": "价格", "scope": "", "period": "当期",
            "dim": "价格", "raw_forms": [], "source": "", "parent": "",
        }
    return {
        "documents": [
            {"id": "ore-doc", "title": "铁矿石周报", "report_date": "2026-04-17",
             "file_name": "a.pdf", "source_file": "a.pdf", "categories": ["铁矿石"]},
            {"id": "pvc-doc", "title": "PVC日报", "report_date": "2026-05-11",
             "file_name": "b.pdf", "source_file": "b.pdf", "categories": ["PVC"]},
        ],
        "events": events, "variables": variables, "abstracts": {},
        "relations": [], "continuity": [], "expectations": [], "cross": [],
        "observation_links": [], "observation_index": {}, "abstract_edges": [],
        "reasoning_chains": [], "local_chains": [], "transmission_chains": [], "storylines": [],
    }


if __name__ == "__main__":
    unittest.main()
