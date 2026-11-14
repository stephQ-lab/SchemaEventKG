# -*- coding: utf-8 -*-
"""Write a Cypher file that Neo4j Browser or cypher-shell can load.

Document nodes stay apart. Event types are one shared ontology, so 属于
still means a grade belongs to a commodity and 类型 means an event's type.
"""
import hashlib
import os

import csv
import json
from pathlib import PureWindowsPath

from atlas_rag.kg_construction.commodity import belongs_edge, shared_scope
from atlas_rag.kg_construction.cypher_writer import write_cypher
from atlas_rag.document_input.evidence import add_structured_evidence, exact_span, load_source_index
from atlas_rag.kg_construction.event_ontology import (
    EVENT_TYPES,
    accept_causal,
    event_status,
    event_type_of,
    normalize_event_relation,
    trigger_word,
)

LABELS = {
    "event": "事件",
    "commodity": "品类",
    "entity": "实体",
    "concept": "概念",
    "passage": "原文",
    "event_type": "事件类型",
    "document": "文档",
    "evidence": "证据片段",
}


def _uid(label, scope, name):
    raw = "%s\n%s\n%s" % (label, scope or "", name or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _escape(value):
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r", "")
        .replace("\n", "\\n")
    )


def _rel(name):
    return "`" + (name or "关联").replace("`", "") + "`"


class CypherGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._seen_edges = set()

    def add_node(self, name, ntype, scope="", text="", status="", extra=None):
        if not name:
            return
        label = LABELS.get(ntype, "实体")
        node_scope = shared_scope(name, ntype, scope)
        uid = _uid(label, node_scope, name)
        row = self.nodes.get(uid)
        if row is None:
            self.nodes[uid] = {
                "uid": uid,
                "label": label,
                "name": name,
                "scope": node_scope,
                "text": text or "",
                "status": status or "",
                "extra": {},
            }
        else:
            if text and not row["text"]:
                row["text"] = text
            if status and not row["status"]:
                row["status"] = status
        if extra:
            for key, value in extra.items():
                if value not in ("", None):
                    self.nodes[uid]["extra"].setdefault(key, value)
        return uid

    def add_edge(self, head, head_type, head_scope, relation, tail, tail_type, tail_scope):
        hu = self.add_node(head, head_type, head_scope)
        tu = self.add_node(tail, tail_type, tail_scope)
        if not hu or not tu or not relation:
            return
        key = (hu, relation, tu)
        if key in self._seen_edges:
            return
        self._seen_edges.add(key)
        self.edges.append((hu, relation, tu))

    def write(self, output_file):
        nodes = {}
        for row in self.nodes.values():
            props = {"名称": row["name"], "文档": row["scope"], "文本": row["text"], "状态": row.get("status")}
            for key, value in (row.get("extra") or {}).items():
                props.setdefault(key, value)
            nodes[row["uid"]] = {"labels": [row["label"]], "props": props}
        edges = [(hu, (relation or "关联").replace("`", ""), tu, {}) for hu, relation, tu in self.edges]
        return write_cypher(output_file, nodes, edges, header=(
            "在 Neo4j Browser 里打开本文件并运行，或执行 cypher-shell -f 本文件",
            "品类来自文章标题。属于：品种 -> 品类。类型：事件 -> 事件类型。导致：原文中有触发词的因果。",
            "节点按 uid MERGE，重复导入只会更新属性，不会产生重复节点。",
        ))


def graphml_to_cypher(graphml_path, output_file):
    """Convert one GraphML graph and apply the causal-trigger rule."""
    import networkx as nx

    graph = nx.read_graphml(graphml_path)
    passages = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") == "passage":
            passages[node] = data.get("id") or ""

    def passage_of(node):
        texts = []
        for _, other, edge in graph.out_edges(node, data=True):
            if graph.nodes[other].get("type") == "passage":
                texts.append(passages.get(other, ""))
        for other, _, edge in graph.in_edges(node, data=True):
            if graph.nodes[other].get("type") == "passage":
                texts.append(passages.get(other, ""))
        return "\n".join(texts)

    cypher = CypherGraph()
    passage_name = {}
    for node, data in graph.nodes(data=True):
        ntype = data.get("type") or "entity"
        scope = "" if ntype in ("concept", "event_type") else (data.get("file_id") or "")
        if ntype == "passage":
            label = data.get("file_id") or node
            passage_name[node] = label
            cypher.add_node(label, "passage", scope, passages.get(node, ""))
        else:
            cypher.add_node(data.get("id") or "", ntype, scope)
        if ntype == "event":
            cypher.add_node(data.get("id") or "", "event", data.get("file_id") or "", status=event_status(data.get("id") or ""))
            kind = event_type_of(data.get("id") or "")
            if kind:
                cypher.add_edge(data.get("id") or "", "event", data.get("file_id") or "", "类型", kind, "event_type", "")

    kept_cause = 0
    dropped_cause = 0
    for head, tail, edge in graph.edges(data=True):
        if edge.get("type") == "Concept" or edge.get("relation") == "has_concept":
            concept_node, owner = (tail, head) if graph.nodes[tail].get("type") == "concept" else (head, tail)
            owner_data = graph.nodes[owner]
            cypher.add_edge(
                owner_data.get("id") or "",
                owner_data.get("type") or "entity",
                owner_data.get("file_id") or "",
                "有概念",
                graph.nodes[concept_node].get("id") or "",
                "concept",
                "",
            )
            continue
        relation = edge.get("relation") or ""
        head_data = graph.nodes[head]
        tail_data = graph.nodes[tail]
        if edge.get("type") == "Source" or relation == "mention in":
            source_node, passage_node = (head, tail) if tail_data.get("type") == "passage" else (tail, head)
            source_data = graph.nodes[source_node]
            cypher.add_edge(
                source_data.get("id") or "",
                source_data.get("type") or "entity",
                source_data.get("file_id") or "",
                "提及",
                passage_name.get(passage_node, ""),
                "passage",
                graph.nodes[passage_node].get("file_id") or "",
            )
            continue
        if head_data.get("type") == "event" and tail_data.get("type") == "event":
            source = passage_of(head) or passage_of(tail)
            rewritten = accept_causal(head_data.get("id") or "", relation, tail_data.get("id") or "", source)
            if rewritten is None:
                dropped_cause += 1
                continue
            if rewritten == "导致":
                kept_cause += 1
                relation = "导致"
        cypher.add_edge(
            head_data.get("id") or "",
            head_data.get("type") or "entity",
            head_data.get("file_id") or "",
            relation,
            tail_data.get("id") or "",
            tail_data.get("type") or "entity",
            tail_data.get("file_id") or "",
        )
    nodes, edges = cypher.write(output_file)
    print(
        "cypher", output_file,
        "nodes", nodes,
        "edges", edges,
        "causal_kept", kept_cause,
        "causal_dropped", dropped_cause,
        "trigger_sample", trigger_word("受检修影响"),
        flush=True,
    )
    return output_file


def _rows(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _evidence_span(name, source):
    if not name or not source:
        return None
    return exact_span(name, source)


def _write_ontology(path, concept_names):
    payload = {
        "事件类型": [name for name, _ in EVENT_TYPES],
        "概念": sorted({name for name in concept_names if name}),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def packaged_path(source):
    if "group" in source and "relative" in source:
        return "source_documents/%s/%s" % (source["group"], source["relative"])
    if "path" in source:
        parts = PureWindowsPath(source["path"]).parts
        if "output" in parts:
            last_output = max(index for index, part in enumerate(parts) if part == "output")
            return "source_documents/output/" + "/".join(parts[last_output + 1:])
    return ""


def csv_to_cypher(node_file, edge_file, text_node_file, text_edge_file, output_file, concept_node_file=None, concept_edge_file=None, source_index_file=None):
    """Write Neo4j Cypher from the triple CSVs. GraphML is not required.

    Categories, varieties and named organizations are one node across articles.
    Events and evidence keep the document they came from.
    """
    nodes = _rows(node_file)
    edges = _rows(edge_file)
    texts = {row.get("text_id:ID"): row.get("original_text") or "" for row in _rows(text_node_file)}
    by_key = {}
    for row in nodes:
        name = row.get("name:ID") or ""
        scope = row.get("scope") or ""
        by_key[(name, scope)] = row.get("type") or "entity"

    passage_of = {}
    passage_ids = {}
    for row in _rows(text_edge_file):
        name = row.get(":START_ID") or ""
        scope = row.get("scope") or ""
        text_id = row.get(":END_ID") or ""
        passage_of.setdefault((name, scope), []).append(texts.get(text_id, ""))
        passage_ids.setdefault((name, scope), []).append(text_id)

    device_scopes = {}
    for (name, scope), ntype in by_key.items():
        if ntype == "entity" and "装置" in name and shared_scope(name, ntype, scope) == scope and scope:
            device_scopes.setdefault(name, set()).add(scope)
    reject_path = os.path.join(os.path.dirname(output_file) or ".", "rejected_devices.jsonl")
    folder = os.path.dirname(output_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(reject_path, "w", encoding="utf-8") as handle:
        for name, scopes in sorted(device_scopes.items()):
            if len(scopes) < 2:
                continue
            handle.write(json.dumps({
                "name": name,
                "reason": "同名装置没有企业或地点，不能跨文章合并",
                "documents": sorted(scopes),
            }, ensure_ascii=False) + "\n")

    concepts = {}
    concept_names = []
    for row in _rows(concept_node_file):
        raw_name = row.get("name") or ""
        collapsed = "".join(raw_name.split()).replace("，", "").replace("、", "").replace(",", "").replace("/", "")
        shown = collapsed or raw_name
        if shown and shown not in concept_names:
            concept_names.append(shown)
        concepts[row.get("concept_id:ID")] = shown

    cypher = CypherGraph()
    source_index = load_source_index(source_index_file)
    registry_path = os.path.join(os.path.dirname(source_index_file), "document_registry.json") if source_index_file else ""
    if registry_path and os.path.isfile(registry_path):
        with open(registry_path, encoding="utf-8") as handle:
            registry = json.load(handle)
        for document in registry:
            source = document["sources"][0]
            packaged = packaged_path(source)
            cypher.add_node(document["id"], "document", "", extra={"标题": document["title"],
                            "来源文件": document["source_file"], "打包路径": packaged,
                            "来源版本": json.dumps(document["sources"], ensure_ascii=False)})
            for parent in document["parent_article_ids"]:
                cypher.add_edge("news_" + parent, "document", "", "附件", document["id"], "document", "")
    for (name, scope), ntype in by_key.items():
        status = event_status(name) if ntype == "event" else ""
        extra = {}
        if ntype == "event":
            source = "\n".join(passage_of.get((name, scope), []))
            span = None if source_index else _evidence_span(name, source)
            if span:
                extra["起始"] = span[0]
                extra["结束"] = span[1]
        cypher.add_node(name, ntype, scope, status=status, extra=extra)
        if ntype == "event":
            kind = event_type_of(name)
            if kind:
                cypher.add_edge(name, "event", scope, "类型", kind, "event_type", "")

    seen_docs = set()
    for text_id, text in texts.items():
        if not text_id:
            continue
        cypher.add_node(text_id, "passage", text_id, text)
    for (scope, text_id), record in source_index.items():
        meta = record["metadata"]
        pages = sorted({unit["page"] for unit in record["units"] if unit.get("page") is not None})
        cypher.add_node(text_id, "passage", text_id, extra={"来源文件": meta["source_file"],
                        "页码": json.dumps(pages), "文本SHA256": meta["text_sha256"]})
        cypher.add_node(scope, "document", "", extra={"标题": meta.get("document_title") or meta["file"],
                        "来源文件": meta["source_file"], "文本SHA256": meta["text_sha256"],
                        "来源版本": meta.get("source_variants")})
        for parent_id in json.loads(meta.get("parent_article_ids") or "[]"):
            cypher.add_edge("news_" + parent_id, "document", "", "附件", scope, "document", "")
    for row in _rows(text_edge_file):
        scope = row.get("scope") or ""
        text_id = row.get(":END_ID") or ""
        name = row.get(":START_ID") or ""
        ntype = by_key.get((name, scope), "entity")
        cypher.add_edge(name, ntype, scope, "提及", text_id, "passage", text_id)
        if scope and scope not in seen_docs:
            seen_docs.add(scope)
            cypher.add_node(scope, "document", "")
        if scope and text_id:
            cypher.add_edge(text_id, "passage", text_id, "收录于", scope, "document", "")

    kept_cause = 0
    dropped_cause = 0
    for row in edges:
        head = row.get(":START_ID") or ""
        tail = row.get(":END_ID") or ""
        relation = row.get("relation") or ""
        scope = row.get("scope") or ""
        head_type = by_key.get((head, scope), "entity")
        tail_type = by_key.get((tail, scope), "entity")
        if relation == "属于" and not belongs_edge(head, tail):
            continue
        if head_type == "event" and tail_type == "event":
            source = "\n".join(passage_of.get((head, scope), []) + passage_of.get((tail, scope), []))
            rewritten = normalize_event_relation(head, relation, tail, source)
            if rewritten is None:
                dropped_cause += 1
                continue
            if rewritten == "导致":
                kept_cause += 1
            relation = rewritten
        cypher.add_edge(head, head_type, scope, relation, tail, tail_type, scope)

    for row in _rows(concept_edge_file):
        owner = row.get(":START_ID") or ""
        scope = row.get("scope") or ""
        concept = concepts.get(row.get(":END_ID") or "") or ""
        if not concept:
            continue
        cypher.add_edge(owner, by_key.get((owner, scope), "entity"), scope, "有概念", concept, "concept", "")

    evidence_report = add_structured_evidence(cypher, source_index, passage_ids, by_key) if source_index else {}
    if source_index:
        with open(os.path.join(os.path.dirname(output_file), "evidence_report.json"), "w", encoding="utf-8") as handle:
            json.dump(evidence_report, handle, ensure_ascii=False, indent=2)
    count_nodes, count_edges = cypher.write(output_file)
    ontology_path = os.path.join(os.path.dirname(output_file) or ".", "ontology.json")
    _write_ontology(ontology_path, concept_names)
    print(
        "cypher", output_file,
        "nodes", count_nodes,
        "edges", count_edges,
        "causal_kept", kept_cause,
        "causal_dropped", dropped_cause,
        "ontology", ontology_path,
        flush=True,
    )
    return output_file


def load_cypher(cypher_file, uri, user, password):
    """Run the Cypher file on a Neo4j server. The file remains the portable output."""
    from neo4j import GraphDatabase

    statements = []
    with open(cypher_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            statements.append(line[:-1] if line.endswith(";") else line)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for statement in statements:
                session.run(statement)
    finally:
        driver.close()
    print("neo4j loaded", len(statements), "statements", flush=True)
    return len(statements)
