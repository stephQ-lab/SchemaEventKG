# -*- coding: utf-8 -*-
"""Multi-hop retrieval over graph.json: seed events, expand along causal and cross-period edges."""
import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

from eventkg.export import abstract_title
from eventkg.paths import allows_causal_walk, evidence_catalog, evidence_state

QUESTION_ALIASES = {
    "港存": "港口库存", "矿价": "价格", "去化": "下降", "去库": "下降", "累库": "上升",
}


class Index:
    def __init__(self, graph):
        self.graph = graph
        self.events = {event["id"]: event for event in graph["events"]}
        self.retrieval_method = graph.get("retrieval_method") or "bm25"
        self.dates = {doc["id"]: doc["report_date"] or "未给出年份" for doc in graph["documents"]}
        self.out = defaultdict(list)
        self.into = defaultdict(list)
        self.temporal_out = defaultdict(list)
        self.temporal_into = defaultdict(list)
        for group in ("relations", "continuity", "expectations", "cross"):
            for edge in graph[group]:
                if allows_causal_walk(edge):
                    self.out[edge["head"]].append(edge)
                    self.into[edge["tail"]].append(edge)
                elif edge.get("type") in ("事态延续", "预期验证"):
                    self.temporal_out[edge["head"]].append(edge)
                    self.temporal_into[edge["tail"]].append(edge)

    def seeds(self, question, limit=8, method=None, encoder=None):
        """Rank events with the configured retriever. Similarity does not prove a cause."""
        from eventkg.retrieval import rank_events
        query = question or ""
        for source, target in QUESTION_ALIASES.items():
            query = query.replace(source, target)
        wanted_date = re.search(r"(20\d{2})[-年.]?(\d{1,2})[-月.]?(\d{1,2})", question or "")
        probe = {"id": "__question__", "name": query, "evidence": query, "indicator": query}
        method = method or self.retrieval_method or "bm25"
        if method in ("dense", "hybrid") and encoder is None:
            raise RuntimeError("KGQA 的 %s 需要向量编码器，不会静默退回 BM25。" % method)
        ranked = rank_events(probe, list(self.events.values()), method=method, encoder=encoder,
                             top_k=max(limit * 3, limit))
        scored = []
        for row in ranked.get("candidates") or []:
            event = self.events.get(row["id"])
            if not event:
                continue
            score = float(row.get("score") or 0)
            if score <= 0:
                continue
            if event.get("salience") == "核心":
                score *= 1.2
            if wanted_date:
                date = "%s-%02d-%02d" % (wanted_date.group(1), int(wanted_date.group(2)), int(wanted_date.group(3)))
                if (event.get("event_time") or "") == date:
                    score *= 1.5
                elif self.dates.get(event["doc_id"]) == date:
                    score *= 1.1
            scored.append((score, row["id"]))
        scored.sort(reverse=True)
        return [event_id for _, event_id in scored[:limit]]

    def expand(self, seeds, hops=4, limit=20, outgoing=None, incoming=None):
        """Causes upstream and consequences downstream of each seed, as paths."""
        outgoing = self.out if outgoing is None else outgoing
        incoming = self.into if incoming is None else incoming
        paths = []
        for seed in seeds:
            for direction, table in (("下游", outgoing), ("上游", incoming)):
                queue = deque([(seed, [])])
                found = 0
                while queue and found < limit:
                    node, edges = queue.popleft()
                    if edges:
                        found += 1
                        paths.append({"seed": seed, "direction": direction, "edges": edges})
                    if len(edges) >= hops:
                        continue
                    visited = {seed}
                    for step in edges:
                        visited.add(step["tail"] if direction == "下游" else step["head"])
                    for edge in table.get(node, []):
                        nxt = edge["tail"] if direction == "下游" else edge["head"]
                        if nxt in visited:
                            continue
                        queue.append((nxt, edges + [edge]))
        return _longest(paths)

    def describe(self, event_id):
        event = self.events[event_id]
        return "%s（%s，%s，%s）" % (
            event["name"], self.dates[event["doc_id"]], event["status"],
            event.get("level_direction") or event.get("dir") or "")


def _longest(paths):
    keys = [tuple(id(edge) for edge in path["edges"]) for path in paths]
    kept = []
    for path, key in zip(paths, keys):
        if any(len(other) > len(key) and other[:len(key)] == key for other in keys):
            continue
        kept.append(path)
    return kept


def _label(edge):
    if edge["type"] == "跨期传导":
        return "跨期" + edge["relation"]
    if edge["type"] == "事态延续":
        return "延续·" + edge["change"]
    if edge["type"] == "预期验证":
        return "验证·" + edge["outcome"]
    return edge["type"]


def question_intent(question):
    text = question or ""
    if any(word in text for word in ("不同解释", "哪些报告", "分歧")):
        return "alternatives"
    if any(word in text for word in ("演化", "如何演变", "随后")):
        return "temporal"
    if any(word in text for word in ("指标", "如何变化", "数值")):
        return "indicator"
    if any(word in text for word in ("为什么", "原因", "为何")):
        return "why"
    return "why"


def _field_in_question(question, text):
    if len(text) >= 2 and text in question:
        return True
    return any(alias in question and text == canonical for alias, canonical in QUESTION_ALIASES.items())


def citation_bundle(edge, events, evidence_index=None):
    """One step's cause fact, effect fact and relation evidence stay separate."""
    head = events.get(edge.get("head")) or {}
    tail = events.get(edge.get("tail")) or {}
    state = evidence_state(edge, evidence_index if evidence_index is not None else edge.get("evidence_index"), events)
    record = edge.get("verification") or {}
    catalog = evidence_catalog(evidence_index if evidence_index is not None else edge.get("evidence_index"))
    relation_text = record.get("text") or ""
    if not relation_text:
        for evidence_id in edge.get("evidence_ids") or []:
            item = catalog.get(evidence_id) or {}
            if item.get("text"):
                relation_text = item["text"]
                break
    notes = {
        "VERIFIED": "",
        "UNVERIFIED": "有证据编号，但还没有验证它支持这条关系",
        "MISSING": "证据不存在或无法恢复，不能当作已确认因果",
        "DISPUTED": "同一事件对同时有支持和反对，保留争议",
        "CONTRADICTED": "存在明确反证，不能当作已确认因果",
        "FORECASTS": "结果仍是预测，不能当作已经发生的因果",
    }
    return {
        "head_event": edge.get("head"),
        "relation": edge.get("relation") or edge.get("type") or "",
        "tail_event": edge.get("tail"),
        "relation_evidence_ids": [item for item in (edge.get("evidence_ids") or []) if item],
        "head_fact_ids": list(edge.get("head_fact_ids") or []),
        "tail_fact_ids": list(edge.get("tail_fact_ids") or []),
        "counter_evidence_ids": list(edge.get("counter_evidence_ids") or []),
        "evidence_status": state,
        "certain": state == "VERIFIED",
        "relation_text": relation_text,
        "verification": record,
        "period": {"head": head.get("event_time") or head.get("time") or "",
                   "tail": tail.get("event_time") or tail.get("time") or ""},
        "forecast": head.get("status") == "预测中" or tail.get("status") == "预测中",
        "note": record.get("reason") or notes[state],
    }


def search(graph, question, hops=4):
    index = Index(graph)
    seeds = index.seeds(question)
    intent = question_intent(question)
    if intent == "temporal":
        paths = index.expand(seeds, hops=hops, outgoing=index.temporal_out, incoming=index.temporal_into)
    else:
        paths = index.expand(seeds, hops=hops)
    evidence_index = graph.get("evidence") or graph.get("evidence_index")
    evidence, numbered = [], {}

    def cite(event_id):
        if event_id not in numbered:
            numbered[event_id] = len(evidence) + 1
            evidence.append({"no": numbered[event_id], "event": index.describe(event_id)})
        return numbered[event_id]

    for seed in seeds:
        cite(seed)
    oriented = []
    for path in paths:
        nodes = [path["seed"]]
        for edge in path["edges"]:
            nodes.append(edge["tail"] if path["direction"] == "下游" else edge["head"])
        if path["direction"] == "上游":
            oriented.append((tuple(reversed(nodes)), list(reversed(path["edges"]))))
        else:
            oriented.append((tuple(nodes), path["edges"]))
    oriented.sort(key=lambda item: -len(item[0]))
    kept = []
    for nodes, edges in oriented:
        text = "\x00" + "\x00".join(nodes) + "\x00"
        if any(text in "\x00" + "\x00".join(other) + "\x00" for other, _ in kept):
            continue
        kept.append((nodes, edges))
    rendered = []
    bundles, uncertain_steps = [], []
    cited_relations = set()
    for nodes, edges in kept:
        parts = ["%s[%d]" % (index.events[nodes[0]]["name"], cite(nodes[0]))]
        stopped = False
        for node, edge in zip(nodes[1:], edges):
            bundle = citation_bundle(edge, index.events, evidence_index)
            if not bundle["certain"]:
                uncertain_steps.append(bundle)
                parts.append("-[证据不足]→ %s" % index.events[node]["name"])
                stopped = True
                break
            bundles.append(bundle)
            parts.append("-[%s]→ %s[%d]" % (_label(edge), index.events[node]["name"], cite(node)))
            support = bundle.get("relation_text") or edge.get("evidence") or edge.get("rule_evidence") or ""
            key = (edge.get("head"), edge.get("tail"), support)
            if support and key not in cited_relations:
                cited_relations.add(key)
                evidence.append({"no": len(evidence) + 1, "relation": {
                    "head": edge.get("head"), "tail": edge.get("tail"), "type": _label(edge),
                    "text": support, "status": edge.get("relation_class") or edge.get("type") or "",
                    "evidence_ids": bundle["relation_evidence_ids"],
                }})
        if stopped:
            parts.append("（该步证据不足，不继续当作已确认因果链）")
        rendered.append(" ".join(parts))
    rules = []
    wanted = {index.events[seed].get("abstract") or "" for seed in seeds}
    for edge in graph["abstract_edges"]:
        if edge["head"] in wanted or edge["tail"] in wanted:
            rules.append("%s -[%s]→ %s（支持 %d 次，%d 篇）" % (abstract_title(edge["head"]), edge["type"],
                                                         abstract_title(edge["tail"]), edge["support"], len(edge["docs"])))
    return {"question": question, "intent": intent, "seeds": [index.describe(seed) for seed in seeds],
            "paths": rendered, "rules": rules[:20], "evidence": evidence,
            "citation_bundles": bundles, "uncertain_steps": uncertain_steps,
            "confirmed_causal": bool(bundles) and not uncertain_steps}


def answer(result, workers=1):
    from eventkg.llm import ChatClient
    from eventkg.prompts import ANSWER_SYSTEM

    chat = ChatClient(Path(".eventkg_answer_cache.jsonl"), workers=workers)
    payload = json.dumps({"question": result["question"], "paths": result["paths"], "rules": result["rules"],
                          "evidence": result["evidence"]}, ensure_ascii=False)
    reply = chat.client.chat.completions.create(
        model=chat.model, temperature=0, max_tokens=2048,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": payload}],
        extra_body={"chat_template_kwargs": {"enable_thinking": chat.thinking}})
    return reply.choices[0].message.content or ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="事理图谱多跳检索")
    parser.add_argument("--graph", required=True, help="事件层输出目录里的 graph.json")
    parser.add_argument("--question", required=True)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--answer", action="store_true", help="调用模型基于检索证据作答")
    args = parser.parse_args(argv)
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    result = search(graph, args.question, hops=args.hops)
    print("种子事件：")
    for line in result["seeds"]:
        print("  -", line)
    print("链路：")
    for line in result["paths"]:
        print("  -", line)
    if result["rules"]:
        print("抽象规律：")
        for line in result["rules"]:
            print("  -", line)
    print("证据：")
    for item in result["evidence"]:
        print("  [%d] %s" % (item["no"], item["event"]))
    if args.answer:
        print("\n回答：\n" + answer(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
