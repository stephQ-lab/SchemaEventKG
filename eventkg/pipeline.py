# -*- coding: utf-8 -*-
"""Event layer: windows -> document synthesis -> variables -> merge -> cross-document links -> chains."""
import argparse
import json
import sys
import time
from pathlib import Path

from eventkg.corpus import load_corpus
from eventkg.export import write_all
from eventkg.extract import extract_windows, synthesize_documents
from eventkg.link import (
    abstract_layer,
    align_polarity,
    attach_observations,
    continuity,
    cross_candidates,
    expectation,
    merge,
    verify_cross,
)
from eventkg.ontology import assign, induce
from eventkg.paths import explain_paths, reasoning_chains, transmission_chains


def _remap_storylines(storylines, events):
    alias = {member: event["id"] for event in events for member in event["members"]}
    kept = []
    for line in storylines:
        sequence = []
        for item in line["events"]:
            target = alias.get(item)
            if target and (not sequence or sequence[-1] != target) and target not in sequence:
                sequence.append(target)
        if len(sequence) >= 3:
            kept.append(dict(line, events=sequence))
    return kept


def run(data_dir, output_dir, workers=8, chat=None, name="unified10", expected_docs=None,
        relation_method="legacy", retrieval_method="bm25", encoder=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timing = {}
    clock = time.time()

    def lap(stage):
        nonlocal clock
        now = time.time()
        timing[stage] = round(now - clock, 1)
        clock = now
        print("阶段完成", stage, timing[stage], "秒", flush=True)

    documents = load_corpus(data_dir, name)
    if expected_docs and len(documents) != expected_docs:
        raise ValueError("文档数为 %d，应为 %d" % (len(documents), expected_docs))
    if chat is None:
        from eventkg.llm import ChatClient
        chat = ChatClient(output / "llm_cache.jsonl", workers=workers)
        chat.model_available()
    stats = {"documents": len(documents), "windows": sum(len(doc["windows"]) for doc in documents)}
    lap("load")

    events, relations, stats["extract"] = extract_windows(chat, documents, workers)
    lap("extract")
    added, duplicates, storylines, stats["document"] = synthesize_documents(chat, documents, events, relations, workers)
    relations = relations + added
    lap("document")
    mapping, variables, stats["ontology"] = induce(chat, events, workers)
    assign(events, mapping)
    lap("ontology")
    if relation_method not in ("legacy", "abductive", "snapshot"):
        raise ValueError("relation_method 只能是 legacy、abductive 或 snapshot")
    events, relations, stats["merge"] = merge(events, relations, duplicates)
    if relation_method == "snapshot":
        stats["merge"]["polarity_relabelled"] = 0
        stats["merge"]["polarity_conflicts"] = align_polarity(events, relations, rewrite=False)
        from eventkg.snapshot import write_event_snapshot
        info = write_event_snapshot(output / "event_snapshot.json", events, relations, documents, stats)
        lap("snapshot")
        info["timing_seconds"] = timing
        (output / "report_snapshot.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"events": info["events"], "relations": info["relations"],
                          "snapshot_sha256": info["snapshot_sha256"]}, ensure_ascii=False), flush=True)
        return {"events": events, "relations": relations, "documents": documents, "stats": stats}, info
    if relation_method == "abductive":
        stats["merge"]["polarity_relabelled"] = 0
        stats["merge"]["polarity_conflicts"] = align_polarity(events, relations, rewrite=False)
    else:
        stats["merge"]["polarity_relabelled"] = 0
        stats["merge"]["polarity_conflicts"] = align_polarity(events, relations, rewrite=False)
    storylines = _remap_storylines(storylines, events)
    continuity_links = [dict(link, record_kind="时间序列") for link in continuity(events, documents)]
    expectation_links = [dict(link, record_kind="持续状态") for link in expectation(events, documents)]
    abduction = None
    if relation_method == "abductive":
        from eventkg.graph_induction import induce as induce_graphs
        if retrieval_method in ("dense", "hybrid") and encoder is None:
            from eventkg.embeddings import require_encoder
            encoder = require_encoder()
        cross_links, abduction = induce_graphs(
            chat, events, relations, documents, workers, multi_source=True, joint=True,
            retrieval_method=retrieval_method, encoder=encoder)
        rows = abduction["targets"]
        stats["cross"] = {"method": "abductive", "targets": len(rows),
                          "candidates": sum(row["recall_stats"]["hits"] for row in rows),
                          "kept": sum(row["recall_stats"]["kept"] for row in rows),
                          "accepted": len(cross_links),
                          "possible": len(abduction["global"]["possible"]),
                          "conflicts": len(abduction["global"]["conflicts"]),
                          "mode": abduction.get("mode", "D"),
                          "rejected": sum(1 for row in rows if row["decision"] == "NO_SUPPORTED_EXPLANATION")}
        lap("link")
        stats["merge"]["polarity_conflicts"] += align_polarity(events, cross_links, rewrite=False)
        lap("cross")
        splice_continuity = False
    else:
        candidates, rule_count = cross_candidates(events, relations, documents)
        stats["cross"] = {"method": "legacy", "rules": rule_count, "candidates": len(candidates)}
        lap("link")
        cross_links = verify_cross(chat, candidates, events, documents, workers)
        stats["merge"]["polarity_conflicts"] = stats["merge"].get("polarity_conflicts", 0) + align_polarity(
            events, cross_links, rewrite=False)
        lap("cross")
        splice_continuity = True
    observations, observation_links = attach_observations(events, documents)
    from eventkg.graph_induction import index_evidence, merge_catalog
    catalog = index_evidence(events, documents, relations + cross_links)
    if abduction and abduction.get("evidence"):
        catalog = merge_catalog(catalog, abduction["evidence"])
    abstracts, abstract_edges = abstract_layer(
        events, relations, cross_links, variables, evidence_index=catalog)
    typed_paths = explain_paths(
        events, relations, continuity_links + expectation_links + cross_links,
        evidence_index=catalog)
    bridge_links = cross_links if relation_method == "abductive" else (
        continuity_links + expectation_links + cross_links)
    classified_chains = {}
    cross_chains, local_chains = reasoning_chains(
        events, relations, bridge_links, splice_continuity=splice_continuity, evidence_index=catalog,
        classified=classified_chains)
    transmission = transmission_chains(abstract_edges)
    lap("paths")

    result = {
        "documents": documents, "events": events, "relations": relations, "variables": variables,
        "continuity": continuity_links, "expectations": expectation_links, "cross": cross_links,
        "series": continuity_links, "persistence": expectation_links,
        "cross_candidates": stats["cross"].get("candidates", 0), "relation_method": relation_method,
        "abduction": abduction, "abstracts": abstracts, "abstract_edges": abstract_edges,
        "observation_links": observation_links,
        "observation_index": {item["id"]: item for item in observations},
        "evidence": catalog,
        "evidence_catalog": catalog,
        "reasoning_chains": cross_chains, "local_chains": local_chains,
        "candidate_chains": classified_chains.get("candidate") or [],
        "rejected_chains": classified_chains.get("rejected") or [],
        "typed_paths": typed_paths,
        "transmission_chains": transmission, "storylines": storylines,
        "stats": stats, "timing": timing, "llm_stats": dict(getattr(chat, "stats", {})),
    }
    report = write_all(result, output)
    lap("export")
    report["timing_seconds"] = timing
    report_name = "report_abductive.json" if relation_method == "abductive" else "report.json"
    (output / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "instance_events", "variables", "abstract_events", "cross_doc_edges", "reasoning_chains",
        "local_chains", "transmission_chains", "cypher_nodes", "cypher_edges")}, ensure_ascii=False), flush=True)
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="事件层：跨文档事理图谱")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="unified10")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-docs", type=int, default=0)
    parser.add_argument("--replay-cache", default="", help="只读这个缓存文件，不调用模型（未命中的调用按失败处理）")
    parser.add_argument("--relation-method", choices=("legacy", "abductive", "snapshot"), default="legacy",
                        help="legacy：历史规律匹配后逐对判断。abductive：按目标事件检索候选并选择解释图。snapshot：只写事件快照")
    parser.add_argument("--retrieval-method", choices=("bm25", "dense", "hybrid"), default="hybrid",
                        help="正式候选召回。dense 与 hybrid 需要 EVENTKG_EMBEDDING，不会回退到字符 Jaccard")
    args = parser.parse_args(argv)
    chat = None
    if args.replay_cache:
        from eventkg.llm import ReplayChat
        chat = ReplayChat(args.replay_cache)
    run(args.data_dir, args.output_dir, workers=args.workers, chat=chat, name=args.name,
        expected_docs=args.expected_docs or None, relation_method=args.relation_method,
        retrieval_method=args.retrieval_method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
