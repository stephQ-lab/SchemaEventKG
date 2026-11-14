# -*- coding: utf-8 -*-
"""Run one relation method on a shared event snapshot.

The snapshot is not extracted again. Legacy and Abductive must be given the same file.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventkg.corpus import load_corpus
from eventkg.experiment_sheet import sheet_for_method, write_sheet
from eventkg.link import cross_candidates, verify_cross
from eventkg.snapshot import code_version, prompt_versions, snapshot_sha256


def _plain(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _load_snapshot(path):
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    events = snapshot.get("events") or []
    relations = snapshot.get("relations") or []
    if not events:
        raise SystemExit("快照里没有 events")
    digest = snapshot_sha256(events, relations)
    recorded = snapshot.get("snapshot_sha256")
    if recorded and recorded != digest:
        raise SystemExit("快照 SHA256 与 events、relations 不一致")
    return events, relations, digest


def _documents(data_dir, name, events):
    """Every source report stays available. A document with no extracted event can still supply a passage."""
    corpus = load_corpus(str(data_dir), name)
    known = {doc["id"] for doc in corpus}
    for event in events:
        doc_id = event.get("doc_id")
        if doc_id and doc_id not in known:
            raise SystemExit("语料里没有文档 %s" % doc_id)
    return corpus


def _tokenizer_version():
    try:
        from eventkg.retrieval import tokenizer_version
        return tokenizer_version()
    except Exception:
        return ""


def _dictionary_hash():
    import hashlib
    from eventkg.retrieval import _DOMAIN_WORDS
    return hashlib.sha256("\n".join(_DOMAIN_WORDS).encode("utf-8")).hexdigest()


def _corpus_hash(data_dir):
    import hashlib
    root = Path(data_dir)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".json", ".pdf", ".txt", ".md"}
                   and "__pycache__" not in path.parts)
    for path in files:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(chat, digest, method, graph, retrieval_method="", data_dir=""):
    return {
        "code_version": code_version(),
        "event_snapshot_sha256": digest,
        "source_corpus_sha256": _corpus_hash(data_dir) if data_dir else "",
        "frozen": True,
        "model": chat.model,
        "parameters": {
            "temperature": 0,
            "thinking": bool(chat.thinking),
            "workers": 1,
            "base_url": os.environ.get("QWEN_BASE_URL", ""),
        },
        "snapshot_sha256": digest,
        "snapshot_path": str(Path(graph).resolve()),
        "prompt_versions": prompt_versions(),
        "relation_method": method,
        "retrieval_method": retrieval_method or "",
        "retrieval_status": retrieval_method if method == "abductive" else "NOT_USED",
        "embedding_backend": os.environ.get("EVENTKG_EMBEDDING", ""),
        "embedding_model": os.environ.get("EVENTKG_EMBEDDING_MODEL", ""),
        "embedding_model_version": os.environ.get("EVENTKG_EMBEDDING_MODEL", ""),
        "embedding_endpoint_fingerprint": hashlib.sha256(
            (os.environ.get("EVENTKG_EMBEDDING_BASE_URL", "") or "").encode("utf-8")).hexdigest()[:16],
        "tokenizer_version": _tokenizer_version(),
        "domain_dictionary_hash": _dictionary_hash(),
        "rrf_k": 60,
        "raw_responses": "llm_cache.jsonl",
        "model_accuracy": None,
        "note": "构造切片上的 F1 不是这次真实模型的准确率。",
    }


def _targets(events, documents):
    from eventkg.graph_induction import important_targets
    rows, stats = important_targets(events, documents)
    return [event["id"] for event in rows], stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="用同一份事件快照跑一种跨文档关系，不重新抽取")
    parser.add_argument("--graph", required=True, help="event_snapshot.json，或含相同 events 与 relations 的 graph.json")
    parser.add_argument("--data-dir", default="example/example_data/unified10")
    parser.add_argument("--name", default="unified10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--relation-method", choices=("legacy", "abductive"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retrieval-method", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--gold", default="", help="人工正例 JSON。没有就不要填，Recall 留空")
    args = parser.parse_args(argv)
    if args.workers != 1:
        raise SystemExit("同一时刻只允许一个请求，请使用 --workers 1")
    output = Path(args.output)
    if (output / "llm_cache.jsonl").exists():
        raise SystemExit("输出目录已有 llm_cache.jsonl。换一个新目录，不要接着旧缓存跑。")
    output.mkdir(parents=True, exist_ok=True)
    events, relations, digest = _load_snapshot(args.graph)
    documents = _documents(args.data_dir, args.name, events)
    by_id = {event["id"]: event for event in events}
    target_ids, target_stats = _targets(events, documents)
    from eventkg.llm import ChatClient
    chat = ChatClient(output / "llm_cache.jsonl", workers=1)
    chat.model_available()
    manifest = _manifest(chat, digest, args.relation_method, args.graph, args.retrieval_method, args.data_dir)
    gold = []
    if args.gold:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8")).get("pairs") or []
    if args.relation_method == "abductive":
        from eventkg.graph_induction import induce
        encoder = None
        if args.retrieval_method in ("dense", "hybrid"):
            from eventkg.embeddings import require_encoder
            encoder = require_encoder()
        links, payload = induce(chat, events, relations, documents, workers=1, multi_source=True, joint=True,
                                retrieval_method=args.retrieval_method, encoder=encoder)
        (output / "abduction.json").write_text(
            json.dumps(_plain(payload), ensure_ascii=False, indent=1), encoding="utf-8")
        overall = payload.get("global") or {}
        per_target = []
        for row in payload.get("targets") or []:
            candidates = row.get("candidates") or []
            per_target.append({
                "target": row.get("target"),
                "candidates": [item.get("id") for item in candidates],
                "candidate_rows": candidates,
                "decision": row.get("decision"),
                "rejection_reason": row.get("rejection_reason") or "",
                "recall_stats": row.get("recall_stats") or {},
            })
        confirmed = links
        uncertain = list(overall.get("possible") or [])
        rejected = list(overall.get("rejected") or [])
        result = {"method": "abductive", "manifest": manifest, "event_ids": [event["id"] for event in events],
                  "target_stats": target_stats, "confirmed": confirmed, "possible": uncertain, "rejected": rejected,
                  "candidate_graph": payload.get("candidate_graph") or []}
    else:
        candidates, rule_count = cross_candidates(events, relations, documents)
        verdicts = []
        links = verify_cross(chat, candidates, events, documents, 1, verdicts_out=verdicts)
        for row in verdicts:
            head = by_id.get(row["head"], {})
            tail = by_id.get(row["tail"], {})
            row["head_evidence"] = head.get("evidence") or ""
            row["tail_evidence"] = tail.get("evidence") or ""
        grouped = {}
        for row in candidates:
            grouped.setdefault(row["tail"], []).append(row["head"])
        per_target = [{"target": tail, "candidates": heads, "candidate_rows": [
            item for item in _plain(candidates) if item["tail"] == tail
        ]} for tail, heads in grouped.items()]
        confirmed = [row for row in verdicts if row["status"] == "confirmed"]
        uncertain = [row for row in verdicts if row["status"] == "uncertain"]
        rejected = [row for row in verdicts if row["status"] == "rejected"]
        result = {"method": "legacy", "manifest": manifest, "event_ids": [event["id"] for event in events],
                  "target_stats": target_stats, "rules": rule_count,
                  "candidates": _plain(candidates), "verdicts": verdicts, "confirmed": links}
    sheet = sheet_for_method(result["method"], manifest, target_ids, per_target, confirmed, uncertain, rejected, gold)
    write_sheet(output / "annotation_sheet.json", sheet)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "relations.json").write_text(json.dumps(_plain(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"method": result["method"], "events": len(result["event_ids"]),
                      "snapshot_sha256": digest, "confirmed": len(confirmed),
                      "uncertain": len(uncertain), "rejected": len(rejected),
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
