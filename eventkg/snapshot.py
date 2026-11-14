# -*- coding: utf-8 -*-
"""Shared event snapshot. Relation methods read this file and do not extract again."""
import hashlib
import json
from pathlib import Path

from eventkg import prompts

ROOT = Path(__file__).resolve().parents[1]
PROMPT_NAMES = ("EXTRACT_SYSTEM", "DOCUMENT_SYSTEM", "ONTOLOGY_SYSTEM", "CROSS_SYSTEM", "ABDUCT_SYSTEM")


def code_version():
    return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def prompt_versions():
    versions = {}
    for name in PROMPT_NAMES:
        text = getattr(prompts, name)
        versions[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return versions


def snapshot_sha256(events, relations):
    raw = json.dumps({"events": events, "relations": relations},
                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_event_snapshot(path, events, relations, documents, stats=None):
    digest = snapshot_sha256(events, relations)
    payload = {
        "kind": "event_snapshot",
        "code_version": code_version(),
        "frozen": True,
        "prompt_versions": prompt_versions(),
        "snapshot_sha256": digest,
        "events": events,
        "relations": relations,
        "documents": [{
            "id": doc.get("id"),
            "title": doc.get("title") or "",
            "file_name": doc.get("file_name") or "",
            "report_date": doc.get("report_date") or "",
        } for doc in documents],
        "stats": stats or {},
        "note": "只含抽取后的 events 和篇内 relations。没有跨期核验，也没有溯因边。",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"path": str(path), "events": len(events), "relations": len(relations),
            "snapshot_sha256": digest, "code_version": payload["code_version"],
            "prompt_versions": payload["prompt_versions"]}
