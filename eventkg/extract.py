# -*- coding: utf-8 -*-
"""Stage 1: events and in-window relations. Stage 2: document-level synthesis."""
import json
import re
from collections import Counter

from eventkg.corpus import document_view, sentences_of
from eventkg.llm import parallel_map
from eventkg.prompts import DOCUMENT_SYSTEM, EXTRACT_SYSTEM
from eventkg.validate import SAME_FACT, clean_event, clean_relation
from eventkg.lexicon import compact


def extract_windows(chat, documents, workers):
    jobs = [(doc, window) for doc in documents for window in doc["windows"]]

    def run(job):
        doc, window = job
        user = json.dumps({
            "title": doc["title"],
            "report_date": doc["report_date"] or "原文未给出年份",
            "categories": doc["categories"],
            "window": window["text"],
        }, ensure_ascii=False)
        parsed = chat.complete_json(EXTRACT_SYSTEM, user, max_tokens=8192, tag="extract:" + window["id"])
        return parse_window(parsed, doc, window)

    results = parallel_map(run, jobs, workers, "窗口抽取")
    events, relations, stats = [], [], {"windows": len(jobs), "failed_windows": 0,
                                        "raw_events": 0, "kept_events": 0,
                                        "raw_relations": 0, "kept_relations": 0}
    event_reasons, relation_reasons = Counter(), Counter()
    for found, linked, counts in results:
        events.extend(found)
        relations.extend(linked)
        for key, value in counts.items():
            if key == "event_reasons":
                event_reasons.update(value)
            elif key == "relation_reasons":
                relation_reasons.update(value)
            else:
                stats[key] = stats.get(key, 0) + value
    stats["dropped_events"] = dict(event_reasons.most_common())
    stats["dropped_relations"] = dict(relation_reasons.most_common())
    return events, relations, stats


def _rows(parsed, key):
    if isinstance(parsed, dict):
        rows = parsed.get(key)
        if rows is None and key == "events":
            rows = parsed.get("data")
        return rows if isinstance(rows, list) else []
    if isinstance(parsed, list) and key == "events":
        return parsed
    return []


def _section_at(window_text, evidence):
    position = window_text.find(evidence[:12]) if evidence else -1
    if position < 0:
        compacted = compact(evidence)[:10]
        for line_start, line in _lines_with_offsets(window_text):
            if compacted and compacted in compact(line):
                position = line_start
                break
    if position < 0:
        position = 0
    headers = [(match.start(), match.group(1)) for match in re.finditer(r"【章节】([^\n]*)", window_text)]
    section = ""
    for start, name in headers:
        if start <= position:
            section = name.replace("（续）", "")
    return section


def _lines_with_offsets(text):
    offset = 0
    for line in text.split("\n"):
        yield offset, line
        offset += len(line) + 1


def parse_window(parsed, doc, window):
    counts = {"raw_events": 0, "kept_events": 0, "raw_relations": 0, "kept_relations": 0, "failed_windows": 0,
              "event_reasons": Counter(), "relation_reasons": Counter()}
    if parsed is None:
        counts["failed_windows"] = 1
        return [], [], counts
    text = window["text"]
    sentences = sentences_of(text)
    events = []
    local = {}
    raw_events = _rows(parsed, "events")
    counts["raw_events"] = len(raw_events)
    for index, raw in enumerate(raw_events, 1):
        cleaned = clean_event(raw, text, sentences, doc["categories"], counts["event_reasons"])
        if not cleaned:
            continue
        cleaned["id"] = "%s:e%d" % (window["id"], index)
        cleaned["doc_id"] = doc["id"]
        cleaned["window_id"] = window["id"]
        cleaned["section"] = _section_at(text, cleaned["evidence"])
        cleaned["evidences"] = [cleaned["evidence"]]
        cleaned["sections"] = [cleaned["section"]]
        local[cleaned["local_id"] or ("E%d" % index)] = cleaned["id"]
        events.append(cleaned)
    counts["kept_events"] = len(events)
    by_id = {event["id"]: event for event in events}
    relations = []
    raw_relations = _rows(parsed, "relations")
    counts["raw_relations"] = len(raw_relations)
    seen = set()
    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue
        mapped = dict(raw)
        mapped["head"] = local.get(str(raw.get("head") or "").strip(), "")
        mapped["tail"] = local.get(str(raw.get("tail") or "").strip(), "")
        cleaned = clean_relation(mapped, by_id, text, sentences, inferred_allowed=False,
                                 reasons=counts["relation_reasons"])
        if not cleaned:
            continue
        key = (cleaned["head"], cleaned["type"], cleaned["tail"])
        if key in seen:
            counts["relation_reasons"]["重复"] += 1
            continue
        seen.add(key)
        cleaned["source"] = "窗口"
        cleaned["doc_id"] = doc["id"]
        relations.append(cleaned)
    counts["kept_relations"] = len(relations)
    return events, relations, counts


def synthesize_documents(chat, documents, events, relations, workers):
    by_doc = {}
    for event in events:
        by_doc.setdefault(event["doc_id"], []).append(event)
    rel_by_doc = {}
    for relation in relations:
        rel_by_doc.setdefault(relation["doc_id"], []).append(relation)
    jobs = [doc for doc in documents if len(by_doc.get(doc["id"], [])) >= 2]

    def run(doc):
        doc_events = by_doc[doc["id"]]
        short = {event["id"]: "D%d" % index for index, event in enumerate(doc_events, 1)}
        back = {value: key for key, value in short.items()}
        catalog = [{
            "id": short[event["id"]], "name": event["name"], "dim": event["dim"], "dir": event["dir"],
            "status": event["status"], "section": event["section"],
        } for event in doc_events]
        existing = [[short[item["head"]], item["type"], short[item["tail"]]]
                    for item in rel_by_doc.get(doc["id"], []) if item["head"] in short and item["tail"] in short]
        user = json.dumps({
            "title": doc["title"],
            "report_date": doc["report_date"] or "原文未给出年份",
            "events": catalog,
            "relations": existing,
            "text": document_view(doc),
        }, ensure_ascii=False)
        parsed = chat.complete_json(DOCUMENT_SYSTEM, user, max_tokens=6144, tag="document:" + doc["id"])
        return parse_document(parsed, doc, doc_events, back, rel_by_doc.get(doc["id"], []))

    results = parallel_map(run, jobs, workers, "整篇综合")
    added, duplicates, storylines = [], [], []
    stats = {"documents": len(jobs), "failed_documents": 0, "doc_relations": 0, "duplicate_groups": 0,
             "storylines": 0}
    reasons = Counter()
    for found, groups, lines, failed, dropped in results:
        added.extend(found)
        duplicates.extend(groups)
        storylines.extend(lines)
        stats["failed_documents"] += failed
        reasons.update(dropped)
    stats["dropped_relations"] = dict(reasons.most_common())
    stats["doc_relations"] = len(added)
    stats["duplicate_groups"] = len(duplicates)
    stats["storylines"] = len(storylines)
    return added, duplicates, storylines, stats


def parse_document(parsed, doc, doc_events, back, existing):
    reasons = Counter()
    if not isinstance(parsed, dict):
        return [], [], [], 1, reasons
    by_id = {event["id"]: event for event in doc_events}
    source = doc["narrative"]
    sentences = sentences_of(source)
    def _relation_key(item):
        return (item.get("head"), item.get("type") or item.get("relation"), item.get("tail"))

    known = {_relation_key(item) for item in existing}
    added = []
    groups = []
    for raw in parsed.get("relations") or []:
        if not isinstance(raw, dict):
            continue
        mapped = dict(raw)
        mapped["head"] = back.get(str(raw.get("head") or "").strip(), "")
        mapped["tail"] = back.get(str(raw.get("tail") or "").strip(), "")
        if mapped["head"] in by_id and mapped["tail"] in by_id and SAME_FACT.search(str(raw.get("reason") or "")):
            groups.append(sorted({mapped["head"], mapped["tail"]}))
        if _relation_key(mapped) in known:
            reasons["窗口已有"] += 1
            continue
        cleaned = clean_relation(mapped, by_id, source, sentences, inferred_allowed=True, reasons=reasons)
        if not cleaned:
            continue
        known.add(_relation_key(cleaned))
        cleaned["source"] = "整篇综合"
        cleaned["doc_id"] = doc["id"]
        if cleaned["basis"] == "原文明示":
            cleaned["confidence"] = 0.85
        added.append(cleaned)
    for group in parsed.get("duplicates") or []:
        if not isinstance(group, list):
            continue
        members = [back[item] for item in group if isinstance(item, str) and item in back]
        if len(set(members)) >= 2:
            groups.append(sorted(set(members)))
    lines = []
    for line in parsed.get("storylines") or []:
        if not isinstance(line, dict):
            continue
        sequence = [back[item] for item in line.get("events") or [] if isinstance(item, str) and item in back]
        deduped = []
        for item in sequence:
            if item not in deduped:
                deduped.append(item)
        if len(deduped) >= 3:
            lines.append({"doc_id": doc["id"], "title": str(line.get("title") or "").strip() or "主线",
                          "events": deduped})
    return added, groups, lines, 0, reasons
