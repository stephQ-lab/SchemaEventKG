# -*- coding: utf-8 -*-
"""Document-level commodity linking and label fusion.

Runs after chunk extraction and before CSV conversion. One call sees the
whole document and every event already extracted from it. Decisions that
fail the guards are dropped, so a model guess cannot merge different
products or attach a commodity the event never names.
"""
import json
import os

from atlas_rag.kg_construction.commodity import (
    block_reason,
    collapse_mentions,
    outline_core,
    strip_measures,
    classify_event,
    document_categories,
    mentions,
    norm,
)

LINK_INSTRUCTION = """你在做文档级对齐，不是重新抽取，也不是摘要。
全文是事实来源。事件列表是前面按片段抽出的结果。
文档品类已经由标题确定，列在下面。文内的具体品种不是品类。

只返回一个 JSON 对象，不要解释：
{
  "event_links": [
    {"event": "必须与事件列表中的某一条完全一致", "commodity": "只能是文档品类中的一个", "grade": "CONTEXTUAL", "evidence": "从全文原样抄出的一句"}
  ],
  "label_merges": [
    {"keep": "保留的事件原文", "drop": "与 keep 是同一事实的另一种写法"}
  ]
}

规则：
1. 品类只能从给定的文档品类里选。PB粉、电石法PVC 这类品种不要写成品类。
2. 事件文字里已经写出品类或品种的，不要再输出，那些关联会另行写入。
3. CONTEXTUAL：事件文字没点名，但全文里能确定它说的就是该文档品类，evidence 必须是全文原句。
4. 不能确定就不要输出。
5. label_merges 合并同一事实的重复说法：措辞不同、断句不同、从长句截出的短句都要合并。价格和数量的具体数字不是另一个事实。地点、47港和45港、散装和袋装、DOP 和 DOTP、电石法和乙烯法、不同矿种，不要合并。
6. 不要新造事件。
"""


def _events_in(record):
    found = []
    for item in record.get("event_entity_dict") or []:
        if isinstance(item, dict) and isinstance(item.get("Event"), str):
            found.append(item["Event"])
    for item in record.get("event_relation_dict") or []:
        if not isinstance(item, dict):
            continue
        for key in ("Head", "Tail"):
            if isinstance(item.get(key), str):
                found.append(item[key])
    return found


def _replace_label(record, drop, keep):
    for item in record.get("event_entity_dict") or []:
        if isinstance(item, dict) and item.get("Event") == drop:
            item["Event"] = keep
    for item in record.get("event_relation_dict") or []:
        if not isinstance(item, dict):
            continue
        for key in ("Head", "Tail"):
            if item.get(key) == drop:
                item[key] = keep
    for item in record.get("entity_relation_dict") or []:
        if not isinstance(item, dict):
            continue
        for key in ("Head", "Tail"):
            if item.get(key) == drop:
                item[key] = keep


def _attach_commodity(record, event, commodity):
    rows = record.setdefault("event_entity_dict", [])
    for item in rows:
        if isinstance(item, dict) and item.get("Event") == event:
            commodities = item.setdefault("Commodity", [])
            if commodity not in commodities:
                commodities.append(commodity)
            return
    rows.append({"Event": event, "Entity": [], "Commodity": [commodity]})


def _attach_members(record, event, categories, members):
    rows = record.setdefault("event_entity_dict", [])
    target = None
    for item in rows:
        if isinstance(item, dict) and item.get("Event") == event:
            target = item
            break
    if target is None:
        target = {"Event": event, "Entity": []}
        rows.append(target)
    stored = target.setdefault("Commodity", [])
    for category in categories:
        if category not in stored:
            stored.append(category)
    grades = target.setdefault("Member", [])
    for member in members:
        if member not in grades:
            grades.append(member)


def _same_wording(left, right):
    """Same fact once the measurement figures and the comparison words are ignored."""
    a, b = strip_measures(norm(left)), strip_measures(norm(right))
    return bool(a) and a == b and block_reason(left, right) == ""


def contained_duplicate(shorter, longer):
    """True when shorter is the same fact cut out of a longer sentence, not a different one."""
    if _same_wording(shorter, longer):
        return True
    na, nb = strip_measures(norm(shorter)), strip_measures(norm(longer))
    core = strip_measures(outline_core(shorter))
    if core and core != na and len(core) >= 6 and core in nb and block_reason(shorter, longer) == "":
        return True
    if not na or not nb or na == nb or na not in nb:
        return False
    if len(na) < 8 or len(na) / len(nb) < 0.45:
        return False
    return block_reason(shorter, longer) == ""


def _labels_of(record):
    found = []
    for item in record.get("entity_relation_dict") or []:
        if isinstance(item, dict):
            found.extend(item.get(key) for key in ("Head", "Tail") if isinstance(item.get(key), str))
    for item in record.get("event_relation_dict") or []:
        if isinstance(item, dict):
            found.extend(item.get(key) for key in ("Head", "Tail") if isinstance(item.get(key), str))
    for item in record.get("event_entity_dict") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("Event"), str):
            found.append(item["Event"])
        found.extend(entity for entity in item.get("Entity") or [] if isinstance(entity, str))
    return found


def _write_label(record, mapping):
    def rename(value):
        return mapping.get(value, value) if isinstance(value, str) else value

    for item in record.get("entity_relation_dict") or []:
        if isinstance(item, dict):
            for key in ("Head", "Tail"):
                item[key] = rename(item.get(key))
    for item in record.get("event_relation_dict") or []:
        if isinstance(item, dict):
            for key in ("Head", "Tail"):
                item[key] = rename(item.get(key))
    for item in record.get("event_entity_dict") or []:
        if not isinstance(item, dict):
            continue
        item["Event"] = rename(item.get("Event"))
        item["Entity"] = [rename(entity) for entity in item.get("Entity") or [] if rename(entity)]


def collapse_document_labels(records):
    """One name per concept inside a document: same wording, alias or truncated cut."""
    labels = [label for record in records for label in _labels_of(record)]
    mapping = collapse_mentions(labels)
    for record in records:
        _write_label(record, mapping)
    return sum(1 for label in labels if mapping.get(label, label) != label)


def apply_category_links(records, categories):
    """Write 涉及 targets from the article category. Grades stay in Member, not as categories."""
    categories = list(categories or [])
    events = []
    explicit = []
    for record in records:
        record["document_categories"] = list(categories)
        for event in _events_in(record):
            if event not in events:
                events.append(event)
    for event in events:
        chosen, members = classify_event(event, categories)
        if not chosen:
            continue
        for record in records:
            if event in _events_in(record):
                _attach_members(record, event, chosen, members)
        explicit.append({"event": event, "commodity": chosen, "member": members})
    return explicit


def merge_contained_labels(records):
    """Merge a shorter label into a longer one when it is the same fact."""
    events = []
    for record in records:
        for event in _events_in(record):
            if event not in events:
                events.append(event)
    events.sort(key=lambda text: len(norm(text)), reverse=True)
    dropped = set()
    merged = []
    for index, longer in enumerate(events):
        if longer in dropped:
            continue
        for shorter in events[index + 1:]:
            if shorter in dropped or shorter == longer:
                continue
            if contained_duplicate(shorter, longer):
                for record in records:
                    _replace_label(record, shorter, longer)
                dropped.add(shorter)
                merged.append({"keep": longer, "drop": shorter})
    return merged


def accept_link(event, commodity, grade, evidence, source_text):
    if not event or not commodity:
        return False
    if grade == "EXPLICIT":
        return mentions(event, commodity)
    if grade == "CONTEXTUAL":
        evidence_n = norm(evidence)
        return bool(evidence_n) and evidence_n in norm(source_text) and mentions(evidence, commodity)
    return False


def accept_merge(keep, drop):
    if not keep or not drop or norm(keep) == norm(drop):
        return False
    return block_reason(keep, drop) == ""


def apply_document_decisions(records, decisions, source_text, known_events, categories=None):
    """Validate one document's model output and edit that document's records.

    `known_events` maps normalized event text to the exact string stored
    on the records. Links and merges that do not hit those strings are ignored.
    """
    accepted_links = []
    accepted_merges = []
    rejected = []
    by_norm = {norm(ev): ev for ev in known_events}

    for link in decisions.get("event_links") or []:
        if not isinstance(link, dict):
            continue
        event = by_norm.get(norm(link.get("event", "")))
        commodity = (link.get("commodity") or "").strip()
        grade = (link.get("grade") or "").strip().upper()
        evidence = link.get("evidence") or ""
        allowed = categories is None or commodity in categories
        if event and allowed and accept_link(event, commodity, grade, evidence, source_text):
            for record in records:
                _attach_commodity(record, event, commodity)
            accepted_links.append({"event": event, "commodity": commodity, "grade": grade})
        else:
            rejected.append({"kind": "link", "event": link.get("event", ""), "commodity": commodity, "grade": grade})

    for merge in decisions.get("label_merges") or []:
        if not isinstance(merge, dict):
            continue
        keep = by_norm.get(norm(merge.get("keep", "")))
        drop = by_norm.get(norm(merge.get("drop", "")))
        if keep and drop and accept_merge(keep, drop):
            for record in records:
                _replace_label(record, drop, keep)
            by_norm[norm(drop)] = keep
            accepted_merges.append({"keep": keep, "drop": drop})
        else:
            rejected.append({"kind": "merge", "keep": merge.get("keep", ""), "drop": merge.get("drop", "")})

    return accepted_links, accepted_merges, rejected


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extraction_files(data_dir, dataset):
    names = [name for name in os.listdir(data_dir) if dataset in name and name.endswith(".json")]
    return [os.path.join(data_dir, name) for name in sorted(names)]


def link_documents(model, config):
    """Call the model once per document, then validate before editing records."""
    if not config.link_commodities:
        print("document link skipped")
        return
    if config.explicit_document_links:
        return link_explicit_documents(config)
    data_dir = os.path.join(config.output_directory, "kg_extraction")
    doc_path = os.path.join(data_dir, "documents.jsonl")
    documents = {row["id"]: row for row in _read_jsonl(doc_path)}
    files = _extraction_files(data_dir, config.filename_pattern)
    if not documents or not files:
        print("document link found no extraction records")
        return

    grouped = {}
    order = []
    for path in files:
        rows = _read_jsonl(path)
        for index, row in enumerate(rows):
            doc_id = row.get("id")
            grouped.setdefault(doc_id, []).append((path, index, row))
            if doc_id not in order:
                order.append(doc_id)
        # rewritten below after decisions
        _ = rows

    decisions_path = os.path.join(data_dir, "document_link_decisions.jsonl")
    decision_handle = open(decisions_path, "w", encoding="utf-8")
    edited = {}
    for doc_id in order:
        located = grouped.get(doc_id) or []
        records = [item[2] for item in located]
        doc_row = documents.get(doc_id) or {}
        source = doc_row.get("text") or "\n".join(
            record.get("original_text") or "" for record in records
        )
        file_name = ""
        for record in records:
            meta = record.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("file"):
                file_name = meta["file"]
                break
        categories = document_categories(
            doc_id=doc_id,
            title=doc_row.get("title") or "",
            file_name=file_name,
            text=source,
        )
        explicit = apply_category_links(records, categories)
        merges = merge_contained_labels(records)
        collapsed = collapse_document_labels(records)
        links = []
        rejected = []
        try:
            events = []
            for record in records:
                for event in _events_in(record):
                    if event not in events:
                        events.append(event)
            if events and categories:
                user = (
                    LINK_INSTRUCTION
                    + "\n\n[文档品类]\n"
                    + "、".join(categories)
                    + "\n\n[全文]\n"
                    + source
                    + "\n\n[已抽出的事件]\n"
                    + "\n".join("%d. %s" % (i + 1, event) for i, event in enumerate(events))
                )
                messages = [[
                    {"role": "system", "content": "你只返回 JSON 对象。"},
                    {"role": "user", "content": user},
                ]]
                raw = model.generate_response(messages, return_text_only=True, max_workers=config.max_workers)
                text = raw[0] if isinstance(raw, list) else raw
                import json_repair
                parsed = json_repair.loads(text) if isinstance(text, str) else text
                if not isinstance(parsed, dict):
                    parsed = {}
                links, extra_merges, rejected = apply_document_decisions(
                    records, parsed, source, events, categories=categories
                )
                merges.extend(extra_merges)
            for path, index, record in located:
                edited.setdefault(path, {})[index] = record
            decision_handle.write(json.dumps({
                "id": doc_id,
                "categories": categories,
                "explicit": explicit,
                "links": links,
                "merges": merges,
                "collapsed": collapsed,
                "rejected": rejected,
            }, ensure_ascii=False) + "\n")
            print(
                "document link", doc_id,
                "categories", "、".join(categories),
                "explicit", len(explicit),
                "contextual", len(links),
                "merges", len(merges),
                "collapsed", collapsed,
                "rejected", len(rejected),
                flush=True,
            )
        except Exception as exc:
            print("document link failed", doc_id, exc, flush=True)
            for path, index, record in located:
                edited.setdefault(path, {})[index] = record
            decision_handle.write(json.dumps({
                "id": doc_id,
                "categories": categories,
                "explicit": explicit,
                "links": [],
                "merges": merges,
                "rejected": [{"kind": "error", "error": str(exc)}],
            }, ensure_ascii=False) + "\n")
    decision_handle.close()

    for path, replacements in edited.items():
        rows = _read_jsonl(path)
        for index, record in replacements.items():
            rows[index] = record
        with open(path, "w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def link_explicit_documents(config):
    """用文档标题与事件原文完成品类关联和重复事件合并。"""
    data_dir = os.path.join(config.output_directory, "kg_extraction")
    documents = {row["id"]: row for row in _read_jsonl(os.path.join(data_dir, "documents.jsonl"))}
    files = _extraction_files(data_dir, config.filename_pattern)
    if not documents or not files:
        raise ValueError("缺少文档索引或抽取结果")
    by_file = {path: _read_jsonl(path) for path in files}
    grouped = {}
    for path, rows in by_file.items():
        for row in rows:
            grouped.setdefault(row["id"], []).append(row)
    decisions = []
    for doc_id, records in grouped.items():
        document = documents[doc_id]
        metadata = document.get("metadata") or {}
        categories = document_categories(doc_id=doc_id, title=document["title"],
                                         file_name=metadata.get("file", ""))
        explicit = apply_category_links(records, categories)
        merges = merge_contained_labels(records)
        collapsed = collapse_document_labels(records)
        decisions.append({"id": doc_id, "categories": categories,
                          "explicit": explicit, "merges": merges, "collapsed": collapsed})
    for path, rows in by_file.items():
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(data_dir, "document_link_decisions.jsonl"), "w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    print("document link", len(decisions), "documents", flush=True)
