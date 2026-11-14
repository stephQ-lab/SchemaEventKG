# -*- coding: utf-8 -*-
"""Global explanations for one target event, then a consistency pass across targets.

A historical rule can put an event on the candidate list. It is not evidence that
this pair is causal. Two event facts, one for the cause and one for the effect,
are not a relation either. A confirmed cause cites a real evidence id whose text
states the link, and it still loses when the timeline or another document conflicts.

Joint selection does not crown one graph per target and then concatenate the
winners. Every hypothesis edge enters one candidate graph. The same passage
counts once. A multi-cause hypothesis may keep several causes; separate
single-cause hypotheses stay competitors. The joint step adjudicates edges.
It is not a global MAP optimization. Time order, a supply-chain link, and an
abstract rule never confirm an edge by themselves. A trigger confirms a pair
only inside that relation's own arguments: a comma before the trigger can still
introduce the cause, while a comma between two triggers does not cross-pair them.
"""
import json
import re
from collections import Counter
from datetime import date

from eventkg.lexicon import CHAIN, CAUSAL, COMPOSITION, TRIGGERS, category_of, compact
from eventkg.relations import RELATION_WORDS, relation_of_trigger
from eventkg.link import MAX_GAP_DAYS, abstract_rules
from eventkg.llm import parallel_map
from eventkg.paths import CAUSAL_CLASSES
from eventkg.prompts import ABDUCT_SYSTEM

TOP_K = 12
MAX_INDIRECT_NODES = 1
KINDS = ("单一直接原因", "多个共同原因", "间接影响", "共同原因解释", "时间或主题关联", "证据不足")
KIND_ALIAS = {"直接因果": "单一直接原因", "多原因": "多个共同原因", "间接作用": "间接影响"}
CAUSAL_RELATIONS = ("导致", "促进", "抑制", "拖累")
ROLES = ("CAUSE_FACT", "EFFECT_FACT", "RELATION_SUPPORT", "COUNTER_EVIDENCE")
DEFAULT_WEIGHTS = {
    "evidence_support": 0.34,
    "evidence_conflict": 0.18,
    "entity": 0.12,
    "temporal": 0.12,
    "relation": 0.08,
    "graph_consistency": 0.10,
    "parsimony": 0.06,
}
_CHAIN_PAIRS = {(left, right) for left, right in CHAIN} | {(right, left) for left, right in CHAIN}
_MENTIONS = {
    "价格": ("矿价", "价格"),
    "价格指数": ("矿价", "价格指数"),
    "盈利率": ("盈利", "利润"),
    "港口库存": ("库存", "港存"),
    "日均铁水产量": ("铁水",),
    "到港量": ("到港",),
    "发运量": ("发运",),
    "成本": ("成本",),
}
_SOURCE_WEIGHT = {
    "局部关系": 6, "原文点名": 5, "原文共现": 5, "产业链": 4, "混合检索": 4, "语义检索": 4, "BM25": 4, "商品及品类": 3,
    "事件类型": 2, "区域": 2, "规范化实体": 2, "历史抽象机制": 1, "状态变量": 1,
}
_UP = re.compile(r"上升|上涨|增加|回升|推高|抬升")
_DOWN = re.compile(r"下降|下跌|减少|回落|去库")
_COUNTER = re.compile(r"并非|不是因为|无关|被抵消|难以|未证实|没有证实")
_DIRECTION_SKIP = re.compile(r"^[的了着与和及、，,\s]+")


def _dates(documents):
    table = {}
    for doc in documents:
        text = doc.get("report_date") or ""
        if not text:
            continue
        year, month, day = (int(part) for part in text.split("-"))
        table[doc["id"]] = date(year, month, day)
    return table


def report_series_id(document):
    """Separate reports by institution, commodity and frequency.

    An explicit report_series_id wins. A title that does not identify both the
    commodity and the frequency has no series, and relative words such as 上期
    then stay unbound instead of borrowing another report.
    """
    document = document or {}
    explicit = str(document.get("report_series_id") or "").strip()
    if explicit:
        return explicit
    title = " ".join(str(document.get(field) or "") for field in ("title", "file_name", "source_file"))
    if re.search(r"mysteel|钢联", title, re.I):
        institution = "mysteel"
    elif "隆众" in title:
        institution = "longzhong"
    else:
        institution = ""
    if "铁矿" in title:
        commodity = "铁矿石"
    elif re.search(r"PVC|聚氯乙烯", title, re.I):
        commodity = "PVC"
    else:
        commodity = ""
    if "周刊" in title or "周报" in title:
        frequency = "weekly"
    elif "日报" in title:
        frequency = "daily"
    elif "月报" in title:
        frequency = "monthly"
    else:
        frequency = ""
    if not institution or not commodity or not frequency:
        return ""
    return "%s:%s:%s" % (institution, commodity, frequency)


def _series_dates(host_id, documents, dates):
    host = next((doc for doc in documents if doc.get("id") == host_id), {})
    series = report_series_id(host)
    if not series:
        return [], ""
    timeline = [dates[doc["id"]] for doc in documents
                if report_series_id(doc) == series and doc.get("id") in dates]
    return timeline, series


def _commodity(event):
    if event.get("commodity"):
        return event["commodity"]
    return (event.get("var") or "").split("|")[0]


def _indicator(event):
    if event.get("indicator"):
        return event["indicator"]
    parts = (event.get("var") or "").split("|")
    return parts[1] if len(parts) > 1 else ""


def _scope_fits(event, clause):
    """45港 is not 47港, and a scoped sentence is not an unscoped event."""
    scope = _scope(event)
    samples = []
    for match in re.finditer(r"\d+港", clause or ""):
        if (clause or "")[max(0, match.start() - 1):match.start()] in "-/.":
            continue
        samples.append(match.group(0))
    if scope:
        return scope in (clause or "")
    return not samples


def _scope(event):
    if event.get("scope"):
        return event["scope"]
    parts = (event.get("var") or "").split("|")
    return parts[2] if len(parts) > 2 else ""


def _aliases(event):
    names = []
    indicator, commodity = _indicator(event), _commodity(event)
    for value in (indicator, commodity, event.get("subject") or "", event.get("name") or ""):
        if value and value not in names:
            names.append(value)
    for alias in _MENTIONS.get(indicator, ()):
        if alias not in names:
            names.append(alias)
    return [name for name in names if len(name) >= 2]


def _mentions_event(text, event):
    return any(alias in (text or "") for alias in _aliases(event))


def _co_mentioned(event, target, events):
    """Another passage names both events. This only recalls; it does not confirm a cause."""
    for other in events:
        if other.get("id") in (event.get("id"), target.get("id")):
            continue
        text = other.get("evidence") or ""
        if text and _mentions_event(text, event) and _mentions_event(text, target):
            return True
    return False


def _has_trigger(text):
    return any(TRIGGERS[name].search(text or "") for name in ("导致", "促进", "抑制", "拖累"))


def _direction_conflict(text, event):
    """A conflicting occurrence counts only when this event has no matching occurrence.

    The same indicator can fall and then rise in one passage. The rising mention
    must not cancel a relation that uses the falling mention.
    """
    if not text or not event.get("dir"):
        return False
    conflict = False
    aligned = False
    for alias in _aliases(event):
        start = 0
        while True:
            found = text.find(alias, start)
            if found < 0:
                break
            after = _DIRECTION_SKIP.sub("", text[found + len(alias):found + len(alias) + 8])
            if event["dir"] == "上升" and _DOWN.match(after):
                conflict = True
            elif event["dir"] == "下降" and _UP.match(after):
                conflict = True
            elif event["dir"] == "上升" and _UP.match(after):
                aligned = True
            elif event["dir"] == "下降" and _DOWN.match(after):
                aligned = True
            start = found + len(alias)
    return conflict and not aligned


def _related(left, right):
    a, b = _commodity(left), _commodity(right)
    if not a or not b:
        return False
    if a == b or (a, b) in _CHAIN_PAIRS:
        return True
    parent_a, parent_b = category_of(a), category_of(b)
    if parent_a and parent_a == parent_b:
        return True
    return parent_a == b or parent_b == a


def _passage_texts(record):
    """evidence plus every extra quote. The same string is kept once, in order."""
    found = []
    primary = record.get("evidence")
    if isinstance(primary, str) and primary.strip():
        found.append(primary.strip())
    for extra in record.get("evidences") or []:
        if isinstance(extra, str):
            text = extra.strip()
        elif isinstance(extra, dict):
            text = str(extra.get("text") or extra.get("evidence") or "").strip()
        else:
            text = ""
        if text and text not in found:
            found.append(text)
    return found


def _sentence_offset(text, offset):
    return 1 + len(re.findall(r"[。；！？]", (text or "")[:offset]))


def _lookup_place(text, record, documents):
    """Use a real paragraph, sentence and character span when the quote can be found."""
    page = record.get("page") if record.get("page") not in (None, "") else ""
    section = record.get("section") or ""
    if not section:
        sections = record.get("sections") or []
        section = sections[0] if sections else ""
    paragraph_id = record.get("paragraph_id") or record.get("window_id") or ""
    sentence_id = record.get("sentence_id") or ""
    char_start = record.get("char_start") if isinstance(record.get("char_start"), int) else None
    char_end = record.get("char_end") if isinstance(record.get("char_end"), int) else None
    doc = next((item for item in documents if item.get("id") == record.get("doc_id")), None)
    place_status = record.get("place_status") or ""
    if doc is not None and text:
        matches = []
        for window in doc.get("windows") or []:
            body = window.get("text") or ""
            start = 0
            while body and text:
                at = body.find(text, start)
                if at < 0:
                    break
                matches.append((window, at))
                start = at + len(text)
        if len(matches) == 1:
            window, at = matches[0]
            if not paragraph_id:
                paragraph_id = window.get("id") or ""
            if not section:
                names = window.get("sections") or []
                section = names[0] if names else ""
            if char_start is None:
                base = window.get("char_start") or 0
                char_start = base + at
                char_end = char_start + len(text)
            if not sentence_id and window.get("id"):
                sentence_id = "%s:s%d" % (window["id"], _sentence_offset(window.get("text") or "", at))
            place_status = place_status or "UNIQUE"
        elif len(matches) > 1:
            char_start = None
            char_end = None
            place_status = "AMBIGUOUS"
        elif place_status != "AMBIGUOUS" and char_start is None and text:
            narrative = doc.get("narrative") or ""
            if narrative.count(text) == 1:
                at = narrative.find(text)
                char_start = at
                char_end = at + len(text)
                place_status = place_status or "UNIQUE"
            elif narrative.count(text) > 1:
                place_status = "AMBIGUOUS"
    return {"page": page, "section": section, "paragraph_id": paragraph_id,
            "sentence_id": sentence_id, "char_start": char_start, "char_end": char_end,
            "place_status": place_status}


def _evidence_role(text):
    """A denial in one clause does not make the rest of the passage counterevidence."""
    if _has_unblocked_trigger(text or ""):
        return "RELATION_SUPPORT"
    if _COUNTER.search(text or "") or _has_trigger(text or ""):
        return "COUNTER_EVIDENCE"
    return "EFFECT_FACT"


def index_evidence(events, documents, relations):
    """Index event.evidence, event.evidences, relation.evidence and relation.evidences.

    One identical passage is counted once. Every event or relation that quoted it
    stays on that passage, and every evidence id remains retrievable.
    """
    report_dates = {doc["id"]: doc.get("report_date") or "" for doc in documents}
    by_doc = {doc["id"]: doc for doc in documents}
    by_text, items = {}, []

    def add(evidence_id, text, record, source):
        key = compact(text)
        if not key:
            return
        place_for_source = _lookup_place(text, record, documents)
        host_doc = by_doc.get(record.get("doc_id") or "", {})
        source = dict(source, evidence_id=evidence_id,
                      document_id=record.get("doc_id") or "",
                      report_date=report_dates.get(record.get("doc_id") or "", record.get("report_date") or ""),
                      report_series_id=report_series_id(host_doc),
                      source_type=source.get("kind") or "",
                      page=place_for_source["page"], section=place_for_source["section"],
                      paragraph_id=place_for_source["paragraph_id"], sentence_id=place_for_source["sentence_id"],
                      char_start=place_for_source["char_start"], char_end=place_for_source["char_end"],
                      place_status="UNIQUE" if place_for_source["char_start"] is not None else "AMBIGUOUS")
        if key in by_text:
            primary = by_text[key]
            primary["sources"].append(source)
            for event_id in source.get("event_ids") or []:
                if event_id and event_id not in primary["source_event_ids"]:
                    primary["source_event_ids"].append(event_id)
            if evidence_id in primary["evidence_ids"]:
                return
            primary["evidence_ids"].append(evidence_id)
            place = _lookup_place(text, record, documents)
            host = by_doc.get(record.get("doc_id") or "", {})
            shadow = dict(primary, evidence_id=evidence_id, counted=False, canonical_id=primary["evidence_id"],
                          document_id=record.get("doc_id") or primary.get("document_id") or "",
                          report_date=report_dates.get(record.get("doc_id") or "", record.get("report_date") or ""),
                          report_series_id=report_series_id(host),
                          event_time=record.get("event_time") or "",
                          observation_period=record.get("observation_period") or "",
                          page=place["page"], section=place["section"], paragraph_id=place["paragraph_id"],
                          sentence_id=place["sentence_id"], char_start=place["char_start"], char_end=place["char_end"])
            shadow["sources"] = primary["sources"]
            shadow["source_event_ids"] = primary["source_event_ids"]
            shadow["evidence_ids"] = primary["evidence_ids"]
            items.append(shadow)
            return
        place = _lookup_place(text, record, documents)
        host = by_doc.get(record.get("doc_id") or "", {})
        item = {
            "evidence_id": evidence_id, "document_id": record.get("doc_id") or "",
            "report_date": report_dates.get(record.get("doc_id") or "", record.get("report_date") or ""),
            "report_series_id": report_series_id(host),
            "event_time": record.get("event_time") or "",
            "observation_period": record.get("observation_period") or "",
            "page": place["page"], "section": place["section"], "paragraph_id": place["paragraph_id"],
            "sentence_id": place["sentence_id"], "char_start": place["char_start"], "char_end": place["char_end"],
            "text": text, "source_event_ids": [event_id for event_id in source.get("event_ids") or [] if event_id],
            "evidence_role": _evidence_role(text), "sources": [source], "evidence_ids": [evidence_id],
            "counted": True, "canonical_id": evidence_id,
        }
        by_text[key] = item
        items.append(item)

    for event in events:
        texts = _passage_texts(event)
        for index, text in enumerate(texts):
            evidence_id = "ev:" + event["id"] if index == 0 else "ev:%s:%d" % (event["id"], index)
            add(evidence_id, text, event, {
                "kind": "event", "event_id": event["id"], "event_ids": [event["id"]],
                "field": "evidence" if index == 0 else "evidences",
            })
    for index, relation in enumerate(relations):
        texts = _passage_texts(relation)
        head, tail = relation.get("head"), relation.get("tail")
        for offset, text in enumerate(texts):
            evidence_id = "ev:rel:%s:%s:%d" % (head, tail, index) if offset == 0 else (
                "ev:rel:%s:%s:%d:%d" % (head, tail, index, offset))
            add(evidence_id, text, relation, {
                "kind": "relation", "head": head, "tail": tail,
                "event_ids": [head, tail], "field": "evidence" if offset == 0 else "evidences",
            })
            if evidence_id not in relation.setdefault("_indexed_evidence_ids", []):
                relation["_indexed_evidence_ids"].append(evidence_id)
        attached = relation.pop("_indexed_evidence_ids", [])
        causal = relation.get("type") in CAUSAL or relation.get("relation") in CAUSAL
        if attached and not relation.get("evidence_ids") and causal:
            relation["evidence_ids"] = attached
            relation["relation"] = relation.get("relation") or relation.get("type")
            if relation.get("basis", "原文明示") == "原文明示" and not relation.get("relation_class"):
                relation["relation_class"] = "ASSERTED_CAUSE"
    return items


def _event_clock(event, dates):
    """event_time wins. report_date is only a publication clock and never replaces it."""
    stated = _stated_event_time(event)
    published = dates.get(event.get("doc_id"))
    conflict = bool(stated and published and stated != published)
    if stated:
        return stated, "event_time", published, conflict
    if published:
        return published, "report_date", published, False
    return None, "TEMPORAL_UNKNOWN", None, False


def important_targets(events, documents, limit=None):
    """Core events stay eligible without a publication date. Skips are counted."""
    dates = _dates(documents)
    rows, reasons = [], Counter()
    for event in events:
        if event.get("salience") != "核心":
            reasons["不是核心事件"] += 1
            continue
        if event.get("status") != "已发生":
            reasons["状态不是已发生"] += 1
            continue
        if event.get("dir") in (None, "", "不明"):
            reasons["方向不明"] += 1
            continue
        rows.append(event)
    rows.sort(key=lambda event: (_event_clock(event, dates)[0] or date.min, event.get("doc_id") or "", event["id"]))
    dropped = []
    if limit is not None and len(rows) > limit:
        dropped = [event["id"] for event in rows[:-limit]]
        reasons["超出目标数量上限"] += len(dropped)
        rows = rows[-limit:]
    return rows, {"targets": len(rows), "limit": limit, "dropped": dropped,
                  "skipped": int(sum(reasons.values())), "skip_reasons": dict(reasons)}


def _diversify(hits, top_k):
    """Keep the strongest hits without letting one document or one indicator fill the neighborhood."""
    per_doc, per_indicator = {}, {}
    cap = max(2, top_k // 4)
    chosen, rest = [], []
    for item in hits:
        doc_key = item.get("doc_id") or ""
        parts = (item.get("var") or "").split("|")
        indicator = "|".join(parts[:2]) if len(parts) >= 2 else (item.get("var") or "")
        if per_doc.get(doc_key, 0) >= cap or per_indicator.get(indicator, 0) >= cap:
            rest.append(item)
            continue
        chosen.append(item)
        per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
        per_indicator[indicator] = per_indicator.get(indicator, 0) + 1
        if len(chosen) >= top_k:
            return chosen
    for item in rest:
        chosen.append(item)
        if len(chosen) >= top_k:
            break
    return chosen


def retrieve_neighborhood(target, events, documents, relations, rules=None, top_k=TOP_K,
                          retrieval_method="bm25", encoder=None):
    """Recall by entity, commodity, chain, local edges, BM25 and, when configured, dense vectors.

    Character Jaccard is not a channel. Semantic similarity only recalls; it does not prove a cause.
    A different region does not drop a candidate. The existing time window still records a skip.
    """
    if retrieval_method == "jaccard":
        raise RuntimeError("字符 Jaccard 不能进入正式召回。离线对照请使用 eventkg.retrieval.rank_events。")
    from eventkg.retrieval import rank_events
    ranked = rank_events(target, events, method=retrieval_method, encoder=encoder, top_k=max(top_k or 0, 50))
    retrieved = {row["id"]: row for row in ranked["candidates"]}
    dates = _dates(documents)
    target_clock, target_basis, _, _ = _event_clock(target, dates)
    rule_heads = {}
    for rule in (rules or {}).values():
        if rule.get("tail") == target.get("abstract"):
            rule_heads[rule.get("head")] = "抽象机制仅用于召回，不作为本次因果证据"
    local_ids = set()
    for relation in relations:
        if relation.get("type") not in CAUSAL + COMPOSITION:
            continue
        if relation.get("head") == target["id"]:
            local_ids.add(relation.get("tail"))
        elif relation.get("tail") == target["id"]:
            local_ids.add(relation.get("head"))
    hits, skipped = [], []
    for event in events:
        if event["id"] == target["id"]:
            continue
        event_clock, event_basis, published, conflict = _event_clock(event, dates)
        same_basis = event_basis == target_basis and event_basis in ("event_time", "report_date")
        window_gap = (target_clock - event_clock).days if same_basis else None
        outside_window = window_gap is not None and abs(window_gap) > MAX_GAP_DAYS
        reasons = []
        if event["id"] in local_ids:
            reasons.append({"source": "局部关系", "reason": "篇内已经抽出与目标相连的关系"})
        if event.get("abstract") in rule_heads and event["var"] != target["var"]:
            reasons.append({"source": "历史抽象机制", "reason": rule_heads[event["abstract"]]})
        if event["var"] == target["var"]:
            reasons.append({"source": "状态变量", "reason": "同一状态变量的另一次观测，不是因果证据"})
        elif _related(event, target):
            reasons.append({"source": "产业链" if (_commodity(event), _commodity(target)) in _CHAIN_PAIRS
                            or (_commodity(target), _commodity(event)) in _CHAIN_PAIRS else "商品及品类",
                            "reason": "%s与%s属于同一链条或同一品类" % (_commodity(event), _commodity(target))})
        if _mentions_event(target.get("evidence") or "", event) or _mentions_event(event.get("evidence") or "", target):
            reasons.append({"source": "原文点名", "reason": "一侧原文点到了另一侧的商品或指标"})
        if event.get("subject") and event.get("subject") == target.get("subject"):
            reasons.append({"source": "规范化实体", "reason": "主体同为%s" % event["subject"]})
        if event.get("dim") and event.get("dim") == target.get("dim"):
            reasons.append({"source": "事件类型", "reason": "同属%s" % event["dim"]})
        if _scope(event) and _scope(event) == _scope(target):
            reasons.append({"source": "区域", "reason": "口径同为%s" % _scope(event)})
        found = retrieved.get(event["id"])
        if found and event["var"] != target["var"]:
            reasons.append({"source": found["source"], "reason": found["reason"], "score": found.get("score") or 0})
        co_mentioned = _co_mentioned(event, target, events)
        if co_mentioned and event["var"] != target["var"]:
            reasons.append({"source": "原文共现", "reason": "另一条事件原文同时点到了当前事件和目标", "score": 1.0})
        if not reasons:
            skipped.append({"id": event["id"], "reason": "没有实体、品类、事件类型、局部关系或语义重合"})
            continue
        if outside_window:
            reasons.append({"source": "时间衰减", "reason": "间隔超过常见窗口，仍保留为候选，不据此判断因果"})
        event_when = _stated_event_time(event)
        target_when = _stated_event_time(target)
        if event_when and target_when:
            order_gap = (target_when - event_when).days
            time_basis = "event_time"
            if order_gap > 0:
                time_relation, causal_use = "before", "事件时间早于目标，可以进入原因候选"
            elif order_gap < 0:
                time_relation, causal_use = "after", "事件时间晚于目标，不能作为该目标的原因"
            else:
                time_relation, causal_use = "same", "事件时间相同，只能作为共现"
        elif event_basis == "report_date" and target_basis == "report_date":
            order_gap = window_gap
            time_basis = "report_date"
            if order_gap > 0:
                time_relation = "before"
                causal_use = "报告发布时间早于目标，这不是事件发生时间，不能据此判断因果方向"
            elif order_gap < 0:
                time_relation = "after"
                causal_use = "报告发布时间晚于目标，不能作为该目标的原因；这不是事件发生时间，不能据此判断因果方向"
            else:
                time_relation = "same"
                causal_use = "同一报告期，只有发布时间，不能据此判断因果方向"
        else:
            order_gap = None
            time_basis = "TEMPORAL_UNKNOWN"
            time_relation = "TEMPORAL_UNKNOWN"
            causal_use = "时间未知，不能确认因果方向"
        if conflict:
            causal_use = "事件时间与报告发布日期冲突，排序和方向只使用事件时间。" + causal_use
        if event["var"] == target["var"] and order_gap is not None:
            causal_use = "同一指标的后续观测" if order_gap > 0 else "同一指标的另一次观测"
        shown = event_clock or published
        hits.append({
            "id": event["id"], "doc_id": event["doc_id"],
            "date": shown.isoformat() if shown else "",
            "report_date": published.isoformat() if published else "",
            "event_time": event_when.isoformat() if event_when else "",
            "observation_period": event.get("observation_period") or "",
            "time_basis": time_basis, "time_conflict": conflict,
            "name": event.get("name") or event["id"], "var": event["var"], "dir": event.get("dir") or "",
            "status": event.get("status") or "", "evidence": event.get("evidence") or "",
            "evidence_ids": ["ev:" + event["id"]], "time_relation": time_relation,
            "causal_use": causal_use, "gap_days": order_gap if order_gap is not None else None,
            "recall": reasons,
            "retrieval_score": max(float((found or {}).get("score") or 0), 1.0 if co_mentioned else 0.0),
            "rank": sum(_SOURCE_WEIGHT.get(item["source"], 0) for item in reasons) + float((found or {}).get("score") or 0),
        })
    hits.sort(key=lambda item: (
        -item["retrieval_score"],
        -item["rank"],
        abs(item["gap_days"]) if item["gap_days"] is not None else 10 ** 6,
        0 if item["time_relation"] == "before" else 1,
        item["id"],
    ))
    by_source = Counter(reason["source"] for item in hits for reason in item["recall"])
    skip_reasons = Counter(item["reason"] for item in skipped)
    pinned = [item for item in hits if any(reason.get("source") == "局部关系" for reason in item.get("recall") or [])]
    pinned_ids = {item["id"] for item in pinned}
    others = [item for item in hits if item["id"] not in pinned_ids]
    if top_k is None:
        kept, omitted = hits, []
    else:
        room = max(int(top_k) - len(pinned), 0)
        kept = pinned + (_diversify(others, room) if room else [])
        kept_ids = {item["id"] for item in kept}
        omitted = [item for item in hits if item["id"] not in kept_ids]
    return {"candidates": kept, "stats": {
        "hits": len(hits), "kept": len(kept), "dropped_by_topk": len(omitted),
        "omitted_ids": [item["id"] for item in omitted], "by_source": dict(by_source),
        "window_days": MAX_GAP_DAYS, "topk": top_k, "retrieval_method": retrieval_method,
        "skipped": len(skipped), "skipped_ids": [item["id"] for item in skipped],
        "skip_reasons": dict(skip_reasons),
    }}


def _view_event(event, dates):
    when = dates.get(event["doc_id"])
    return {"id": event["id"], "date": when.isoformat() if when else "", "name": event.get("name") or event["id"],
            "var": event["var"], "dir": event.get("dir") or "", "doc_id": event["doc_id"]}


def propose_graphs(chat, target, neighborhood, relations, documents, evidence=None, extra_ids=None):
    """Ask for competing graphs. Evidence ids come from the catalog; an empty graph is allowed."""
    dates = _dates(documents)
    candidates = neighborhood.get("candidates", neighborhood) if isinstance(neighborhood, dict) else list(neighborhood)
    catalog = evidence if evidence is not None else index_evidence([target], documents, relations)
    extra_ids = set(extra_ids or [])
    allowed = {item["id"] for item in candidates}
    local = []
    for relation in relations:
        if relation.get("type") not in CAUSAL + COMPOSITION:
            continue
        ends = {relation.get("head"), relation.get("tail")}
        if target["id"] in ends and ends & allowed:
            local.append({"head": relation.get("head"), "tail": relation.get("tail"), "type": relation.get("type"),
                          "evidence_id": next((item["evidence_id"] for item in catalog
                                               if set(item["source_event_ids"]) == ends), "")})
    payload = {
        "target": _view_event(target, dates),
        "candidates": [{"id": item["id"], "date": item.get("date", ""), "var": item.get("var", ""),
                        "dir": item.get("dir", ""), "time_relation": item.get("time_relation", ""),
                        "causal_use": item.get("causal_use", ""),
                        "recall": item.get("recall", []), "evidence_ids": item.get("evidence_ids", [])}
                       for item in candidates],
        "evidence": [{"evidence_id": item["evidence_id"],
                      "evidence_role": item.get("evidence_role") or item.get("source_type") or "",
                      "document_id": item.get("document_id") or "", "report_date": item.get("report_date") or "",
                      "source_event_ids": item.get("source_event_ids") or [], "text": item["text"][:400]}
                     for item in catalog if item.get("evidence_id") in extra_ids
                     or target["id"] in (item.get("source_event_ids") or [])
                     or allowed & set(item.get("source_event_ids") or [])],
        "local_relations": local,
    }
    parsed = chat.complete_json(ABDUCT_SYSTEM, json.dumps(payload, ensure_ascii=False, sort_keys=True),
                                max_tokens=4096, tag="abduct")
    rows = parsed.get("hypotheses") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    hypotheses = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        kind = KIND_ALIAS.get(str(row.get("kind") or "").strip(), str(row.get("kind") or "").strip())
        if kind not in KINDS:
            kind = "证据不足"
        edges = [] if kind == "证据不足" else row.get("edges") or []
        hypotheses.append({
            "id": str(row.get("id") or ("h%d" % (index + 1))), "kind": kind,
            "nodes": list(row.get("nodes") or []), "why": str(row.get("why") or "").strip(),
            "missing": str(row.get("missing") or "").strip(),
            "needs_retrieval": bool(row.get("needs_retrieval")),
            "retrieval_query": str(row.get("retrieval_query") or row.get("missing") or "").strip(),
            "edges": [edge for edge in edges if isinstance(edge, dict)],
        })
    if not hypotheses:
        hypotheses.append({"id": "h0", "kind": "证据不足", "nodes": [], "why": "模型没有给出可解析的解释",
                           "missing": "", "needs_retrieval": False, "edges": []})
    return hypotheses


def _reversed_between(head, tail, events, dates, evidence_index=None):
    """A reversal counts only when both ends have an explicit event time."""
    del dates
    start, end = resolve_event_time(head, evidence_index), resolve_event_time(tail, evidence_index)
    opposite = {"上升": "下降", "下降": "上升"}.get(tail.get("dir") or "")
    if not start or not end or not opposite:
        return None
    for event in events:
        when = resolve_event_time(event, evidence_index)
        if event["var"] == tail["var"] and event.get("dir") == opposite and when and start < when < end:
            return event["id"]
    return None


_ISO_DATE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_MD_DATE = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})日")
_THIS_PERIOD = re.compile(r"本期|本周|同期")
_PREV_PERIOD = re.compile(r"前期|上期|上周")
_NEXT_PERIOD = re.compile(r"下期")
_CLAUSE_EDGE = re.compile(r"带动|导致|致使|推动|推高|因此|并且|以及|与|[，,；;。！？\n]")
_CONTRAST = re.compile(r"但是|然而|尽管|但|却")
_COORDINATION = re.compile(r"与|和|及|共同|叠加")
_NEGATION_PREFIX = re.compile(r"(?:不一定|未必|不能|没有|并未|并不|不是|难以|未|没)$")
_MODAL_PREFIX = re.compile(r"(?:预计|或许|或将|可能|也许|有望)$")
_CLAUSE_HYPOTHESIS = re.compile(r"^\s*(?:如果|假如|假设|一旦|预计|可能|或许|或将|若(?!干))")
_NOMINAL_EXPECTATION = re.compile(
    r"(?:收紧|放松|松动|上涨|下跌|下降|上升|走弱|走强|增加|减少|回升|回落|提涨).{0,4}(?:预期|预测)$")
_GLUED_CONNECTOR = {"从而", "进而", "继而", "以致", "因此", "所以"}
_SEPARATOR = re.compile(r"[，,]|但是|然而|尽管|但|却")
_BOUNDARY = re.compile(r"[，,；;。！？\n]|但是|然而|尽管|但|却")
_SHOU = re.compile(r"受([^，,；;。！？\n]{0,30}?)(影响|拖累|提振|支撑|带动)")
_TRIGGER_SPECS = (
    ("抑制", "forward", re.compile("|".join(RELATION_WORDS["抑制"]))),
    ("拖累", "forward", re.compile("|".join(RELATION_WORDS["拖累"]))),
    ("促进", "forward", re.compile("|".join(RELATION_WORDS["促进"]))),
    ("导致", "reason", re.compile("因为|由于")),
    ("导致", "forward", re.compile("|".join(
        word for word in RELATION_WORDS["导致"] if word not in ("因为", "由于")))),
)
ALIGN_OK = 0.75


def _specific_aliases(event):
    """Aliases that identify this indicator, without treating every 价格 as the same event."""
    indicator, commodity = _indicator(event), _commodity(event)
    if indicator == "价格" and commodity in ("铁矿石", "铁矿"):
        names = ["矿价", "铁矿石价格"]
    elif indicator == "价格" and commodity:
        names = [commodity + "价格", commodity]
    else:
        names = []
        if indicator:
            names.append(indicator)
        names.extend(_MENTIONS.get(indicator, ()))
    unique = []
    for name in names:
        if name and len(name) >= 2 and name not in unique:
            unique.append(name)
    return unique


def _best_alias(event, text):
    found = [alias for alias in _specific_aliases(event) if alias in (text or "")]
    if not found:
        return ""
    return max(found, key=len)


def _explicit_dates(text, host):
    found = []
    for year, month, day in _ISO_DATE.findall(text or ""):
        found.append(date(int(year), int(month), int(day)))
    if host is not None:
        for month, day in _MD_DATE.findall(text or ""):
            try:
                found.append(date(host.year, int(month), int(day)))
            except ValueError:
                continue
    return found


def _period_word(text):
    for pattern in (_PREV_PERIOD, _NEXT_PERIOD, _THIS_PERIOD):
        match = pattern.search(text or "")
        if match:
            return match.group(0)
    return ""


_GAP_LIMIT = {"daily": 2, "weekly": 10, "monthly": 40}
_SKIP_PERIOD = re.compile(r"延期|跳期|顺延")


def _series_frequency(series):
    series = series or ""
    if series.endswith("daily"):
        return "daily"
    if series.endswith("weekly"):
        return "weekly"
    if series.endswith("monthly"):
        return "monthly"
    return ""


def _neighbor_date(host, timeline, step, frequency="", allow_skip=False):
    """Next report in the series. A missing issue is not the next period."""
    if host is None:
        return None
    ordered = sorted(set(timeline) | {host})
    index = ordered.index(host) + step
    if not (0 <= index < len(ordered)):
        return None
    candidate = ordered[index]
    if frequency and not allow_skip:
        if abs((candidate - host).days) > _GAP_LIMIT[frequency]:
            return None
    return candidate


def _locks_for_clause(clause, host, timeline, frequency="", allow_skip=False):
    """Explicit dates win. A relative period never falls back to the host event."""
    explicit = _explicit_dates(clause, host)
    word = _period_word(clause)
    if explicit:
        return explicit, word
    if word in ("本期", "本周", "同期"):
        return ([host] if host is not None else None), word
    if word in ("前期", "上期", "上周"):
        previous = _neighbor_date(host, timeline, -1, frequency, allow_skip)
        return ([previous] if previous is not None else None), word
    if word == "下期":
        nxt = _neighbor_date(host, timeline, 1, frequency, allow_skip)
        return ([nxt] if nxt is not None else None), word
    return [], ""


def _time_expression(text, host):
    parts = ["%04d-%02d-%02d" % (item.year, item.month, item.day) for item in _explicit_dates(text, host)]
    word = _period_word(text)
    if word:
        parts.append(word)
    unique = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return " ".join(unique)


def _stated_direction(text, alias, start=None):
    if start is None:
        start = (text or "").find(alias)
    if start is None or start < 0:
        return ""
    after = _DIRECTION_SKIP.sub("", text[start + len(alias):start + len(alias) + 8])
    if _UP.match(after):
        return "上升"
    if _DOWN.match(after):
        return "下降"
    return ""


def _clause_bounds_sentence(text, at):
    left, right = 0, len(text or "")
    for match in re.finditer(r"[；;。！？\n]", text or ""):
        if match.end() <= at:
            left = match.end()
        elif match.start() >= at:
            right = match.start()
            break
    return left, right


def _clause_bounds(text, start, end):
    left, right = 0, len(text or "")
    for match in _CLAUSE_EDGE.finditer(text or ""):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return left, right


def _clause_of(text, alias, at=None):
    """Dates apply only to the clause that names this occurrence."""
    if at is None:
        at = (text or "").find(alias)
    if at < 0:
        return text or ""
    left, right = _clause_bounds(text, at, at + len(alias))
    return text[left:right]


def _clause_event_time(clause):
    """An explicit calendar date in this clause. A report publication date is not copied in."""
    found = _explicit_dates(clause, None)
    if len(found) != 1:
        return ""
    return "%04d-%02d-%02d" % (found[0].year, found[0].month, found[0].day)


def _stated_event_time(event):
    """Event time comes from event_time or a date beside the event's own name."""
    parsed = _parse_iso(event.get("event_time") or "")
    if parsed:
        return parsed
    text = event.get("evidence") or ""
    alias = _best_alias(event, text) if text else ""
    if not alias:
        return None
    return _parse_iso(_clause_event_time(_clause_of(text, alias)))


def _parse_iso(value):
    match = _ISO_DATE.search(value or "")
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _mention_occurrences(text, events):
    """Every non-overlapping occurrence. The same indicator can appear more than once."""
    hits = []
    for event in events:
        for alias in _specific_aliases(event):
            start = 0
            while True:
                at = text.find(alias, start)
                if at < 0:
                    break
                stated = _stated_direction(text, alias, at)
                if not (stated and event.get("dir") and stated != event.get("dir")):
                    hits.append((at, at + len(alias), alias, event))
                start = at + len(alias)
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0]), -len(item[2])))
    occurrences = []
    for start, end, alias, event in hits:
        current = next((kept for kept in occurrences if kept["char_start"] == start and kept["char_end"] == end), None)
        if current is not None:
            if event["id"] not in [item["id"] for item in current["events"]]:
                current["events"].append(event)
            continue
        if any(not (end <= kept["char_start"] or start >= kept["char_end"]) for kept in occurrences):
            continue
        occurrences.append({"char_start": start, "char_end": end, "alias": alias, "events": [event]})
    return occurrences


def _mentions_for(item, events, documents):
    """One mention per occurrence. No date, or more than one matching event, leaves the mention unlinked."""
    dates = _dates(documents)
    host = dates.get(item.get("document_id"))
    timeline, series = _series_dates(item.get("document_id"), documents, dates)
    text = item.get("text") or ""
    mentions = []
    for occurrence in _mention_occurrences(text, events):
        alias = occurrence["alias"]
        fresh = occurrence["events"]
        clause = _clause_of(text, alias, occurrence["char_start"])
        sentence_left, sentence_right = _clause_bounds_sentence(text, occurrence["char_start"])
        allow_skip = bool(_SKIP_PERIOD.search(text[sentence_left:sentence_right]))
        locks, _word = _locks_for_clause(clause, host, timeline, _series_frequency(series), allow_skip)
        if locks is None or not locks:
            chosen, confidence = None, 0.2
        else:
            dated = [event for event in fresh if dates.get(event["doc_id"]) in locks and _scope_fits(event, clause)]
            if len(dated) == 1:
                chosen, confidence = dated[0], 0.9
            elif len(dated) > 1:
                chosen, confidence = None, 0.25
            else:
                chosen, confidence = None, 0.3 if len(fresh) == 1 else 0.2
        sample = chosen or fresh[0]
        number = re.search(r"\d+(?:\.\d+)?%?", clause)
        base_sentence = item.get("sentence_id") or item["evidence_id"]
        mentions.append({
            "mention_id": "m:%s:%s:%d" % (item["evidence_id"], alias, occurrence["char_start"]),
            "document_id": item.get("document_id") or "",
            "sentence_id": "%s:s%d" % (base_sentence, _sentence_offset(text, occurrence["char_start"])),
            "paragraph_id": item.get("paragraph_id") or "",
            "page": item.get("page") or "",
            "section": item.get("section") or "",
            "char_start": occurrence["char_start"],
            "char_end": occurrence["char_end"],
            "source_text": text,
            "clause_text": clause,
            "entity": sample.get("subject") or sample.get("commodity") or "",
            "indicator": _indicator(fresh[0]),
            "commodity": _commodity(fresh[0]),
            "time_expression": _time_expression(clause, host),
            "event_direction": _stated_direction(text, alias, occurrence["char_start"]),
            "value": number.group(0) if number else "",
            "period": _period_word(clause),
            "report_date": item.get("report_date") or "",
            "report_series_id": series,
            "event_time": _clause_event_time(clause) or item.get("event_time") or "",
            "observation_period": item.get("observation_period") or _period_word(clause),
            "linked_event_id": chosen["id"] if chosen else "",
            "alignment_confidence": confidence,
            "candidate_event_ids": [event["id"] for event in fresh],
        })
    return mentions


def attach_mentions(catalog, events, documents):
    if not catalog or all("mentions" in item for item in catalog):
        return catalog
    for item in catalog:
        item["mentions"] = _mentions_for(item, events, documents)
    return catalog


def _mention_link(item, event_id):
    for mention in item.get("mentions") or []:
        if mention.get("linked_event_id") == event_id and mention.get("alignment_confidence", 0) >= ALIGN_OK:
            return mention
    return None


def _sentence_spans(text):
    """Split on sentence punctuation only. A comma can still belong to one causal clause."""
    spans, start = [], 0
    for match in re.finditer(r"[；;。！？\n]", text or ""):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text or ""):
        spans.append((start, len(text)))
    return spans


def _triggers_in(text, start, end):
    found = []
    chunk = text[start:end]
    for match in _SHOU.finditer(chunk):
        found.append({
            "start": start + match.start(), "end": start + match.end(),
            "inner_start": start + match.start() + 1,
            "inner_end": start + match.end() - len(match.group(2)),
            "text": match.group(0),
            "polarity": "抑制" if match.group(2) == "拖累" else "导致",
            "order": "shou",
        })
    for polarity, order, pattern in _TRIGGER_SPECS:
        for match in pattern.finditer(chunk):
            found.append({"start": start + match.start(), "end": start + match.end(),
                          "inner_start": start + match.start(), "inner_end": start + match.end(),
                          "text": match.group(0), "polarity": polarity, "order": order})
    found.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    kept = []
    for item in found:
        if any(not (item["end"] <= other["start"] or item["start"] >= other["end"]) for other in kept):
            continue
        kept.append(item)
    kept.sort(key=lambda item: item["start"])
    return kept


def _between(mentions, start, end):
    return [mention for mention in mentions
            if start <= mention.get("char_start", -1) and mention.get("char_end", -1) <= end]


def _comma_before_next_mention(text, start, end, mentions):
    """A comma ends this trigger's effect only when another mention follows it."""
    for match in re.finditer(r"[，,]", text[start:end]):
        after = start + match.end()
        if any(after <= mention["char_start"] and mention["char_end"] <= end for mention in mentions):
            return start + match.start()
    return None


def _proposition_blocked(text, trigger, causes, effects):
    """Negation and prediction bind the causal proposition, not every 预期 in the sentence.

    预期/预测 after a content word is the event itself, as in 成本预期松动 or
    流动性收紧预期. 预计、可能、不能、未必 operate on the trigger.
    """
    prefix = text[max(0, trigger["start"] - 6):trigger["start"]]
    if _NEGATION_PREFIX.search(prefix) or _MODAL_PREFIX.search(prefix):
        return True
    if prefix.endswith("预期") or prefix.endswith("预测"):
        nominal = text[max(0, trigger["start"] - 12):trigger["start"]]
        if not _NOMINAL_EXPECTATION.search(nominal):
            return True
    left_edge = min([trigger["start"]] + [mention["char_start"] for mention in causes + effects]) if (
        causes or effects) else trigger["start"]
    left = 0
    for match in _BOUNDARY.finditer(text[:left_edge]):
        left = match.end()
    if _CLAUSE_HYPOTHESIS.search(text[left:trigger["start"]]):
        return True
    right_edge = max([trigger["end"]] + [mention["char_end"] for mention in causes + effects]) if (
        causes or effects) else trigger["end"]
    return bool(_COUNTER.search(text[left:right_edge]))


def _trigger_blocked(text, trigger, left=None, right=None):
    del left, right
    return _proposition_blocked(text, trigger, [], [])


def _has_unblocked_trigger(text):
    for start, end in _sentence_spans(text):
        for trigger in _triggers_in(text, start, end):
            if not _trigger_blocked(text, trigger):
                return True
    return False


def _pair_blocked(text, trigger, causes, effects):
    return _proposition_blocked(text, trigger, causes, effects)


def _strip_glued_connectors(text, triggers, mentions):
    """进而导致 is one construction. The connector is not a separate cause."""
    kept = []
    for index, trigger in enumerate(triggers):
        if trigger["text"] in _GLUED_CONNECTOR and index + 1 < len(triggers):
            nxt = triggers[index + 1]
            if not _between(mentions, trigger["end"], nxt["start"]):
                continue
        kept.append(trigger)
    return kept


def _separators(text, start, end):
    return list(_SEPARATOR.finditer(text[start:end]))


def _direct_causes(text, mentions, left_from, trigger_start, banned):
    """The mention touching the trigger. One empty comma may step back once.

    Earlier mentions stay background. 价格下跌，成本推升，导致盈利 keeps 成本
    and does not also take 价格.
    """
    found = _separators(text, left_from, trigger_start)
    if not found:
        return [mention for mention in _between(mentions, left_from, trigger_start)
                if mention["mention_id"] not in banned]
    last = found[-1]
    own = [mention for mention in _between(mentions, left_from + last.end(), trigger_start)
           if mention["mention_id"] not in banned]
    if own:
        return own
    if last.group(0) in ("但", "却", "但是", "然而", "尽管"):
        return []
    prev_end = left_from
    if len(found) >= 2:
        prev_end = left_from + found[-2].end()
    return [mention for mention in _between(mentions, prev_end, left_from + last.start())
            if mention["mention_id"] not in banned]


def _relation_pairs(text, mentions):
    """Direct arguments sit against the trigger. Background mentions do not.

    'A下降，导致B下降' steps back across that one comma. 'A与B共同推高C' keeps
    both causes. 'A导致B，随后B导致C' stays a chain. A single intermediate
    mention becomes two direct edges, not a shortcut across it.
    """
    pairs = []
    for sent_start, sent_end in _sentence_spans(text):
        local = [mention for mention in mentions if sent_start <= mention.get("char_start", -1) < sent_end]
        triggers = _strip_glued_connectors(text, _triggers_in(text, sent_start, sent_end), local)
        anchor = sent_start
        pending = None
        for index, trigger in enumerate(triggers):
            left_from = anchor
            right_to = triggers[index + 1]["start"] if index + 1 < len(triggers) else sent_end
            has_next = index + 1 < len(triggers)
            if trigger["order"] == "shou":
                causes = _between(local, trigger["inner_start"], trigger["inner_end"])
                effects = _between(local, trigger["end"], right_to)[:1]
            elif trigger["order"] == "reason":
                before = _between(local, left_from, trigger["start"])
                cut = _comma_before_next_mention(text, trigger["end"], right_to, local)
                if before:
                    effects = before[-1:]
                    causes = _between(local, trigger["end"], cut if cut is not None else right_to)[:1] or _between(
                        local, trigger["end"], right_to)[:1]
                elif cut is not None:
                    causes = _between(local, trigger["end"], cut)
                    effect_end = right_to
                    if has_next:
                        later = _comma_before_next_mention(text, cut + 1, right_to, local)
                        if later is not None:
                            effect_end = later
                    effects = _between(local, cut + 1, effect_end)[:1]
                else:
                    causes, effects = [], []
            else:
                effect_end = right_to
                if has_next:
                    cut = _comma_before_next_mention(text, trigger["end"], right_to, local)
                    if cut is not None:
                        effect_end = cut
                effects = _between(local, trigger["end"], effect_end)
                causes = _direct_causes(text, local, left_from, trigger["start"], set())
                if causes and not effects and has_next:
                    # 推升 in 成本推升，导致盈利 names the cause. It is not its own edge.
                    continue
            if effects:
                anchor = trigger["end"]
            if trigger["order"] == "shou" and causes and not effects:
                pending = (causes, trigger)
            elif causes and effects:
                pending = None
            blocked = _pair_blocked(text, trigger, causes, effects) if causes and effects else False
            role = "direct"
            emitted = False
            for cause in causes:
                for effect in effects:
                    if cause["mention_id"] == effect["mention_id"]:
                        continue
                    emitted = True
                    pairs.append({"cause": cause, "effect": effect, "trigger": trigger["text"],
                                  "polarity": trigger["polarity"],
                                  "mood": blocked or _effect_is_forecast(text, trigger, effect),
                                  "role": role, "missing_intermediate": ""})
            if pending and trigger["order"] != "shou" and effects and not causes:
                missing = text[pending[1]["end"]:trigger["start"]].strip("，,；; ")
                for cause in pending[0]:
                    for effect in effects:
                        pairs.append({"cause": cause, "effect": effect, "trigger": trigger["text"],
                                      "polarity": trigger["polarity"], "mood": blocked, "role": "indirect",
                                      "missing_intermediate": missing})
                pending = None
            elif emitted:
                pending = None
    return pairs


def _effect_is_forecast(text, trigger, effect):
    """预期/预测 between the trigger and this result governs that result, not the cause."""
    span = text[trigger["end"]:effect.get("char_start", trigger["end"])]
    return bool(re.search(r"预期|预测", span))


def _accepts_polarity(relation, polarity):
    if relation in ("抑制", "压制", "限制"):
        return polarity == "抑制"
    if relation in ("拖累", "挤压", "削弱"):
        return polarity == "拖累"
    if relation in ("促进", "推动", "带动", "推高", "推升"):
        return polarity == "促进"
    if relation in ("导致", "造成", "致使"):
        return polarity == "导致"
    return False


def _linked_mentions(item, event_id):
    return [mention for mention in item.get("mentions") or []
            if mention.get("linked_event_id") == event_id and mention.get("alignment_confidence", 0) >= ALIGN_OK]


def _compose_bundle(items, head, tail, relation):
    """What each passage supports. A trigger in another clause does not support this pair."""
    cause, effect, relation_ids, temporal, counter = [], [], [], [], []
    propositions, alignments = [], []
    mechanism_only = bool(items)
    alignment_failed = False
    scope_failed = False
    for item in _unique_passages(items):
        text = item.get("text") or ""
        if item.get("evidence_role") == "COUNTER_EVIDENCE":
            counter.append(item["evidence_id"])
            propositions.append({"evidence_id": item["evidence_id"], "supports": "反向或冲突证据"})
            continue
        if _historical_rule(text):
            propositions.append({"evidence_id": item["evidence_id"], "supports": "历史抽象规律不能作为这条关系的证据"})
            continue
        if not str(item.get("evidence_id") or "").startswith("ev:rel:"):
            mechanism_only = False
        head_mentions = _linked_mentions(item, head["id"])
        tail_mentions = _linked_mentions(item, tail["id"])
        for mention in head_mentions:
            alignments.append(mention["alignment_confidence"])
            if item["evidence_id"] not in cause:
                cause.append(item["evidence_id"])
            propositions.append({"evidence_id": item["evidence_id"], "mention_id": mention["mention_id"],
                                 "alignment_confidence": mention["alignment_confidence"],
                                 "supports": "原因事件实例 %s" % head["id"]})
        for mention in tail_mentions:
            alignments.append(mention["alignment_confidence"])
            if item["evidence_id"] not in effect:
                effect.append(item["evidence_id"])
            propositions.append({"evidence_id": item["evidence_id"], "mention_id": mention["mention_id"],
                                 "alignment_confidence": mention["alignment_confidence"],
                                 "supports": "结果事件实例 %s" % tail["id"]})
        pairs = _relation_pairs(text, item.get("mentions") or [])
        aligned = [pair for pair in pairs
                   if pair["cause"]["linked_event_id"] == head["id"] and pair["effect"]["linked_event_id"] == tail["id"]
                   and pair["cause"]["alignment_confidence"] >= ALIGN_OK and pair["effect"]["alignment_confidence"] >= ALIGN_OK
                   and _accepts_polarity(relation, pair["polarity"])]
        matched = [pair for pair in aligned if not pair["mood"] and pair.get("role") != "indirect"]
        indirect = [pair for pair in aligned if not pair["mood"] and pair.get("role") == "indirect"]
        middles = []
        for left in pairs:
            if left.get("mood") or left.get("role") == "indirect" or left["cause"].get("linked_event_id") != head["id"]:
                continue
            for right in pairs:
                if right.get("mood") or right.get("role") == "indirect":
                    continue
                if right["effect"].get("linked_event_id") == tail["id"] and right["cause"].get("linked_event_id") == left["effect"].get("linked_event_id"):
                    middles.append(left["effect"].get("linked_event_id") or left["effect"].get("mention_id"))
        if matched:
            relation_ids.append(item["evidence_id"])
            propositions.append({"evidence_id": item["evidence_id"], "trigger": matched[0]["trigger"],
                                 "cause_mention": matched[0]["cause"]["mention_id"],
                                 "effect_mention": matched[0]["effect"]["mention_id"],
                                 "supports": "文档主张这两个提及之间存在因果关系"})
        elif middles:
            propositions.append({"evidence_id": item["evidence_id"],
                                 "supports": "中间事件 %s 把两端分开，不建立直接因果边" % "、".join(middles)})
        elif indirect:
            missing = indirect[0].get("missing_intermediate") or ""
            propositions.append({"evidence_id": item["evidence_id"], "missing_intermediate": missing,
                                 "supports": "间接影响候选，中间事件未抽取：%s" % missing})
        elif any(pair["cause"]["linked_event_id"] == head["id"] and pair["effect"]["linked_event_id"] == tail["id"]
                 and pair["mood"] for pair in pairs):
            scope_failed = True
            propositions.append({"evidence_id": item["evidence_id"],
                                 "supports": "否定、转折、假设或预测语气，不作为已发生因果"})
        elif head_mentions and tail_mentions and _has_trigger(text):
            scope_failed = True
            propositions.append({"evidence_id": item["evidence_id"],
                                 "supports": "因果触发词不在这两个提及的同一关系分句"})
        elif _has_trigger(text) and head["id"] in _candidate_ids(item) and tail["id"] in _candidate_ids(item):
            alignment_failed = True
            propositions.append({"evidence_id": item["evidence_id"],
                                 "supports": "文本提到了两端，但无法确认是哪一次事件实例"})
        if head_mentions or tail_mentions:
            mention = (head_mentions or tail_mentions)[0]
            temporal.append({"evidence_id": item["evidence_id"], "time_expression": mention.get("time_expression") or ""})
    missing = [item.get("missing_intermediate") or item.get("supports") or ""
               for item in propositions if "中间事件" in (item.get("supports") or "") or item.get("missing_intermediate")]
    return {"cause_instance": cause, "effect_instance": effect, "relation_support": relation_ids,
            "temporal_alignment": temporal, "counter": counter, "propositions": propositions,
            "alignment": min(alignments) if alignments else 0,
            "mechanism_only": bool(mechanism_only and not relation_ids),
            "alignment_failed": alignment_failed, "scope_failed": scope_failed,
            "missing_intermediates": [item for item in missing if item]}


def _candidate_ids(item):
    found = []
    for mention in item.get("mentions") or []:
        for event_id in mention.get("candidate_event_ids") or []:
            if event_id not in found:
                found.append(event_id)
        if mention.get("linked_event_id") and mention["linked_event_id"] not in found:
            found.append(mention["linked_event_id"])
    return found


def _model_score(relation_class, components):
    """An uncalibrated ranking score. It is not a probability."""
    base = {"ASSERTED_CAUSE": 0.74, "INFERRED_SUPPORTED": 0.61, "POSSIBLE_CAUSE": 0.36,
            "POSSIBLE_INDIRECT_CAUSE": 0.4,
            "NONE": 0.12, "NEXT_OBSERVATION": 0.2, "REVERSES": 0.18, "FORECASTS": 0.16,
            "SAME_EVENT": 0.2, "CONTINUES": 0.2}.get(relation_class, 0.2)
    score = (base + 0.1 * components.get("alignment", 0) + 0.06 * components.get("temporal", 0)
             - 0.28 * components.get("evidence_conflict", 0) - 0.12 * components.get("mechanism_only", 0))
    return round(min(0.92, max(0.05, score)), 3)


def _full_match(text, head, tail):
    return _has_trigger(text) and _mentions_event(text, head) and _mentions_event(text, tail) and not (
        _direction_conflict(text, head) or _direction_conflict(text, tail))


def _unique_passages(items):
    """The same text is one passage, even when two evidence ids repeat it."""
    unique, seen = [], set()
    for item in items:
        key = compact(item.get("text") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _inferred_passages(items, head, tail):
    """Several documents can support one implicit cause. Pasting a trigger from another pair cannot.

    One passage must tie a causal trigger to the cause event. Another passage must
    name the effect. Two copies of one sentence count once, and two facts with no
    trigger on the cause stay unsupported.
    """
    useful = [item for item in _unique_passages(items)
              if _mentions_event(item.get("text") or "", head) or _mentions_event(item.get("text") or "", tail)]
    documents = {item.get("document_id") or "" for item in useful}
    documents.discard("")
    if len(useful) < 2 or len(documents) < 2:
        return []
    if any(_direction_conflict(item.get("text") or "", head) or _direction_conflict(item.get("text") or "", tail)
           for item in useful):
        return []
    cause_tied = any(_has_trigger(item.get("text") or "") and _mentions_event(item.get("text") or "", head)
                     and not _direction_conflict(item.get("text") or "", head) for item in useful)
    effect_named = any(_mentions_event(item.get("text") or "", tail)
                       and not _direction_conflict(item.get("text") or "", tail) for item in useful)
    if not (cause_tied and effect_named):
        return []
    return useful


def _independent_count(evidence_ids, by_evidence):
    seen = set()
    for evidence_id in evidence_ids:
        item = by_evidence.get(evidence_id)
        if item is None:
            continue
        seen.add(compact(item.get("text") or ""))
    seen.discard("")
    return len(seen)


def resolve_event_time(event, evidence_index=None):
    """Event time from the node, then an aligned mention, then one dated instance passage.

    report_date is never used.
    """
    parsed = _parse_iso((event or {}).get("event_time") or "")
    if parsed:
        return parsed
    items = []
    if isinstance(evidence_index, dict):
        items = list(evidence_index.values())
    elif evidence_index:
        items = list(evidence_index)
    mention_times = []
    for item in items:
        for mention in item.get("mentions") or []:
            if mention.get("linked_event_id") != event.get("id"):
                continue
            if mention.get("alignment_confidence", 0) < ALIGN_OK:
                continue
            found = _parse_iso(mention.get("event_time") or "")
            if found and found not in mention_times:
                mention_times.append(found)
    if len(mention_times) == 1:
        return mention_times[0]
    if len(mention_times) > 1:
        return None
    owned = []
    event_id = event.get("id")
    for item in items:
        owners = set(item.get("source_event_ids") or [])
        if event_id not in owners and item.get("evidence_id") != "ev:" + str(event_id):
            continue
        text = item.get("text") or ""
        alias = _best_alias(event, text) if text else ""
        if not alias:
            continue
        found = _parse_iso(_clause_event_time(_clause_of(text, alias)))
        if found and found not in owned:
            owned.append(found)
    if len(owned) == 1:
        return owned[0]
    stated = _stated_event_time(event)
    return stated


def _observed_time(event, by_evidence):
    """Same clock as resolve_event_time. Kept so older calls stay consistent."""
    return resolve_event_time(event, by_evidence)


def _historical_rule(text):
    return bool(re.search(r"历史抽象|历史机制|历史规律|历史句式", text or ""))


_TRANSPORT_WORDS = ("发运", "到港", "运输", "航线", "海运", "疏港", "物流")


def _transport_bridge(left, right):
    """Different regions can connect when the events themselves name a transport step."""
    names = "".join(str(item.get(key) or "") for item in (left, right) for key in ("indicator", "name", "var"))
    roles = {}
    for item in (left, right):
        roles.update(item.get("roles") or {})
    route = any(word in names for word in _TRANSPORT_WORDS) or bool(
        roles.get("transport_route") or roles.get("origin_region") or roles.get("destination_region"))
    commodity_ok = _related(left, right) or (_commodity(left) and _commodity(left) == _commodity(right))
    return bool(route and commodity_ok)


def _transport_text(left, right, blob):
    """A transport quote supports this pair only inside one clause that names both events."""
    left_name, right_name = left.get("indicator") or "", right.get("indicator") or ""
    if len(left_name) < 2 or len(right_name) < 2:
        return False
    related = _related(left, right) or (_commodity(left) and _commodity(left) == _commodity(right))
    if not related:
        return False
    for part in re.split(r"[。；;]", blob or ""):
        if left_name in part and right_name in part and any(word in part for word in _TRANSPORT_WORDS):
            return True
    return False


def _pair_issues(left, right, evidence_index=None):
    """Compatibility of one hop. Different regions are not incompatible by themselves."""
    issues = []
    start, end = resolve_event_time(left, evidence_index), resolve_event_time(right, evidence_index)
    if start and end and not start < end:
        issues.append("时间不兼容")
    if _scope(left) and _scope(right) and _scope(left) != _scope(right) and not _transport_bridge(left, right):
        issues.append("不同地区且没有运输机制")
    if left.get("subject") and right.get("subject") and left["subject"] != right["subject"] and not _related(left, right):
        issues.append("主体不一致")
    if _commodity(left) and _commodity(right) and not _related(left, right):
        issues.append("品类不兼容")
    if left.get("var") and left.get("var") == right.get("var"):
        issues.append("状态变量相同，不是中间因果")
    return issues


def _relation_label(pair):
    polarity = pair.get("polarity") or ""
    if polarity in ("导致", "促进", "抑制", "拖累"):
        return polarity
    return relation_of_trigger(pair.get("trigger") or "")


def _segment_relation(item, left, right):
    labels = []
    for pair in _relation_pairs(item.get("text") or "", item.get("mentions") or []):
        if pair.get("mood") or pair.get("role") == "indirect":
            continue
        cause, effect = pair.get("cause") or {}, pair.get("effect") or {}
        if cause.get("linked_event_id") != left.get("id") or effect.get("linked_event_id") != right.get("id"):
            continue
        if cause.get("alignment_confidence", 0) < ALIGN_OK or effect.get("alignment_confidence", 0) < ALIGN_OK:
            continue
        labels.append(_relation_label(pair))
    unique = list(dict.fromkeys(labels))
    if len(unique) != 1:
        return ""
    return unique[0]


def _hop_evidence(cited_items, left, right, evidence_index=None):
    """One hop, with its own relation type. Time order is not evidence."""
    support_ids, counters, labels = [], [], []
    for item in cited_items:
        text = item.get("text") or ""
        evidence_id = item.get("evidence_id") or ""
        if _historical_rule(text):
            continue
        if item.get("evidence_role") == "COUNTER_EVIDENCE" and (
                _mentions_event(text, left) or _mentions_event(text, right)):
            counters.append(evidence_id)
            continue
        label = _segment_relation(item, left, right)
        if not label:
            continue
        support_ids.append(evidence_id)
        labels.append(label)
    unique = list(dict.fromkeys(labels))
    relation = unique[0] if len(unique) == 1 and not counters else ""
    if len(unique) != 1:
        support_ids = []
    start = resolve_event_time(left, evidence_index)
    end = resolve_event_time(right, evidence_index)
    temporal = bool(start and end and start < end and support_ids and relation)
    return {"head": left.get("id"), "tail": right.get("id"), "relation": relation,
            "evidence_ids": support_ids, "counter_evidence_ids": [item for item in counters if item],
            "relation_support": list(support_ids) if relation else [],
            "temporal_compatibility": temporal}


def _other_factors(cited_items, head, middle, tail, events, evidence_index=None):
    by_id = {event["id"]: event for event in events}
    found = []
    for item in cited_items:
        if _historical_rule(item.get("text") or ""):
            continue
        for event_id in item.get("source_event_ids") or []:
            if event_id in (head["id"], middle["id"], tail["id"]) or event_id in found:
                continue
            other = by_id.get(event_id)
            if other is None:
                continue
            hop = _hop_evidence([item], other, tail, evidence_index)
            if hop["relation_support"]:
                found.append(event_id)
    return found


def _event_cited(event, cited_items):
    marker = "ev:" + event["id"]
    for item in cited_items:
        if event["id"] in (item.get("source_event_ids") or []):
            return True
        if item.get("evidence_id") == marker or marker in (item.get("evidence_ids") or []):
            return True
    return False


def _candidate_middles(cited_items, head, tail, events, evidence_index):
    start, end = resolve_event_time(head, evidence_index), resolve_event_time(tail, evidence_index)
    if not start or not end or not start < end:
        return []
    found = []
    for event in events:
        if event["id"] in (head["id"], tail["id"]) or event.get("var") in (head.get("var"), tail.get("var")):
            continue
        when = resolve_event_time(event, evidence_index)
        if not when or not start < when < end or not _event_cited(event, cited_items):
            continue
        found.append(event)
    return found


def _path_record(head, tail, nodes, segments):
    missing, blocked = [], []
    for segment in segments:
        if not segment["relation_support"]:
            missing.append("%s→%s 缺少桥接证据" % (segment["head"], segment["tail"]))
        elif not segment["temporal_compatibility"]:
            missing.append("%s→%s 时间不兼容" % (segment["head"], segment["tail"]))
        blocked.extend(segment.get("counter_evidence_ids") or [])
    supported = bool(nodes and segments and not missing and not blocked
                     and all(segment["relation"] for segment in segments))
    return {"middle": nodes[0]["id"] if len(nodes) == 1 else "",
            "middles": [node["id"] for node in nodes],
            "segments": segments,
            "left_evidence": list(segments[0]["evidence_ids"]) if segments else [],
            "right_evidence": list(segments[-1]["evidence_ids"]) if segments else [],
            "missing": missing, "blocked": blocked, "supported": supported}


def find_indirect_paths(cited_items, head, tail, events, evidence_index=None, max_intermediate_nodes=None):
    """Bounded hop search. The default stays one intermediate and does not widen itself."""
    if max_intermediate_nodes is None:
        max_intermediate_nodes = MAX_INDIRECT_NODES
    max_intermediate_nodes = int(max_intermediate_nodes)
    if max_intermediate_nodes < 1:
        return []
    pool = _candidate_middles(cited_items, head, tail, events, evidence_index)
    if not pool:
        return []
    end = resolve_event_time(tail, evidence_index)
    paths = []

    def walk(current, nodes, segments):
        if len(nodes) > max_intermediate_nodes:
            return
        if nodes:
            closing = _hop_evidence(cited_items, current, tail, evidence_index)
            issues = _pair_issues(current, tail, evidence_index)
            if (closing["relation_support"] and closing["temporal_compatibility"]
                    and not issues and not closing["counter_evidence_ids"]):
                paths.append(_path_record(head, tail, nodes, segments + [closing]))
        if len(nodes) >= max_intermediate_nodes:
            return
        current_when = resolve_event_time(current, evidence_index)
        for event in pool:
            if event["id"] in {node["id"] for node in nodes}:
                continue
            when = resolve_event_time(event, evidence_index)
            hop = _hop_evidence(cited_items, current, event, evidence_index)
            issues = _pair_issues(current, event, evidence_index)
            if (not hop["relation_support"] or not hop["temporal_compatibility"]
                    or issues or hop["counter_evidence_ids"]):
                continue
            if not (current_when and when and end and current_when < when < end):
                continue
            walk(event, nodes + [event], segments + [hop])

    walk(head, [], [])
    return [path for path in paths if path["supported"]]


def _indirect_path(cited_items, head, tail, events, evidence_index=None, max_intermediate_nodes=None):
    """The first evidenced path, or a note when a timed middle has no bridge."""
    found = find_indirect_paths(cited_items, head, tail, events, evidence_index, max_intermediate_nodes)
    if found:
        return found[0]
    middles = _candidate_middles(cited_items, head, tail, events, evidence_index)
    if not middles:
        return None
    missing, blocked = [], []
    for event in middles:
        left = _hop_evidence(cited_items, head, event, evidence_index)
        right = _hop_evidence(cited_items, event, tail, evidence_index)
        if not left["relation_support"]:
            missing.append("%s→%s 缺少桥接证据" % (head["id"], event["id"]))
        if not right["relation_support"]:
            missing.append("%s→%s 缺少桥接证据" % (event["id"], tail["id"]))
        missing.extend(_pair_issues(head, event, evidence_index))
        missing.extend(_pair_issues(event, tail, evidence_index))
        blocked.extend(left["counter_evidence_ids"] + right["counter_evidence_ids"])
        others = _other_factors(cited_items, head, event, tail, events, evidence_index)
        if others:
            missing.append("其他因素：" + "、".join(others))
    return {"middle": middles[0]["id"], "middles": [event["id"] for event in middles], "segments": [],
            "left_evidence": [], "right_evidence": [], "missing": missing, "blocked": blocked,
            "supported": False}


def _score_edge(edge, target, allowed, by_id, by_evidence, dates, events):
    head_id = str(edge.get("head") or "").strip()
    tail_id = str(edge.get("tail") or "").strip()
    relation = str(edge.get("relation") or "").strip()
    reasons, support_ids, fact_ids, counter_ids = [], [], [], []
    blank = {"evidence_support": 0, "evidence_conflict": 0, "entity": 0, "temporal": 0,
             "relation": 0, "independent_passages": 0, "alignment": 0, "mechanism_only": 0}
    if head_id not in allowed or tail_id not in allowed or head_id not in by_id or tail_id not in by_id:
        return _edge_row(edge, "NONE", blank, ["端点不在候选集"], [], False)
    head, tail = by_id[head_id], by_id[tail_id]
    if relation not in CAUSAL_RELATIONS:
        reasons.append("关系类型不在因果词表")
    head_date, tail_date = dates.get(head["doc_id"]), dates.get(tail["doc_id"])
    publication_gap = (tail_date - head_date).days if head_date and tail_date else None
    components = dict(blank)
    components["relation"] = 1 if relation in CAUSAL_RELATIONS else 0
    if head["var"] == tail["var"]:
        kind = "REVERSES" if {head.get("dir"), tail.get("dir")} == {"上升", "下降"} else "NEXT_OBSERVATION"
        reasons.append("同一指标的观测，不作为因果")
        return _edge_row(edge, kind, components, reasons, [], False, publication_gap)
    if head.get("status") == "预测中" or tail.get("status") == "预测中":
        if head.get("status") == "预测中" and tail.get("status") == "预测中":
            reasons.append("两端都还在预测中，不记为已发生因果")
        elif tail.get("status") == "预测中":
            reasons.append("已发生的原因指向预测中的结果，不记为已发生因果")
        else:
            reasons.append("预测不是已经发生的原因")
        return _edge_row(edge, "FORECASTS", components, reasons, [], False, publication_gap)
    head_when, tail_when = resolve_event_time(head, by_evidence), resolve_event_time(tail, by_evidence)
    if head_when and tail_when:
        gap = (tail_when - head_when).days
        components["time_basis"] = "event_time"
        reversal = _reversed_between(head, tail, events, dates, by_evidence)
        if gap <= 0:
            components["temporal"] = 0
            reasons.append("时间不兼容")
        elif reversal:
            components["temporal"] = 0
            components["evidence_conflict"] = 1
            counter_ids.append("ev:" + reversal)
            reasons.append("中间出现反向观测")
        elif head["doc_id"] == tail["doc_id"]:
            components["temporal"] = 0.5
            reasons.append("同篇关系不升级为跨文档因果")
        else:
            components["temporal"] = 1
    elif head["doc_id"] == tail["doc_id"]:
        gap = publication_gap
        components["time_basis"] = "report_date"
        components["temporal"] = 0.5
        reasons.append("同篇关系不升级为跨文档因果")
    else:
        gap = publication_gap
        components["time_basis"] = "report_date"
        components["temporal"] = 0.5
        reasons.append("事件实际时间未知，不根据报告发布时间确定因果方向")
    left_scope, right_scope = _scope(head), _scope(tail)
    scope_differs = bool(left_scope and right_scope and left_scope != right_scope)
    if _related(head, tail) or head.get("subject") and head.get("subject") == tail.get("subject"):
        components["entity"] = 1
    else:
        components["entity"] = 0.5
    cited = [str(item) for item in (edge.get("evidence_ids") or [])]
    if not cited and edge.get("evidence"):
        reasons.append("只给了引用原文，没有证据ID")
    missing_ids = [evidence_id for evidence_id in cited if evidence_id not in by_evidence]
    for evidence_id in missing_ids:
        reasons.append("证据不存在:%s" % evidence_id)
    for evidence_id in [str(item) for item in (edge.get("counter_evidence_ids") or [])]:
        if evidence_id not in by_evidence:
            reasons.append("反证不存在:%s" % evidence_id)
        else:
            components["evidence_conflict"] = 1
            counter_ids.append(evidence_id)
    cited_items = [by_evidence[item] for item in cited if item in by_evidence]
    if scope_differs:
        texts = [(item.get("text") or "") for item in cited_items]
        texts.append(edge.get("why") or "")
        if _transport_bridge(head, tail) or any(_transport_text(head, tail, text) for text in texts):
            components["entity"] = 1
            reasons.append("不同地区，运输关系支持跨地区联系")
        else:
            components["entity"] = 0
            reasons.append("不同地区且没有运输机制")
    bundle = _compose_bundle(cited_items, head, tail, relation)
    components["evidence_bundle"] = bundle
    components["alignment"] = bundle["alignment"]
    components["mechanism_only"] = 1 if bundle["mechanism_only"] else 0
    if bundle["counter"]:
        components["evidence_conflict"] = 1
        for evidence_id in bundle["counter"]:
            if evidence_id not in counter_ids:
                counter_ids.append(evidence_id)
        reasons.append("证据与方向冲突")
    if bundle.get("scope_failed"):
        reasons.append("因果触发词不在这两个提及的同一关系分句")
    if bundle["alignment_failed"]:
        reasons.append("文本提及无法对齐到这两个事件实例")
    host_docs = {head["doc_id"], tail["doc_id"]}
    host_relation = [evidence_id for evidence_id in bundle["relation_support"]
                     if by_evidence[evidence_id].get("document_id") in host_docs]
    third_relation = [evidence_id for evidence_id in bundle["relation_support"] if evidence_id not in host_relation]
    relation_texts = {compact(by_evidence[evidence_id]["text"]) for evidence_id in bundle["relation_support"]}
    extra_instance = []
    for evidence_id in bundle["cause_instance"] + bundle["effect_instance"]:
        if compact(by_evidence[evidence_id]["text"]) not in relation_texts and evidence_id not in extra_instance:
            extra_instance.append(evidence_id)
    for note in bundle.get("missing_intermediates") or []:
        if note not in reasons:
            reasons.append(note)
    known_conflict = any(item in "；".join(reasons) for item in ("时间不兼容", "中间出现反向观测"))
    unknown_time = "事件实际时间未知" in "；".join(reasons)
    asserted_time_ok = components["temporal"] == 1 or (components["temporal"] == 0.5 and unknown_time)
    inferred_time_ok = components["temporal"] == 1
    base_ready = components["evidence_conflict"] == 0 and components["entity"] > 0 and components["relation"] == 1 and not known_conflict
    if host_relation and base_ready and asserted_time_ok and not bundle["mechanism_only"]:
        components["evidence_support"] = 1
        components["inference_path"] = "asserted"
        relation_class = "ASSERTED_CAUSE"
        support_ids = host_relation
        confirmed = True
    elif third_relation and extra_instance and base_ready and inferred_time_ok and not bundle["mechanism_only"]:
        components["evidence_support"] = 1
        components["inference_path"] = "explicit_third_document"
        relation_class = "INFERRED_SUPPORTED"
        support_ids = list(dict.fromkeys(third_relation + extra_instance))
        confirmed = True
    elif cited_items or bundle["cause_instance"] or bundle["effect_instance"]:
        components["evidence_support"] = 0.4
        relation_class = "POSSIBLE_CAUSE"
        fact_ids = [item["evidence_id"] for item in cited_items if item["evidence_id"] not in support_ids]
        triggered = any(_has_trigger(item.get("text") or "") for item in cited_items)
        if bundle.get("scope_failed"):
            reasons.append("因果触发词不覆盖这两个事件提及，保留为候选")
        elif bundle["alignment_failed"] or (triggered and not bundle["relation_support"]):
            reasons.append("无法确认对应的是同一次具体事件，保留为候选")
        else:
            reasons.append("缺少能同时覆盖两端且已对齐到事件实例的关系支持证据，保留为候选")
        confirmed = False
    else:
        relation_class = "NONE"
        reasons.append("缺少关系支持证据")
        confirmed = False
    if head["doc_id"] == tail["doc_id"] and confirmed:
        confirmed = False
        relation_class = "POSSIBLE_CAUSE"
        components["evidence_support"] = 0.4
        reasons.append("同篇关系不升级为跨文档因果")
    if tail_id != target["id"] and confirmed:
        reasons.append("边未指向目标，留在图里接受整体检查")
    components["independent_passages"] = _independent_count(support_ids or fact_ids, by_evidence)
    from eventkg.support import verify_relation
    verdict = verify_relation({
        "head": head_id, "tail": tail_id, "relation": relation,
        "evidence_ids": cited,
        "counter_evidence_ids": [str(item) for item in (edge.get("counter_evidence_ids") or [])],
    }, events, by_evidence)
    components["verification"] = verdict
    vstatus = verdict.get("verification_status")
    if vstatus in ("FORECASTS", "DISPUTED"):
        relation_class = vstatus
        confirmed = False
        if verdict.get("reason") and verdict["reason"] not in reasons:
            reasons.append(verdict["reason"])
    elif vstatus != "VERIFIED":
        if confirmed:
            confirmed = False
            relation_class = "POSSIBLE_CAUSE"
            note = verdict.get("reason") or "统一验证未通过"
            if note not in reasons:
                reasons.append(note)
    return _edge_row(edge, relation_class, components, reasons, support_ids, confirmed, gap, fact_ids, counter_ids)


def _edge_row(edge, relation_class, components, reasons, support_ids, confirmed, gap=None, fact_ids=None, counter_ids=None):
    score = _model_score(relation_class, components)
    return {"head": str(edge.get("head") or ""), "tail": str(edge.get("tail") or ""),
            "relation": str(edge.get("relation") or ""), "relation_class": relation_class,
            "evidence_ids": list(edge.get("evidence_ids") or []), "support_ids": support_ids,
            "fact_ids": fact_ids or [], "counter_evidence_ids": counter_ids or [],
            "why": str(edge.get("why") or ""), "missing": str(edge.get("missing") or ""),
            "needs_retrieval": bool(edge.get("needs_retrieval")), "components": components,
            "evidence_bundle": components.get("evidence_bundle") or {},
            "edge_score": score, "model_score": score, "score_kind": "model_score",
            "confidence": score, "reasons": reasons, "confirmed": confirmed, "gap_days": gap}


def _graph_components(rows, by_evidence, by_id, weights):
    if not rows:
        return {"evidence_support": 0, "evidence_conflict": 0, "entity": 0, "temporal": 0, "relation": 0,
                "graph_consistency": 1, "parsimony": 0, "shared_evidence": [], "total": 0, "ready": False}
    use = Counter(evidence_id for row in rows for evidence_id in row["support_ids"])
    shared = []
    for evidence_id, count in use.items():
        if count < 2:
            continue
        item = by_evidence.get(evidence_id)
        covered = [row for row in rows if evidence_id in row["support_ids"]]
        if item and all(_mentions_event(item["text"], by_id[row["head"]]) and _mentions_event(item["text"], by_id[row["tail"]])
                        for row in covered if row["head"] in by_id and row["tail"] in by_id):
            shared.append({"evidence_id": evidence_id, "edges": len(covered), "independent": False,
                           "reason": "同一句同时点名这些端点"})
        else:
            shared.append({"evidence_id": evidence_id, "edges": len(covered), "independent": False,
                           "reason": "同一证据被重复当作独立证据"})
            for row in covered[1:]:
                row["components"]["evidence_support"] = min(row["components"]["evidence_support"], 0.4)
                row["confirmed"] = False
                row["relation_class"] = "POSSIBLE_CAUSE"
                row["reasons"].append("与其他边共用证据，且该证据没有同时点名这条边的两端")
    pair = Counter((row["head"], row["tail"], row["relation"]) for row in rows)
    duplicate = [key for key, count in pair.items() if count > 1]
    conflict = max(row["components"]["evidence_conflict"] for row in rows)
    opposite = {}
    for row in rows:
        opposite.setdefault((row["head"], row["tail"]), set()).add(row["relation"])
    contradictory = [pair for pair, kinds in opposite.items() if {"导致", "抑制"} <= kinds or {"促进", "抑制"} <= kinds]
    nodes, forward = set(), {}
    for row in rows:
        forward.setdefault(row["head"], []).append(row["tail"])
        nodes.update((row["head"], row["tail"]))
    cyclic = _has_cycle(forward)
    consistent = not duplicate and not contradictory and not cyclic
    weakest = min(row["components"]["evidence_support"] for row in rows)
    unsupported = sum(1 - row["components"]["evidence_support"] for row in rows) / float(len(rows))
    parts = {
        "evidence_support": round(weakest, 3),
        "evidence_conflict": round(conflict, 3),
        "entity": round(min(row["components"]["entity"] for row in rows), 3),
        "temporal": round(min(row["components"]["temporal"] for row in rows), 3),
        "relation": round(min(row["components"]["relation"] for row in rows), 3),
        "graph_consistency": 1 if consistent else 0,
        "parsimony": round(unsupported, 3),
        "shared_evidence": shared,
        "duplicate_edges": duplicate,
        "contradictory_pairs": contradictory,
    }
    total = (weights["evidence_support"] * weakest
             - weights["evidence_conflict"] * conflict
             + weights["entity"] * parts["entity"]
             + weights["temporal"] * parts["temporal"]
             + weights["relation"] * parts["relation"]
             + weights["graph_consistency"] * parts["graph_consistency"]
             - weights["parsimony"] * unsupported)
    confirmed = [row for row in rows if row["confirmed"] and row["relation_class"] in CAUSAL_CLASSES]
    parts["total"] = round(total, 3)
    parts["ready"] = bool(confirmed) and conflict == 0 and consistent
    return parts


def _has_cycle(forward):
    seen, stack = set(), set()

    def visit(node):
        if node in stack:
            return True
        if node in seen:
            return False
        seen.add(node)
        stack.add(node)
        for nxt in forward.get(node, []):
            if visit(nxt):
                return True
        stack.remove(node)
        return False

    return any(visit(node) for node in list(forward))


def _aligned_fact(event_id, by_evidence, relation_ids):
    relation_texts = {compact((by_evidence.get(item) or {}).get("text") or "") for item in relation_ids}
    relation_texts.discard("")
    ordered = []
    preferred = "ev:" + event_id
    if preferred in by_evidence:
        ordered.append(preferred)
    for evidence_id, item in by_evidence.items():
        if event_id in (item.get("source_event_ids") or []) and evidence_id not in ordered:
            ordered.append(evidence_id)
    for evidence_id in ordered:
        item = by_evidence[evidence_id]
        if compact(item.get("text") or "") in relation_texts:
            continue
        if any(mention.get("linked_event_id") == event_id and mention.get("alignment_confidence", 0) >= ALIGN_OK
               for mention in item.get("mentions") or []):
            return evidence_id
    return ""


def _score_segment(segment, target, allowed, by_id, by_evidence, dates, events):
    fact = _aligned_fact(segment["head"], by_evidence, segment["evidence_ids"])
    edge = {"head": segment["head"], "tail": segment["tail"], "relation": segment["relation"],
            "evidence_ids": list(dict.fromkeys(list(segment["evidence_ids"]) + ([fact] if fact else []))),
            "counter_evidence_ids": list(segment.get("counter_evidence_ids") or []),
            "why": "分段中间关系"}
    hop = _score_edge(edge, target, set(allowed) | set(segment.get("middles") or []) | {segment["tail"], segment["head"]},
                      by_id, by_evidence, dates, events)
    hop["components"]["segment"] = {
        "head": segment["head"], "tail": segment["tail"], "relation": segment["relation"],
        "evidence_ids": list(segment["evidence_ids"]),
        "counter_evidence_ids": list(segment.get("counter_evidence_ids") or []),
        "relation_support": list(segment.get("relation_support") or []),
        "temporal_compatibility": segment.get("temporal_compatibility"),
    }
    return hop


def _materialize_indirect(rows, target, allowed, by_id, by_evidence, dates, events, max_intermediate_nodes=None):
    """Keep each evidenced hop with its own relation. Do not add a direct A→B edge."""
    extra = []
    for row in rows:
        if row.get("confirmed"):
            continue
        head, tail = by_id.get(row["head"]), by_id.get(row["tail"])
        if head is None or tail is None:
            continue
        cited = [by_evidence[item] for item in row.get("evidence_ids") or [] if item in by_evidence]
        paths = find_indirect_paths(cited, head, tail, events, by_evidence, max_intermediate_nodes)
        note = "历史抽象规律不能作为中间关系的证据"
        if note not in row["reasons"]:
            row["reasons"].append(note)
        kept_paths = []
        for path in paths:
            hops = []
            middles = set(path["middles"])
            for segment in path["segments"]:
                segment = dict(segment, middles=path["middles"])
                hop = _score_segment(segment, target, set(allowed) | middles, by_id, by_evidence, dates, events)
                hops.append(hop)
            if hops and all(hop.get("confirmed") and hop.get("relation_class") in CAUSAL_CLASSES for hop in hops):
                kept_paths.append((path, hops))
        row["components"]["indirect_paths"] = [{
            "middles": path["middles"],
            "segments": path["segments"],
            "missing": path["missing"],
        } for path, _hops in kept_paths] or []
        if kept_paths:
            row["confirmed"] = False
            row["relation_class"] = "NONE"
            row["components"]["evidence_support"] = 0
            described = "、".join("→".join([row["head"]] + path["middles"] + [row["tail"]]) for path, _hops in kept_paths)
            row["reasons"].append("中间路径 %s 已有分段证据，不建立直接因果边" % described)
            for path, hops in kept_paths:
                for hop in hops:
                    hop["components"]["inference_path"] = "implicit_segment"
                    hop["reasons"].append("分段关系有独立证据，不合并成直接因果边")
                    if note not in hop["reasons"]:
                        hop["reasons"].append(note)
                    extra.append(hop)
            continue
        if paths:
            row["confirmed"] = False
            row["relation_class"] = "POSSIBLE_INDIRECT_CAUSE"
            row["components"]["evidence_support"] = 0.4
            row["reasons"].append("分段关系没有全部通过校验，不保留这条间接路径")
            continue
        fallback = _indirect_path(cited, head, tail, events, by_evidence, max_intermediate_nodes)
        if fallback is None:
            continue
        row["components"]["indirect_path"] = {
            "middle": fallback.get("middle") or "", "middles": fallback.get("middles") or [],
            "segments": fallback.get("segments") or [],
            "left_evidence": fallback.get("left_evidence") or [],
            "right_evidence": fallback.get("right_evidence") or [],
            "missing": fallback.get("missing") or [],
        }
        row["confirmed"] = False
        row["relation_class"] = "POSSIBLE_INDIRECT_CAUSE"
        row["components"]["evidence_support"] = 0.4
        row["reasons"].append("中间事件没有证据证明它连接了两端，不能确认因果")
        for item in fallback.get("missing") or []:
            if item not in row["reasons"]:
                row["reasons"].append(item)
        if fallback.get("blocked"):
            row["reasons"].append("中间关系存在反证")
    seen = {(row["head"], row["relation"], row["tail"]) for row in rows}
    for hop in extra:
        key = (hop["head"], hop["relation"], hop["tail"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(hop)
    return rows


def select_graph(target, hypotheses, neighborhood, events, documents, evidence=None, weights=None,
                 max_intermediate_nodes=None):
    """Edge scores and a graph score. No confirmed edge means NO_SUPPORTED_EXPLANATION."""
    weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    dates = _dates(documents)
    by_id = {event["id"]: event for event in events}
    by_id[target["id"]] = target
    catalog = evidence if evidence is not None else index_evidence(events, documents, [])
    attach_mentions(catalog, events, documents)
    by_evidence = {item["evidence_id"]: item for item in catalog}
    candidates = neighborhood.get("candidates", neighborhood) if isinstance(neighborhood, dict) else list(neighborhood)
    allowed = {item["id"] for item in candidates} | {target["id"]}
    scored = []
    for hypothesis in hypotheses:
        kind = hypothesis.get("kind") or "证据不足"
        proposed = [] if kind == "证据不足" else list(hypothesis.get("edges") or [])
        rows = [_score_edge(edge, target, allowed, by_id, by_evidence, dates, events) for edge in proposed]
        rows = _materialize_indirect(rows, target, allowed, by_id, by_evidence, dates, events,
                                     max_intermediate_nodes)
        graph = _graph_components(rows, by_evidence, by_id, weights)
        reasons = ["%s→%s：%s" % (row["head"], row["tail"], "；".join(row["reasons"]))
                   for row in rows if row["reasons"]]
        if kind == "证据不足":
            reasons.append(hypothesis.get("why") or "证据不足")
        if graph["ready"] and not any(row["tail"] == target["id"] and row["confirmed"] for row in rows):
            graph["ready"] = False
            graph["graph_consistency"] = 0
            reasons.append("已确认的边没有解释目标事件")
        scored.append({
            "id": hypothesis.get("id") or "", "kind": kind, "why": hypothesis.get("why") or "",
            "missing": hypothesis.get("missing") or "", "needs_retrieval": bool(hypothesis.get("needs_retrieval")),
            "nodes": hypothesis.get("nodes") or [], "edge_scores": rows, "graph_scores": graph,
            "reasons": reasons, "ready": graph["ready"],
        })
    ready = [item for item in scored if item["ready"]]
    ready.sort(key=lambda item: (-item["graph_scores"]["total"], len(item["edge_scores"]), item["id"]))
    chosen = ready[0] if ready else None
    if chosen:
        emitted = [_public_edge(row, target, chosen) for row in chosen["edge_scores"] if row["confirmed"]]
        decision, reason = chosen["id"], ""
    else:
        emitted = []
        decision, reason = "NO_SUPPORTED_EXPLANATION", "所有解释都缺少足够的关系支持证据"
        if scored:
            best = max(scored, key=lambda item: (item["graph_scores"]["total"], -len(item["edge_scores"]), item["id"]))
            detail = "；".join(best["reasons"])
            if detail:
                reason += "：" + detail
    possible = []
    pool = chosen["edge_scores"] if chosen else [row for item in scored for row in item["edge_scores"]]
    for row in pool:
        if not row["confirmed"] and row["relation_class"] in ("POSSIBLE_CAUSE", "POSSIBLE_INDIRECT_CAUSE"):
            possible.append(_public_edge(row, target, chosen or {"id": "", "kind": ""}))
    return {"target": target["id"], "candidates": candidates, "hypotheses": scored, "weights": weights,
            "selected": None if chosen is None else chosen["id"], "decision": decision,
            "rejected": chosen is None, "rejection_reason": reason, "edges": emitted, "possible_edges": possible}


def _public_edge(row, target, hypothesis):
    return {"head": row["head"], "tail": row["tail"], "type": "跨期传导", "relation": row["relation"],
            "relation_class": row["relation_class"], "evidence_ids": row["support_ids"] or row["evidence_ids"],
            "counter_evidence_ids": row["counter_evidence_ids"], "why": row["why"], "missing": row["missing"],
            "gap_days": row["gap_days"], "reason": row["why"] or "；".join(row["reasons"]),
            "confidence": row["edge_score"], "model_score": row.get("model_score", row["edge_score"]),
            "score_kind": "model_score", "edge_score": row["edge_score"], "components": row["components"],
            "evidence_bundle": row.get("evidence_bundle") or {},
            "source": "溯因", "hypothesis": hypothesis.get("id", ""), "hypothesis_kind": hypothesis.get("kind", ""),
            "target": target["id"]}


def select_global(records, evidence, events):
    """Merge target-level graphs. The same sentence is not two independent proofs, and conflicts drop both."""
    by_id = {event["id"]: event for event in events}
    by_evidence = {item["evidence_id"]: item for item in evidence}
    grouped = {}
    possible, conflicts = [], []
    for record in records:
        for edge in record["edges"]:
            key = (edge["head"], edge["relation"], edge["tail"])
            grouped.setdefault(key, []).append(edge)
        possible.extend(record.get("possible_edges") or [])
    confirmed = []
    evidence_use = Counter()
    for key, edges in grouped.items():
        best = max(edges, key=lambda edge: edge.get("edge_score") or 0)
        best = dict(best, targets=sorted({edge["target"] for edge in edges}))
        for evidence_id in best.get("evidence_ids") or []:
            evidence_use[evidence_id] += 1
        confirmed.append(best)
    use_counts = Counter(evidence_id for edge in confirmed for evidence_id in edge.get("evidence_ids") or [])
    kept = []
    for edge in confirmed:
        shared = [evidence_id for evidence_id in edge.get("evidence_ids") or [] if use_counts[evidence_id] > 1]
        item_ok = True
        for evidence_id in shared:
            item = by_evidence.get(evidence_id)
            if item is None or edge["head"] not in by_id or edge["tail"] not in by_id:
                item_ok = False
            elif not (_mentions_event(item["text"], by_id[edge["head"]]) and _mentions_event(item["text"], by_id[edge["tail"]])):
                item_ok = False
        if shared and not item_ok:
            edge = dict(edge, relation_class="POSSIBLE_CAUSE",
                        reason=(edge.get("reason") or "") + "；同一证据不能重复证明另一条边")
            possible.append(edge)
        else:
            kept.append(edge)
    pair_kinds = {}
    for edge in kept:
        pair_kinds.setdefault((edge["head"], edge["tail"]), []).append(edge)
    final = []
    for pair, edges in pair_kinds.items():
        kinds = {edge["relation"] for edge in edges}
        if {"导致", "抑制"} <= kinds or {"促进", "抑制"} <= kinds:
            conflicts.append({"pair": list(pair), "relations": sorted(kinds), "reason": "不同目标对同一对事件给出了冲突关系"})
            possible.extend(edges)
        else:
            final.append(max(edges, key=lambda edge: edge.get("edge_score") or 0))
    linked = {edge["head"] for edge in final} | {edge["tail"] for edge in final}
    unlinked = [record["target"] for record in records if record["target"] not in linked]
    return {"confirmed": final, "possible": possible, "conflicts": conflicts,
            "evidence_use": dict(evidence_use), "unlinked_targets": unlinked}


def mode_name(multi_source=False, joint=False):
    if joint and multi_source:
        return "D"
    if joint:
        return "C"
    if multi_source:
        return "B"
    return "A"


def build_candidate_graph(records, evidence):
    """Every hypothesis edge, with its own evidence and local status. Nothing is chosen yet."""
    by_evidence = {item["evidence_id"]: item for item in evidence}
    grouped = {}
    for record in records:
        for hypothesis in record.get("hypotheses") or []:
            for row in hypothesis.get("edge_scores") or []:
                key = (row["head"], row["relation"], row["tail"])
                bucket = grouped.setdefault(key, {
                    "head": row["head"], "tail": row["tail"], "relation": row["relation"], "proposals": [],
                })
                bucket["proposals"].append({
                    "target": record["target"], "hypothesis": hypothesis.get("id") or "",
                    "kind": hypothesis.get("kind") or "", "why": row.get("why") or hypothesis.get("why") or "",
                    "relation_class": row.get("relation_class"), "confirmed": row.get("confirmed"),
                    "evidence_ids": list(row.get("evidence_ids") or []),
                    "edge_score": row.get("edge_score"), "components": row.get("components") or {},
                    "reasons": list(row.get("reasons") or []),
                })
    rows = []
    for bucket in grouped.values():
        seen, sources = set(), []
        for proposal in bucket["proposals"]:
            for evidence_id in proposal["evidence_ids"]:
                item = by_evidence.get(evidence_id)
                text_key = compact(item["text"]) if item else evidence_id
                counted = text_key not in seen
                if counted:
                    seen.add(text_key)
                sources.append({
                    "evidence_id": evidence_id,
                    "document_id": item.get("document_id", "") if item else "",
                    "evidence_role": item.get("evidence_role", "") if item else "",
                    "text": item.get("text", "") if item else "",
                    "counted": counted,
                })
        bucket["evidence_sources"] = sources
        bucket["independent_passages"] = sum(1 for item in sources if item["counted"])
        rows.append(bucket)
    rows.sort(key=lambda item: (item["tail"], item["head"], item["relation"]))
    return rows


def _collect_evidence(proposals, by_evidence):
    evidence_ids, counters, sources = [], [], []
    seen = set()
    for proposal in proposals:
        row = proposal["row"]
        for evidence_id in row.get("evidence_ids") or []:
            item = by_evidence.get(evidence_id)
            text_key = compact(item["text"]) if item else evidence_id
            counted = text_key not in seen
            if counted:
                seen.add(text_key)
                evidence_ids.append(evidence_id)
            sources.append({
                "evidence_id": evidence_id, "document_id": item.get("document_id", "") if item else "",
                "report_date": item.get("report_date", "") if item else "",
                "evidence_role": item.get("evidence_role", "") if item else "",
                "text": item.get("text", "") if item else "", "counted": counted,
                "target": proposal["target"], "hypothesis": proposal["hypothesis"],
            })
        for evidence_id in row.get("counter_evidence_ids") or []:
            if evidence_id not in counters:
                counters.append(evidence_id)
    return evidence_ids, counters, sources


def _attach_block_reason(edge, texts, by_id):
    if edge["status"] == "confirmed" or edge["relation"] not in CAUSAL_RELATIONS:
        return
    head = by_id.get(edge["head"])
    trigger_on_cause = head and any(
        _has_trigger(text) and _mentions_event(text, head) for text in texts)
    if not any(_has_trigger(text) for text in texts):
        extra = "没有因果触发词，时间顺序、产业链或历史机制不能确认因果"
    elif not trigger_on_cause:
        extra = "因果触发词没有落在原因事件上，不能靠时间或产业链补成因果"
    else:
        extra = ""
    if not extra:
        return
    edge["rejection_reason"] = (edge["rejection_reason"] + "；" + extra) if edge["rejection_reason"] else extra


def _emit_joint(edge):
    return {"head": edge["head"], "tail": edge["tail"], "type": "跨期传导", "relation": edge["relation"],
            "relation_class": edge["relation_class"], "status": edge["status"],
            "evidence_ids": edge["support_ids"] or edge["evidence_ids"],
            "evidence_sources": edge["evidence_sources"],
            "counter_evidence_ids": edge["counter_evidence_ids"], "components": edge["components"],
            "edge_score": edge["edge_score"], "model_score": edge["edge_score"], "score_kind": "model_score",
            "confidence": edge["edge_score"], "sources": edge["sources"],
            "hypothesis_ids": [item["hypothesis"] for item in edge["sources"]],
            "hypothesis_keys": ["%s/%s" % (item.get("target") or "", item.get("hypothesis") or "")
                                for item in edge["sources"]],
            "evidence_bundle": edge.get("evidence_bundle") or edge.get("components", {}).get("evidence_bundle") or {},
            "alternatives": edge["alternatives"], "rejection_reason": edge["rejection_reason"],
            "why": edge["why"], "gap_days": edge["gap_days"],
            "reason": edge["why"] or edge["rejection_reason"], "source": "溯因",
            "hypothesis": ",".join(sorted({item["hypothesis"] for item in edge["sources"]})),
            "hypothesis_kind": ",".join(sorted({item["kind"] for item in edge["sources"]})),
            "adjudications": edge.get("adjudications") or [],
            "target": edge["tail"]}


def _note(edge, kind, reason):
    notes = edge.setdefault("adjudications", [])
    if not any(item["kind"] == kind and item["reason"] == reason for item in notes):
        notes.append({"kind": kind, "head": edge["head"], "tail": edge["tail"], "reason": reason})


_MULTI_CAUSE = {"多个共同原因", "共同原因解释"}


def _hypothesis_key(source):
    return (source.get("target") or "", source.get("hypothesis") or "")


def _hypothesis_label(key):
    return "%s/%s" % key


def _demote_exclusive(edge, reason, drop):
    edge["status"] = "uncertain"
    edge["relation_class"] = "POSSIBLE_CAUSE"
    edge["rejection_reason"] = reason
    _note(edge, "exclusive_hypothesis", reason)
    drop.add(id(edge))


def _edge_has_support(edge):
    if edge.get("support_ids") or edge.get("evidence_ids"):
        return True
    return (edge.get("components") or {}).get("independent_passages", 0) > 0


def _cause_compatibility(edges):
    """Coexistence follows evidence, not whether one hypothesis happened to list every cause.

    Counterevidence on one edge is that edge's own problem. It does not make the
    other contributors mutually exclusive.
    """
    blob = " ".join((edge.get("why") or "") + " " + " ".join(edge.get("reasons") or []) for edge in edges)
    if any(word in blob for word in ("不能同时", "互相否定", "互斥", "相反事实")):
        return "MUTUALLY_EXCLUSIVE"
    same_head = len({edge["head"] for edge in edges}) == 1
    relations = {edge.get("relation") for edge in edges}
    if same_head and "导致" in relations and "抑制" in relations:
        return "MUTUALLY_EXCLUSIVE"
    if all(_edge_has_support(edge) for edge in edges):
        return "MULTIPLE_CONTRIBUTORS"
    if any(_edge_has_support(edge) for edge in edges):
        return "ALTERNATIVE_EXPLANATIONS"
    return "UNKNOWN_COMPATIBILITY"


def _resolve_cause_competition(confirmed_edges):
    """A hypothesis is (target, hypothesis id). The same h1 under two targets is not one hypothesis.

    Separate single-cause hypotheses stay together when each has support and nothing
    says they exclude each other. Only an explicit conflict removes an edge.
    """
    by_tail = {}
    for edge in confirmed_edges:
        by_tail.setdefault(edge["tail"], []).append(edge)
    competitions, drop = [], set()
    for tail, edges in sorted(by_tail.items()):
        heads = sorted({edge["head"] for edge in edges})
        if len(heads) < 2:
            continue
        covered = {}
        for edge in sorted(edges, key=lambda item: item["head"]):
            for source in sorted(edge.get("sources") or [], key=_hypothesis_key):
                covered.setdefault(_hypothesis_key(source), set()).add(edge["head"])
        labels = [_hypothesis_label(key) for key in sorted(covered)]
        raw_ids = sorted({key[1] for key in covered})
        for edge in edges:
            if edge.get("counter_evidence_ids"):
                edge["edge_support_status"] = "contradicted"
                edge["compatibility_status"] = "NOT_GROUP_EXCLUSION"
                edge["final_decision"] = "rejected"
                _demote_exclusive(edge, "这条边自己有反证，不据此否定其他原因", drop)
        remaining = [edge for edge in edges if id(edge) not in drop]
        if len({edge["head"] for edge in remaining}) < 2:
            for edge in remaining:
                edge["edge_support_status"] = "supported" if _edge_has_support(edge) else "uncertain"
                edge["compatibility_status"] = "NOT_GROUP_EXCLUSION"
                edge["final_decision"] = "confirmed"
            competitions.append({
                "type": "INDIVIDUAL_COUNTEREVIDENCE", "tail": tail, "heads": heads,
                "status": "individual", "compatibility_status": "NOT_GROUP_EXCLUSION",
                "final_decision": "rejected_only_contradicted_edges", "kept": None,
                "hypothesis_ids": raw_ids, "hypothesis_keys": labels,
                "reason": "反证只作用于携带它的那一条边"})
            continue
        kind = _cause_compatibility(remaining)
        if kind == "MUTUALLY_EXCLUSIVE":
            reason = "有明确互斥、相反事实或关系冲突，不能同时保留"
            for edge in remaining:
                edge["edge_support_status"] = "supported" if _edge_has_support(edge) else "uncertain"
                edge["compatibility_status"] = kind
                edge["final_decision"] = "rejected"
                _demote_exclusive(edge, reason, drop)
            competitions.append({"type": kind, "tail": tail, "heads": heads, "status": "exclusive",
                                 "compatibility_status": kind, "final_decision": "rejected",
                                 "kept": None, "hypothesis_ids": raw_ids, "hypothesis_keys": labels,
                                 "reason": reason})
            continue
        reason = {
            "MULTIPLE_CONTRIBUTORS": "各自有独立支持，作用机制没有冲突，可以同时作为贡献因素",
            "ALTERNATIVE_EXPLANATIONS": "几种解释并存，证据还不够把它们合成或排除",
            "UNKNOWN_COMPATIBILITY": "目前无法判断这些原因是否兼容",
        }[kind]
        joint = kind == "MULTIPLE_CONTRIBUTORS"
        for edge in remaining:
            supported = _edge_has_support(edge)
            edge["edge_support_status"] = "supported" if supported else "uncertain"
            edge["compatibility_status"] = kind
            edge["final_decision"] = "jointly_confirmed" if joint and supported else "individually_supported"
        competitions.append({
            "type": kind, "tail": tail, "heads": sorted(edge["head"] for edge in remaining),
            "status": "compatible" if joint else "uncertain",
            "compatibility_status": kind,
            "final_decision": "jointly_confirmed" if joint else "not_jointly_confirmed",
            "kept": None, "hypothesis_ids": raw_ids, "hypothesis_keys": labels, "reason": reason})
    competitions.sort(key=lambda item: (item.get("tail") or "", item.get("type") or "", item.get("reason") or ""))
    return [edge for edge in confirmed_edges if id(edge) not in drop], competitions


def select_joint(records, evidence, events, documents, multi_source=False, weights=None):
    """Adjudicate edges across hypotheses. This is not a global MAP optimization.

    weights score each target's own graphs inside select_graph. They are not
    applied again here. Causes of one result stay together when each is supported
    and no explicit exclusion, contrary fact, or relation conflict is present.
    """
    del weights
    by_id = {event["id"]: event for event in events}
    attach_mentions(evidence, events, documents)
    by_evidence = {item["evidence_id"]: item for item in evidence}
    dates = _dates(documents)
    grouped = {}
    for record in records:
        for hypothesis in record.get("hypotheses") or []:
            for row in hypothesis.get("edge_scores") or []:
                key = (row["head"], row["relation"], row["tail"])
                grouped.setdefault(key, []).append({
                    "row": row, "target": record["target"], "hypothesis": hypothesis.get("id") or "",
                    "kind": hypothesis.get("kind") or "", "hyp_why": hypothesis.get("why") or "",
                    "graph_total": (hypothesis.get("graph_scores") or {}).get("total"),
                })
    decided = []
    for key in sorted(grouped):
        head, relation, tail = key
        proposals = sorted(grouped[key], key=lambda item: (
            item.get("target") or "", item.get("hypothesis") or "", item["row"].get("why") or ""))
        evidence_ids, counters, sources = _collect_evidence(proposals, by_evidence)
        if multi_source:
            judged = _score_edge(
                {"head": head, "tail": tail, "relation": relation, "evidence_ids": evidence_ids,
                 "counter_evidence_ids": counters, "why": proposals[0]["row"].get("why") or "",
                 "missing": proposals[0]["row"].get("missing") or ""},
                {"id": tail}, set(by_id), by_id, by_evidence, dates, events)
        else:
            confirmed_items = [item for item in proposals if item["row"].get("confirmed")]
            pool = confirmed_items or proposals
            judged = max(pool, key=lambda item: (
                item["row"].get("edge_score") or 0,
                item["row"].get("components", {}).get("independent_passages", 0),
                item.get("target") or "", item.get("hypothesis") or ""))["row"]
        confirmed = bool(judged.get("confirmed")) and judged.get("relation_class") in CAUSAL_CLASSES
        if confirmed:
            status = "confirmed"
        elif judged.get("relation_class") == "POSSIBLE_CAUSE":
            status = "uncertain"
        else:
            status = "rejected"
        reason = "；".join(judged.get("reasons") or [])
        texts = [by_evidence[item]["text"] for item in evidence_ids if item in by_evidence]
        edge = {"head": head, "tail": tail, "relation": relation,
                "relation_class": judged.get("relation_class"), "status": status,
                "evidence_ids": list(judged.get("evidence_ids") or []),
                "support_ids": list(judged.get("support_ids") or []),
                "counter_evidence_ids": list(judged.get("counter_evidence_ids") or []),
                "components": judged.get("components") or {}, "edge_score": judged.get("edge_score") or 0,
                "reasons": list(judged.get("reasons") or []),
                "rejection_reason": "" if status == "confirmed" else reason,
                "why": judged.get("why") or "", "gap_days": judged.get("gap_days"),
                "sources": [{"target": item["target"], "hypothesis": item["hypothesis"],
                             "hypothesis_key": "%s/%s" % (item["target"], item["hypothesis"]),
                             "kind": item["kind"],
                             "why": item["row"].get("why") or item["hyp_why"],
                             "graph_total": item.get("graph_total")} for item in proposals],
                "evidence_sources": sources, "adjudications": []}
        joined = "；".join(edge["reasons"])
        if "时间不兼容" in joined or "中间出现反向观测" in joined:
            _note(edge, "time_conflict", "时间冲突：" + joined)
        _attach_block_reason(edge, texts, by_id)
        decided.append(edge)

    conflicts = []
    by_pair = {}
    for edge in decided:
        by_pair.setdefault((edge["head"], edge["tail"]), []).append(edge)
    for pair, edges in by_pair.items():
        live = [edge for edge in edges if edge["status"] == "confirmed"]
        relations = {edge["relation"] for edge in live}
        if len(live) < 2 or not ({"导致", "抑制"} <= relations or {"促进", "抑制"} <= relations):
            continue
        ranked = sorted(live, key=lambda edge: (
            -edge["edge_score"], -edge["components"].get("independent_passages", 0), edge["relation"]))
        top_key = (ranked[0]["edge_score"], ranked[0]["components"].get("independent_passages", 0))
        second_key = (ranked[1]["edge_score"], ranked[1]["components"].get("independent_passages", 0))
        if top_key == second_key:
            for edge in live:
                edge["status"] = "uncertain"
                edge["relation_class"] = "POSSIBLE_CAUSE"
                edge["rejection_reason"] = "互相冲突的因果解释分数相同，不能取舍"
                _note(edge, "relation_conflict", edge["rejection_reason"])
            conflicts.append({"pair": list(pair), "relations": sorted(relations), "kept": None,
                              "reason": "互相冲突且分数相同"})
        else:
            for edge in ranked[1:]:
                if edge["relation"] == ranked[0]["relation"]:
                    continue
                edge["status"] = "rejected"
                edge["relation_class"] = "NONE"
                edge["rejection_reason"] = "低于另一条有证据的冲突解释：%s" % ranked[0]["relation"]
                _note(edge, "relation_conflict", edge["rejection_reason"])
            conflicts.append({"pair": list(pair), "relations": sorted(relations), "kept": ranked[0]["relation"],
                              "reason": "保留证据分更高的关系，另一条解释被拒绝"})

    accepted, forward = [], {}
    ranked = sorted((edge for edge in decided if edge["status"] == "confirmed"),
                    key=lambda edge: (-edge["edge_score"], -edge["components"].get("independent_passages", 0),
                                      edge["head"], edge["tail"]))
    for edge in ranked:
        trial = {node: list(children) for node, children in forward.items()}
        trial.setdefault(edge["head"], []).append(edge["tail"])
        if _has_cycle(trial):
            edge["status"] = "rejected"
            edge["relation_class"] = "NONE"
            edge["rejection_reason"] = "会与已接受的边形成环"
            _note(edge, "causal_cycle", edge["rejection_reason"])
            continue
        forward.setdefault(edge["head"], []).append(edge["tail"])
        accepted.append(edge)

    use = Counter(evidence_id for edge in accepted for evidence_id in edge.get("support_ids") or [])
    final = []
    for edge in accepted:
        shared_bad = []
        for evidence_id in edge.get("support_ids") or []:
            if use[evidence_id] <= 1:
                continue
            item = by_evidence.get(evidence_id)
            head, tail = by_id.get(edge["head"]), by_id.get(edge["tail"])
            if item is None or head is None or tail is None or not (
                    _mentions_event(item["text"], head) and _mentions_event(item["text"], tail)):
                shared_bad.append(evidence_id)
        if not shared_bad:
            final.append(edge)
            continue
        remain = [evidence_id for evidence_id in edge["support_ids"] if evidence_id not in shared_bad]
        retried = _score_edge(
            {"head": edge["head"], "tail": edge["tail"], "relation": edge["relation"],
             "evidence_ids": remain, "counter_evidence_ids": edge["counter_evidence_ids"],
             "why": edge["why"]},
            {"id": edge["tail"]}, set(by_id), by_id, by_evidence, dates, events)
        if retried.get("confirmed") and retried.get("relation_class") in CAUSAL_CLASSES:
            edge["support_ids"] = list(retried.get("support_ids") or [])
            edge["components"] = retried.get("components") or {}
            edge["edge_score"] = retried.get("edge_score") or 0
            final.append(edge)
        else:
            edge["status"] = "uncertain"
            edge["relation_class"] = "POSSIBLE_CAUSE"
            edge["rejection_reason"] = "同一段原文被另一条边使用，且没有同时点名这条边的两端"
            _note(edge, "duplicate_evidence", edge["rejection_reason"])

    final, cause_competitions = _resolve_cause_competition(final)
    for edge in decided:
        edge["alternatives"] = [{
            "head": other["head"], "tail": other["tail"], "relation": other["relation"],
            "status": other["status"], "relation_class": other["relation_class"],
            "edge_score": other["edge_score"], "rejection_reason": other["rejection_reason"],
            "hypotheses": [item["hypothesis"] for item in other["sources"]],
        } for other in decided if other is not edge and (other["tail"] == edge["tail"] or (
            other["head"] == edge["head"] and other["tail"] == edge["tail"]))]
    competitions = list(cause_competitions)
    competitions.extend({"type": "conflicting_relations", "pair": item["pair"], "relations": item["relations"],
                         "kept": item["kept"], "reason": item["reason"]} for item in conflicts)
    confirmed = [_emit_joint(edge) for edge in final]
    uncertain = [_emit_joint(edge) for edge in decided if edge["status"] == "uncertain"]
    rejected = [_emit_joint(edge) for edge in decided if edge["status"] == "rejected"]
    linked = {edge["head"] for edge in confirmed} | {edge["tail"] for edge in confirmed}
    adjudications = [note for edge in decided for note in edge.get("adjudications") or []]
    return {"confirmed": confirmed, "possible": uncertain, "rejected": rejected, "conflicts": conflicts,
            "competitions": competitions, "edges": confirmed + uncertain + rejected,
            "joint_score": round(sum(edge["edge_score"] for edge in confirmed), 3),
            "procedure": "逐边联合裁决", "map_optimization": False, "weights_applied": False,
            "multi_source": multi_source, "adjudications": adjudications,
            "unlinked_targets": [record["target"] for record in records if record["target"] not in linked]}


def _reselect_multisource(records, evidence, events, documents, weights):
    """Give every proposal of the same edge the union of its passages, then pick per target again."""
    by_evidence = {item["evidence_id"]: item for item in evidence}
    grouped = {}
    for record in records:
        for hypothesis in record.get("hypotheses") or []:
            for row in hypothesis.get("edge_scores") or []:
                key = (row["head"], row["relation"], row["tail"])
                grouped.setdefault(key, {"evidence_ids": [], "counter_evidence_ids": [], "seen": set()})
                bucket = grouped[key]
                for evidence_id in row.get("evidence_ids") or []:
                    item = by_evidence.get(evidence_id)
                    text_key = compact(item["text"]) if item else evidence_id
                    if text_key in bucket["seen"]:
                        continue
                    bucket["seen"].add(text_key)
                    bucket["evidence_ids"].append(evidence_id)
                for evidence_id in row.get("counter_evidence_ids") or []:
                    if evidence_id not in bucket["counter_evidence_ids"]:
                        bucket["counter_evidence_ids"].append(evidence_id)
    by_id = {event["id"]: event for event in events}
    rescored = []
    for record in records:
        hypotheses = []
        for hypothesis in record.get("hypotheses") or []:
            edges = []
            for row in hypothesis.get("edge_scores") or []:
                key = (row["head"], row["relation"], row["tail"])
                bucket = grouped.get(key, {})
                edges.append({"head": row["head"], "tail": row["tail"], "relation": row["relation"],
                              "evidence_ids": bucket.get("evidence_ids") or row.get("evidence_ids") or [],
                              "counter_evidence_ids": bucket.get("counter_evidence_ids") or row.get("counter_evidence_ids") or [],
                              "why": row.get("why") or "", "missing": row.get("missing") or "",
                              "needs_retrieval": row.get("needs_retrieval") or False})
            hypotheses.append({"id": hypothesis.get("id") or "", "kind": hypothesis.get("kind") or "",
                               "why": hypothesis.get("why") or "", "missing": hypothesis.get("missing") or "",
                               "needs_retrieval": hypothesis.get("needs_retrieval") or False,
                               "nodes": hypothesis.get("nodes") or [], "edges": edges})
        choice = select_graph(by_id[record["target"]], hypotheses, record.get("candidates") or [],
                              events, documents, evidence, weights)
        choice["recall_stats"] = record.get("recall_stats") or {}
        rescored.append(choice)
    return rescored


def _retrieval_questions(hypotheses):
    questions = []
    for hypothesis in hypotheses or []:
        if hypothesis.get("needs_retrieval"):
            text = hypothesis.get("retrieval_query") or hypothesis.get("missing") or ""
            if text:
                questions.append(text)
    return questions


def merge_catalog(evidence, retrieved):
    """Supplement passages join the catalog used by the final graph and the joint decision."""
    seen = {item.get("evidence_id") for item in evidence if item.get("evidence_id")}
    merged = list(evidence)
    for item in retrieved or []:
        marker = item.get("evidence_id")
        if marker and marker not in seen:
            merged.append(item)
            seen.add(marker)
    return merged


def _supplement(hypotheses, documents, corpus, evidence, retrieval_method, encoder, rounds, top_k, max_chars):
    """One bounded loop. New passages are shown to the model; they are not yet a cause."""
    from eventkg.evidence_index import retrieve_passages

    stats = {"rounds": 0, "added": 0, "requests": 0, "stopped": "没有补充检索需求", "chars": 0}
    cache = {}
    extra = []
    seen = {item.get("evidence_id") for item in evidence}
    for _ in range(max(0, rounds)):
        questions = _retrieval_questions(hypotheses)
        if not questions:
            stats["stopped"] = "没有补充检索需求"
            break
        stats["requests"] += len(questions)
        found = retrieve_passages(questions, corpus, method=retrieval_method, encoder=encoder,
                                  top_k=top_k, cache=cache, max_chars=max_chars)
        fresh = [item for item in found if item.get("evidence_id") not in seen]
        stats["rounds"] += 1
        if not fresh:
            stats["stopped"] = "没有新增有效证据"
            break
        for item in fresh:
            seen.add(item["evidence_id"])
            extra.append(item)
            stats["chars"] += len(item.get("text") or "")
        stats["added"] += len(fresh)
        if stats["chars"] >= max_chars:
            stats["stopped"] = "达到输入长度上限"
            break
        stats["stopped"] = "已补充证据，等待重新验证"
        break
    return extra, stats


def induce(chat, events, relations, documents, workers=4, target_limit=None, top_k=TOP_K, weights=None,
           multi_source=False, joint=False, retrieval_method="bm25", encoder=None, retrieval_rounds=1,
           retrieval_top_k=4, retrieval_max_chars=2000):
    """Score target hypotheses, then either keep the per-target winners or select one global subset.

    A is the per-target winner followed by a merge. B unions passages before that
    choice. C and D choose from the whole candidate graph; D unions passages first.
    """
    if retrieval_method in ("dense", "hybrid") and encoder is None:
        raise RuntimeError("检索方式 %s 需要向量编码器，不会回退到字符 Jaccard。" % retrieval_method)
    if retrieval_method in ("dense", "hybrid") and encoder is not None:
        from eventkg.retrieval import VectorCache
        if not isinstance(encoder, VectorCache):
            encoder = VectorCache(encoder)
    from eventkg.evidence_index import index_corpus
    evidence = index_evidence(events, documents, relations)
    corpus = index_corpus(documents)
    rules = abstract_rules(events, relations)
    targets, target_stats = important_targets(events, documents, target_limit)

    def run(target):
        neighborhood = retrieve_neighborhood(
            target, events, documents, relations, rules, top_k=top_k,
            retrieval_method=retrieval_method, encoder=encoder)
        if not neighborhood["candidates"]:
            hypotheses = [{"id": "h0", "kind": "证据不足", "nodes": [], "why": "没有召回到相关事件",
                           "missing": "", "needs_retrieval": False, "edges": []}]
            extra, retrieval_stats = [], {"rounds": 0, "added": 0, "requests": 0, "stopped": "没有候选"}
        else:
            hypotheses = propose_graphs(chat, target, neighborhood, relations, documents, evidence)
            extra, retrieval_stats = _supplement(
                hypotheses, documents, corpus, evidence, retrieval_method, encoder,
                retrieval_rounds, retrieval_top_k, retrieval_max_chars)
            if extra:
                catalog = list(evidence) + extra
                hypotheses = propose_graphs(
                    chat, target, neighborhood, relations, documents, catalog,
                    extra_ids={item["evidence_id"] for item in extra})
                retrieval_stats["stopped"] = "补充证据后已重新验证"
        choice = select_graph(target, hypotheses, neighborhood, events, documents,
                              list(evidence) + extra, weights)
        choice["recall_stats"] = neighborhood["stats"]
        choice["retrieval_stats"] = retrieval_stats
        choice["retrieved_evidence"] = extra
        return choice

    rows = parallel_map(run, targets, workers, "溯因解释")
    retrieved = []
    for row in rows:
        retrieved.extend(row.get("retrieved_evidence") or [])
    evidence = merge_catalog(evidence, retrieved)
    candidate_graph = build_candidate_graph(rows, evidence)
    if joint:
        overall = select_joint(rows, evidence, events, documents, multi_source=multi_source, weights=weights)
    elif multi_source:
        overall = select_global(_reselect_multisource(rows, evidence, events, documents, weights), evidence, events)
    else:
        overall = select_global(rows, evidence, events)
    overall["mode"] = mode_name(multi_source, joint)
    overall["candidate_graph"] = candidate_graph
    mentions = [mention for item in evidence for mention in item.get("mentions") or []]
    payload = {"targets": rows, "evidence": evidence, "mentions": mentions, "global": overall,
               "candidate_graph": candidate_graph, "target_stats": target_stats, "mode": overall["mode"],
               "weights": dict(DEFAULT_WEIGHTS, **(weights or {}))}
    return overall["confirmed"], payload
