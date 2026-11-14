# -*- coding: utf-8 -*-
"""One relation-evidence check for scoring, paths and answers.

A causal word plus two indicator strings is not support. The cause mention,
the trigger and the effect mention have to sit in that order inside one clause,
and the trigger has to be asserted rather than denied or left uncertain.
"""
import hashlib
import re

from atlas_rag.kg_construction.commodity import match_commodities
from eventkg.ontology import EXACT_INDICATORS, _alias_indicator
from eventkg.relations import RELATION_WORDS, compat_map
from eventkg.validate import explicit_date, time_bucket

_COMPAT = compat_map()
_CAUSE = RELATION_WORDS["导致"]
_SHORT = {
    "发运": "发运量", "到港": "到港量", "铁水": "日均铁水产量",
    "港存": "港口库存", "矿价": "价格",
}
_STOCK_KINDS = (
    ("港口库存", ("港口库存", "港存")),
    ("社会库存", ("社会库存", "社库")),
    ("钢厂库存", ("钢厂库存", "厂库")),
    ("总库存", ("总库存",)),
)
_DIR_WORDS = (
    ("上涨", "上升"), ("上升", "上升"), ("增加", "上升"), ("回升", "上升"), ("累库", "上升"), ("持涨", "上升"),
    ("下跌", "下降"), ("下降", "下降"), ("减少", "下降"), ("回落", "下降"), ("去库", "下降"), ("下滑", "下降"),
    ("持平", "持平"), ("持稳", "持平"),
)
_NEG = re.compile(r"(不可能|没有|并未|不是|并非|并不|未|无)$")
_UNCERTAIN = re.compile(r"(不一定|未必|不见得|或许|也许|可能)$")
_FORECAST = re.compile(r"预计|展望|或将|料将|预期")
_BOUND = re.compile(r"[，,、。；;\n]")
_BECAUSE = re.compile(r"(?:由于|因为)([^，,、。；;\n]{1,48})[，,、]([^，,、。；;\n]{1,48})")
_AFFECT = re.compile(r"受([^，,、。；;\n]{0,24}?)(影响|拖累|压制|挤压)[，,]([^，,、。；;\n]{1,48})")


def _events(events):
    if isinstance(events, dict):
        return events
    return {item["id"]: item for item in events or [] if isinstance(item, dict) and item.get("id")}


def _aliases(event):
    """Canonical names and registered aliases. Arbitrary slices of the indicator are not mentions."""
    indicator = (event or {}).get("indicator") or ""
    canonical = _alias_indicator(indicator, (event or {}).get("scope") or "")
    found = []
    for key, value in EXACT_INDICATORS.items():
        if value in (canonical, indicator) or key in (canonical, indicator):
            found.extend((key, value))
    for short, full in _SHORT.items():
        if full in (canonical, indicator):
            found.append(short)
    if "港口库存" in (canonical, indicator):
        found.append("库存")
    if len(indicator) >= 2:
        found.append(indicator)
    if len(canonical) >= 2:
        found.append(canonical)
    unique = []
    for item in found:
        if len(item) >= 2 and item not in unique:
            unique.append(item)
    unique.sort(key=len, reverse=True)
    return unique


def _triggers(relation):
    return _COMPAT.get(relation, ())


def _event_period(event):
    dated = explicit_date(event.get("event_time") or "") or explicit_date(event.get("time") or "")
    if dated:
        return dated
    raw = event.get("time") or ""
    if not raw:
        return ""
    bucket = time_bucket(raw)
    if bucket in ("本期", "上期", "下期"):
        return bucket
    return bucket if explicit_date(bucket) else ""


def _span_period(text):
    dated = explicit_date(text)
    if dated:
        return dated
    day = re.search(r"\d{1,2}月\d{1,2}日", text or "")
    if day:
        return day.group(0)
    relative = re.sub(r"较上周|较上期|较上月|比上周|比上期", "", text or "")
    if re.search(r"下周|下期", relative):
        return "下期"
    if re.search(r"上周|上期", relative):
        return "上期"
    if re.search(r"本周|本期", relative):
        return "本期"
    return ""


def _month_day(value):
    dated = explicit_date(value)
    if dated:
        return dated[5:]
    match = re.search(r"(\d{1,2})月(\d{1,2})日", value or "")
    if not match:
        return ""
    return "%02d-%02d" % (int(match.group(1)), int(match.group(2)))


def _named_stock(text):
    raw = text or ""
    for name, words in _STOCK_KINDS:
        if any(word in raw for word in words):
            return name
    return ""


def _periods_compatible(event, event_period, text_period):
    """本期 matches only this event's own report date, never an arbitrary earlier date."""
    if not event_period or not text_period:
        return True
    if event_period == text_period:
        return True
    if event_period == "本期" and _month_day(text_period):
        report = explicit_date((event or {}).get("report_date") or "") or ""
        dated = explicit_date(text_period) or ""
        if report and dated:
            return report == dated
        return False
    one, other = _month_day(event_period), _month_day(text_period)
    return bool(one and other and one == other)


def _direction_in(span):
    found = None
    for word, level in _DIR_WORDS:
        at = span.rfind(word)
        if at >= 0 and (found is None or at >= found[0]):
            found = (at, level)
    if not found:
        return "", False
    prefix = re.sub(r"(明显|大幅|继续|进一步)$", "", span[max(0, found[0] - 6):found[0]])
    return found[1], bool(_NEG.search(prefix))


def _mention_covers(event, item, start, end):
    """When a passage already names its mentions, do not bind a different event by string search."""
    mentions = (item or {}).get("mentions") or []
    if not mentions:
        return True
    for mention in mentions:
        if mention.get("linked_event_id") != event.get("id"):
            continue
        if float(mention.get("alignment_confidence") or 0) < 0.75:
            continue
        left, right = mention.get("char_start"), mention.get("char_end")
        if left is None or right is None:
            return True
        if not (end <= left or start >= right):
            return True
    return False


def _instance_hit(event, text, start, end, item=None):
    """A hit counts only when this span is this event, not a shorter or different one."""
    if start >= end:
        return None
    span = text[start:end]
    hit = _hit(text, _aliases(event), start, end)
    if not hit:
        return None
    if not _mention_covers(event, item, start, end):
        return None
    source_doc = (item or {}).get("document_id") or ""
    if source_doc and event.get("doc_id") and source_doc != event.get("doc_id"):
        linked = any(
            mention.get("linked_event_id") == event.get("id")
            and float(mention.get("alignment_confidence") or 0) >= 0.75
            for mention in (item or {}).get("mentions") or [])
        dated = bool(_span_period(span) or _span_period(text[:start]) or _span_period(text))
        scope = (event.get("scope") or "").strip()
        named = bool(scope and scope in text)
        if not linked and not dated and not named:
            return None
    target_stock, named_stock = _named_stock(event.get("indicator") or ""), _named_stock(span)
    if named_stock and target_stock and named_stock != target_stock:
        return None
    scope = (event.get("scope") or "").strip()
    if scope and scope not in span:
        return None
    subject = (event.get("subject") or "").strip()
    if subject and subject not in span:
        return None
    commodity = event.get("commodity") or ""
    prefix = text[max(start, hit[1] - 8):hit[1]]
    mentioned = match_commodities(prefix) if commodity else []
    if mentioned and commodity not in mentioned and not any(commodity in item or item in commodity for item in mentioned):
        return None
    event_period = _event_period(event)
    text_period = _span_period(span) or _span_period(text[:start])
    if event_period and text_period and not _periods_compatible(event, event_period, text_period):
        return None
    if _month_day(event_period) and not text_period:
        return None
    direction, negated = _direction_in(span)
    event_dir = event.get("dir") or ""
    if direction and event_dir in ("上升", "下降", "持平") and direction != event_dir:
        return None
    return hit, negated, bool(direction)


def _spots(text, words):
    found, occupied = [], []
    for word in sorted(set(words), key=len, reverse=True):
        start = 0
        while True:
            at = text.find(word, start)
            if at < 0:
                break
            end = at + len(word)
            if not any(at < right and end > left for left, right in occupied):
                found.append((at, word))
                occupied.append((at, end))
            start = end
    found.sort()
    return found


def _bound_before(text, pos):
    at = 0
    for match in _BOUND.finditer(text):
        if match.end() <= pos:
            at = match.end()
        else:
            break
    return at


def _bound_after(text, pos):
    match = _BOUND.search(text, pos)
    return match.start() if match else len(text)


def _hit(text, aliases, start, end):
    best = None
    for alias in aliases:
        at = text.find(alias, start, end)
        if at >= 0 and (best is None or len(alias) > best[0]):
            best = (len(alias), at, at + len(alias))
    return best


def _modality(text, cause_start, trigger_at):
    prefix = re.sub(r"(明显|大幅|进一步)$", "", text[max(cause_start, trigger_at - 8):trigger_at])
    if _NEG.search(prefix):
        return "negated"
    if _UNCERTAIN.search(prefix):
        return "uncertain"
    if _FORECAST.search(text[max(0, cause_start - 12):trigger_at]):
        return "forecast"
    return "asserted"


def _sides(head, tail, text, cause_start, cause_end, effect_start, effect_end, item=None):
    forward = (_instance_hit(head, text, cause_start, cause_end, item),
               _instance_hit(tail, text, effect_start, effect_end, item))
    backward = (_instance_hit(tail, text, cause_start, cause_end, item),
                _instance_hit(head, text, effect_start, effect_end, item))
    return forward, backward


def _both_directions_missing(head, tail, left, right):
    """A directed instance needs a direction somewhere in the argument.

    「发运量导致到港量」 names neither decline. A sentence that states one side's
    direction can still be that instance; a sentence that states neither cannot.
    """
    needed = [event.get("dir") in ("上升", "下降", "持平") for event in (head, tail)]
    stated = [bool(side and side[2]) for side in (left, right)]
    return all(needed) and not any(stated)


def _record_pair(found, text, mode, forward, backward, snippet, head, tail):
    if forward[0] and forward[1]:
        if _both_directions_missing(head, tail, forward[0], forward[1]):
            return
        if mode == "asserted" and (forward[0][1] or forward[1][1]):
            mode = "negated"
        found.append(("forward", mode, snippet, forward))
    elif backward[0] and backward[1]:
        if _both_directions_missing(tail, head, backward[0], backward[1]):
            return
        found.append(("backward", mode, snippet, backward))


def _cause_start(text, pos, words):
    """A comma immediately before 导致 still belongs to the cause. A comma before the next relation does not."""
    start = _bound_before(text, pos)
    if _instance_region_has_text(text, start, pos):
        return start
    if start <= 0:
        return start
    earlier = _bound_before(text, start - 1)
    if _spots(text[earlier:start], words):
        return start
    return earlier


def _instance_region_has_text(text, start, end):
    return bool(compact_span(text[start:end]))


def compact_span(text):
    return re.sub(r"[\s，,、]", "", text or "")


def _judge_text(text, head, tail, relation, item=None):
    """Support requires this event pair, their directions, and one compatible trigger in one argument."""
    words = _triggers(relation)
    cause_family = relation in _CAUSE
    spots = _spots(text, words)
    found = []
    for pos, word in spots:
        cause_start = _cause_start(text, pos, words)
        effect_end = _bound_after(text, pos + len(word))
        mode = _modality(text, cause_start, pos)
        if mode == "asserted" and re.search(r"预计|展望|或将|料将", text[pos:effect_end]):
            mode = "forecast"
        forward, backward = _sides(head, tail, text, cause_start, pos, pos + len(word), effect_end, item)
        _record_pair(found, text, mode, forward, backward, text[cause_start:effect_end], head, tail)
    if cause_family:
        for match in _BECAUSE.finditer(text):
            mode = _modality(text, match.start(), match.start(1))
            forward, backward = _sides(head, tail, text, match.start(1), match.end(1), match.start(2), match.end(2), item)
            _record_pair(found, text, mode, forward, backward, match.group(0), head, tail)
        for match in _AFFECT.finditer(text):
            if match.group(2) != "影响":
                continue
            mode = _modality(text, match.start(), match.start(1))
            forward, backward = _sides(head, tail, text, match.start(1), match.end(1), match.start(3), match.end(3), item)
            _record_pair(found, text, mode, forward, backward, match.group(0), head, tail)
    if relation in ("拖累", "挤压", "抑制", "压制"):
        for match in _AFFECT.finditer(text):
            if match.group(2) not in _triggers(relation) and not (
                relation in ("抑制", "压制") and match.group(2) == "压制"
            ) and not (relation in ("拖累", "挤压") and match.group(2) in ("拖累", "挤压")):
                continue
            mode = _modality(text, match.start(), match.start(1))
            forward, backward = _sides(head, tail, text, match.start(1), match.end(1), match.start(3), match.end(3), item)
            _record_pair(found, text, mode, forward, backward, match.group(0), head, tail)
    if any(item[0] == "forward" and item[1] == "asserted" for item in found) and any(
            item[0] == "backward" and item[1] == "asserted" for item in found):
        return "conflict", "", None
    for label, mode, snippet, spans in found:
        if label == "forward" and mode == "asserted":
            return "support", snippet, spans
    for label, mode, snippet, spans in found:
        if label == "forward" and mode in ("negated", "uncertain", "forecast"):
            return mode, snippet, spans
        if label == "backward" and mode == "asserted":
            return "reverse", snippet, spans
    if spots or (cause_family and (_BECAUSE.search(text) or _AFFECT.search(text))):
        return "unrelated", "", None
    return "unrelated", "", None


def _periods_differ(left, right):
    one, other = _span_period(left), _span_period(right)
    return bool(one and other and one != other)


def instance_fingerprint(event):
    """Identity fields that make an old verification stale when they change."""
    event = event or {}
    signals = []
    for signal in event.get("trend_signals") or []:
        signals.append("%s=%s" % (signal.get("comparison_basis") or "", signal.get("level_direction") or ""))
    roles = []
    for role in event.get("roles") or []:
        if isinstance(role, dict):
            roles.append("%s:%s:%s" % (
                role.get("role") or role.get("name") or "",
                role.get("entity_id") or role.get("id") or "",
                role.get("value") or role.get("name") or ""))
        else:
            roles.append(str(role))
    parts = [
        event.get("commodity") or "", event.get("indicator") or "", event.get("subject") or "",
        event.get("scope") or "", event.get("event_time") or event.get("time") or "",
        event.get("period") or "", event.get("observation_period") or "", event.get("mention_date") or "",
        "|".join(sorted(role for role in roles if role)),
        event.get("dir") or event.get("level_direction") or "", event.get("status") or "",
        event.get("comparison_basis") or "", "|".join(signals),
    ]
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _source_views(evidence_id, item):
    """One passage may be quoted by several reports. Each report keeps its own identity."""
    listed = [source for source in (item.get("sources") or []) if isinstance(source, dict)]
    if not listed:
        listed = [{}]
    views = []
    for source in listed:
        views.append({
            "evidence_id": source.get("evidence_id") or evidence_id,
            "document_id": source.get("document_id") or item.get("document_id") or item.get("doc_id") or "",
            "report_date": source.get("report_date") or item.get("report_date") or "",
            "page": source.get("page", item.get("page")),
            "section": source.get("section") or item.get("section") or "",
            "char_start": source.get("char_start", item.get("char_start")),
            "char_end": source.get("char_end", item.get("char_end")),
            "place_status": source.get("place_status") or item.get("place_status") or "",
            "mention_alignment": source.get("mention_alignment") or "|".join(
                "%s@%s" % (mention.get("linked_event_id") or "", mention.get("alignment_confidence") or "")
                for mention in (item.get("mentions") or [])),
        })
    return views


def _evidence_rows(catalog, ids):
    rows = []
    sources = []
    for evidence_id in ids:
        item = catalog.get(evidence_id) or {}
        for source in _source_views(evidence_id, item):
            sources.append(source)
            rows.append("|".join(str(source[key]) for key in (
                "evidence_id", "document_id", "report_date", "page", "section",
                "char_start", "char_end", "place_status", "mention_alignment"))
                        + "|" + (item.get("text") or ""))
    version = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]
    return version, sources


def _focus_period(event):
    dated = explicit_date(event.get("event_time") or "") or explicit_date(event.get("time") or "")
    if dated:
        return dated
    raw = event.get("time") or event.get("event_time") or ""
    bucket = time_bucket(raw) if raw else "UNKNOWN"
    return "" if not raw or bucket == "UNKNOWN" else bucket


def _text_period_key(text):
    dates = list(dict.fromkeys(re.findall(r"20\d{2}-\d{2}-\d{2}", text or "")))
    if len(dates) == 1:
        return dates[0]
    if len(dates) > 1:
        return "|".join(dates)
    bucket = time_bucket(text or "")
    return "" if bucket == "UNKNOWN" else bucket


def _periods_dispute(support_key, counter_key):
    """An undated snippet cannot be used to say a denial belongs to another period."""
    if not support_key or not counter_key:
        return True
    return support_key == counter_key


def verify_relation(edge, events, evidence_index):
    """Return the verification record consumed by scoring, paths and answers.

    Support, counter-evidence and unrelated ids are one request. The record is stale
    when the event identity or the evidence text and source position change.
    """
    events = _events(events)
    catalog = evidence_index if isinstance(evidence_index, dict) else {
        item.get("evidence_id"): item for item in evidence_index or [] if item.get("evidence_id")}
    support_ids = [item for item in (edge.get("evidence_ids") or []) if item]
    counter_ids = [item for item in (edge.get("counter_evidence_ids") or []) if item and item not in support_ids]
    other_ids = [item for item in (edge.get("unrelated_evidence_ids") or []) if item
                 and item not in support_ids and item not in counter_ids]
    ids = support_ids + counter_ids + other_ids
    head = events.get(edge.get("head")) or {}
    tail = events.get(edge.get("tail")) or {}
    relation = edge.get("relation") or edge.get("type") or ""
    version, sources = _evidence_rows(catalog, ids)
    identity = instance_fingerprint(head) + ":" + instance_fingerprint(tail)
    record = {
        "head_event": edge.get("head"), "tail_event": edge.get("tail"), "relation": relation,
        "relation_evidence_ids": ids, "support_ids": support_ids, "counter_ids": counter_ids,
        "evidence_version": version, "identity_fingerprint": identity, "sources": sources,
        "supporting_evidence_ids": [], "counter_evidence_ids": [], "unrelated_evidence_ids": [],
        "mention_alignment": "unbound", "relation_scope": "missing",
        "negation": False, "modality": "asserted", "verification_status": "UNVERIFIED", "reason": "",
        "text": "", "relation_type": relation,
        "head_identity_fingerprint": instance_fingerprint(head),
        "tail_identity_fingerprint": instance_fingerprint(tail),
        "event_period": _focus_period(head) or _focus_period(tail),
        "report_series_id": head.get("report_series_id") or tail.get("report_series_id") or "",
    }

    def _finish(row):
        row["edge_verification_status"] = row.get("verification_status")
        row["relation_type"] = row.get("relation") or relation
        return row
    if not support_ids or any(item not in catalog for item in ids):
        record["verification_status"] = "MISSING"
        record["reason"] = "证据不存在或无法恢复"
        return _finish(record)
    buckets = {name: [] for name in ("support", "reverse", "conflict", "negated", "uncertain", "forecast", "unrelated")}
    snippets = {}
    judged = []
    for evidence_id in ids:
        item = catalog.get(evidence_id) or {}
        text = item.get("text") or ""
        kind, snippet, _spans = _judge_text(text, head, tail, relation, item)
        buckets[kind].append(evidence_id)
        snippets.setdefault(kind, snippet or text)
        judged.append((evidence_id, kind, _text_period_key(snippet or text), snippet or text))
    record["supporting_evidence_ids"] = buckets["support"]
    record["counter_evidence_ids"] = buckets["reverse"] + buckets["conflict"] + buckets["negated"]
    record["unrelated_evidence_ids"] = buckets["unrelated"]
    periods = [item for item in (_focus_period(head), _focus_period(tail)) if item]
    report = explicit_date(head.get("report_date") or "") or explicit_date(tail.get("report_date") or "") or ""

    def _about_this_event(key):
        if not periods:
            return True
        if not key:
            return True
        parts = set(key.split("|"))
        if any(period in parts for period in periods):
            return True
        return bool("本期" in periods and report and report in parts)

    support_rows = [(evidence_id, key, snippet) for evidence_id, kind, key, snippet in judged
                    if kind == "support" and _about_this_event(key)]
    counter_rows = [(evidence_id, key, snippet) for evidence_id, kind, key, snippet in judged
                    if kind in ("reverse", "conflict", "negated") and _about_this_event(key)]
    if support_rows and counter_rows and any(
            _periods_dispute(left[1], right[1]) for left in support_rows for right in counter_rows):
        record.update(mention_alignment="aligned", relation_scope="conflict", modality="disputed",
                      negation=any(kind == "negated" for _id, kind, key, _snippet in judged if _about_this_event(key)),
                      verification_status="DISPUTED", text=support_rows[0][2],
                      reason="同一事件、同一时期同时有支持和否定")
        return _finish(record)
    if not periods:
        support_keys = {(key or "UNSPECIFIED") for _id, key, _snippet in support_rows}
        if len(support_keys) > 1:
            record["reason"] = "支持证据分属不同时期，事件时期无法唯一对应"
            record["relation_scope"] = "split"
            return _finish(record)
    elif not support_rows and buckets["support"]:
        record["reason"] = "支持证据的时期不是这条事件的时期"
        record["relation_scope"] = "split"
        return _finish(record)
    if support_rows:
        buckets["support"] = [row[0] for row in support_rows]
        snippets["support"] = support_rows[0][2]
    else:
        buckets["support"] = []
    if buckets["conflict"]:
        record.update(relation_scope="conflict", modality="conflict", reason="同一条证据里同时有相反方向")
        return _finish(record)
    if buckets["support"]:
        forecast_event = head.get("status") == "预测中" or tail.get("status") == "预测中"
        record.update(mention_alignment="aligned", relation_scope="same_argument",
                      modality="forecast" if forecast_event else "asserted",
                      verification_status="FORECASTS" if forecast_event else "VERIFIED",
                      text=snippets.get("support") or "",
                      reason="事件仍是预测，不能记成已经发生的因果" if forecast_event else "")
        return _finish(record)
    if buckets["forecast"]:
        record.update(mention_alignment="aligned", relation_scope="same_argument", modality="forecast",
                      verification_status="FORECASTS", text=snippets.get("forecast") or "",
                      reason="结果侧仍是预测，不是已经发生的因果")
        return _finish(record)
    if buckets["uncertain"]:
        record.update(mention_alignment="aligned", relation_scope="same_argument", modality="uncertain",
                      text=snippets.get("uncertain") or "", reason="原文没有把这条关系说成确定事实")
        return _finish(record)
    if buckets["negated"]:
        record.update(mention_alignment="aligned", relation_scope="same_argument", negation=True,
                      modality="negated", text=snippets.get("negated") or "", reason="原文否定了这条关系")
        return _finish(record)
    if buckets["reverse"]:
        record.update(mention_alignment="reversed", relation_scope="same_argument",
                      text=snippets.get("reverse") or "", reason="因果方向与原文相反")
        return _finish(record)
    record["reason"] = "原文没有把这两个事件连成这条关系"
    record["relation_scope"] = "split"
    return _finish(record)
