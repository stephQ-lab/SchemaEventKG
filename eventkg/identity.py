# -*- coding: utf-8 -*-
"""Event instance identity. Clustering may group types; it does not merge instances."""
import hashlib
import re

from eventkg.lexicon import compact
from eventkg.ontology import raw_key
from eventkg.validate import explicit_date, time_bucket


def event_type_id(event):
    label = "|".join((
        compact(event.get("commodity") or ""),
        compact(event.get("indicator") or event.get("var") or ""),
        compact(event.get("dir") or ""),
    ))
    return "et_" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


def period_key(event):
    """An explicit date is an identity. Unknown time is not 本期, and a month-day is not a month."""
    explicit = explicit_date(event.get("event_time") or "") or explicit_date(event.get("time") or "")
    if explicit:
        return ("event_time", explicit)
    mention = compact(event.get("mention_date") or "")
    if re.search(r"\d{1,2}月\d{1,2}日|20\d{2}-\d{2}-\d{2}", mention):
        return ("mention_date", explicit_date(mention) or mention)
    raw = compact(event.get("time") or "")
    if not raw or time_bucket(raw) == "UNKNOWN":
        return ("unknown", "")
    return ("bucket", time_bucket(raw))


def periods_conflict(left, right):
    one, other = period_key(left), period_key(right)
    if "unknown" in (one[0], other[0]):
        return False
    return one != other


def scopes_conflict(left, right):
    """A named scope and a missing scope are not the same statistic."""
    one, other = compact(left.get("scope") or ""), compact(right.get("scope") or "")
    return one != other


_GRADE = re.compile(r"\d+(?:\.\d+)?%")
_SAMPLE = re.compile(r"\d+港")


def _marks(event, pattern):
    blob = "".join(str(event.get(key) or "") for key in (
        "commodity", "indicator", "scope", "name", "subject", "object"))
    return set(pattern.findall(blob))


def _identity_marks_conflict(left, right):
    """62% and 65%, or 47港 and 45港, are different objects. A mark on only one side is not the same object."""
    for pattern in (_GRADE, _SAMPLE):
        one, other = _marks(left, pattern), _marks(right, pattern)
        if one != other:
            return True
    return False


def _bases(event):
    found = []
    for signal in event.get("trend_signals") or []:
        basis = signal.get("comparison_basis") or ""
        if basis and basis not in found:
            found.append(basis)
    for part in re.split(r"[；;]", event.get("comparison_basis") or ""):
        if part and part not in found:
            found.append(part)
    return found


def _market_kind(event):
    blob = compact("".join(str(event.get(key) or "") for key in ("indicator", "name", "object")))
    if "期货" in blob:
        return "期货"
    if "现货" in blob:
        return "现货"
    return ""


def _subjects_conflict(left, right):
    """A missing subject is not the same party as a named one, unless the mentions corefer."""
    one, other = compact(left.get("subject") or ""), compact(right.get("subject") or "")
    if one == other:
        return False
    if one and other:
        return True
    return not _text_corefers(left, right)


def _status_conflict(left, right):
    return (left.get("status") or "") != (right.get("status") or "")


def _signal_map(event):
    """Direction of each comparison basis. One event dir is not copied onto every basis."""
    found = {}
    for signal in event.get("trend_signals") or []:
        basis = signal.get("comparison_basis") or ""
        direction = signal.get("level_direction") or ""
        if basis and direction in ("上升", "下降", "持平"):
            found[basis] = direction
    if found:
        return found
    bases = [part for part in re.split(r"[；;]", event.get("comparison_basis") or "") if part]
    direction = event.get("dir") or ""
    if len(bases) == 1 and direction in ("上升", "下降", "持平"):
        return {bases[0]: direction}
    return {}


def _direction_conflict(left, right):
    one, other = left.get("dir") or "", right.get("dir") or ""
    if one in ("", "不明") or other in ("", "不明"):
        return True
    left_map, right_map = _signal_map(left), _signal_map(right)
    shared = set(left_map) & set(right_map)
    if any(left_map[basis] != right_map[basis] for basis in shared):
        return True
    if one == other:
        return False
    if not left_map or not right_map:
        return True
    if shared or set(left_map).isdisjoint(right_map):
        return False
    return True


def _same_object(left, right):
    """Canonical indicator equality. 价格 is not 现货价格 just because one string contains the other."""
    if left.get("var") and left.get("var") == right.get("var"):
        return True
    try:
        one, other = raw_key(left).split("|")[1], raw_key(right).split("|")[1]
    except (KeyError, TypeError, ValueError, IndexError):
        one, other = compact(left.get("indicator") or ""), compact(right.get("indicator") or "")
    return bool(one and other and one == other)


def _same_commodity(left, right):
    one, other = compact(left.get("commodity") or ""), compact(right.get("commodity") or "")
    if not one or not other:
        return one == other
    return one == other


def _objects_conflict(left, right):
    one, other = compact(left.get("object") or ""), compact(right.get("object") or "")
    return bool(one and other and one != other)


def can_merge_event_instances(left, right, context="same_document"):
    """The only merge gate. Similar wording can nominate a pair; it cannot accept it.

    Unknown time or an unknown scope is not the same as a known one.
    Same indicator and same direction are not enough.
    """
    if context == "same_document" and left.get("doc_id") != right.get("doc_id"):
        return False
    if context == "cross_document" and left.get("doc_id") == right.get("doc_id"):
        return False
    if _status_conflict(left, right) or _direction_conflict(left, right):
        return False
    if not _same_commodity(left, right) or not _same_object(left, right):
        return False
    if _subjects_conflict(left, right) or _objects_conflict(left, right):
        return False
    one_market, other_market = _market_kind(left), _market_kind(right)
    if one_market and other_market and one_market != other_market:
        return False
    if scopes_conflict(left, right) or _identity_marks_conflict(left, right):
        return False
    one, other = period_key(left), period_key(right)
    if "unknown" in (one[0], other[0]) or one != other:
        return False
    if context == "cross_document" and one[0] != "event_time":
        return False
    return True


def same_instance(left, right):
    """Same document, same object, and a known period. Unknown time does not merge."""
    return can_merge_event_instances(left, right, "same_document")


def _text_corefers(left, right):
    from atlas_rag.kg_construction.commodity import outline_core

    one = compact(left.get("evidence") or "")
    other = compact(right.get("evidence") or "")
    if one and one == other:
        return True
    short, host = (one, other) if len(one) <= len(other) else (other, one)
    core = outline_core(short)
    return len(core) >= 6 and core in host


def coreferent(left, right):
    """Different documents may share an instance only with an explicit time and the same wording."""
    if not can_merge_event_instances(left, right, "cross_document"):
        return False
    return _text_corefers(left, right)


def mention_of(event, method):
    page = event.get("page")
    section = event.get("section") or ""
    if not section:
        sections = event.get("sections") or []
        section = sections[0] if sections else None
    return {
        "mention_id": "men:" + event["id"],
        "document_id": event.get("doc_id"),
        "source_document_id": event.get("doc_id"),
        "source_text": event.get("evidence") or "",
        "paragraph_id": event.get("paragraph_id") or event.get("window_id") or None,
        "sentence_id": event.get("sentence_id") or None,
        "page": page if page not in ("", None) else None,
        "section": section or None,
        "char_start": event.get("char_start") if isinstance(event.get("char_start"), int) else None,
        "char_end": event.get("char_end") if isinstance(event.get("char_end"), int) else None,
        "linked_event_instance_id": None,
        "alignment_method": method,
        "alignment_score": 1.0 if method != "unlinked" else None,
    }


def stamp_instance(item, members, method):
    """Keep the chosen event id. Mentions stay separate even when the instance is shared."""
    item["event_instance_id"] = item["id"]
    item["event_type_id"] = event_type_id(item)
    item["action"] = item.get("dir") or ""
    item["object"] = item.get("indicator") or ""
    item["region"] = item.get("scope") or ""
    key = period_key(item)
    observed = item.get("observation_period") or ""
    if not observed and key[0] in ("event_time", "mention_date", "bucket"):
        observed = key[1]
    item["observation_period"] = observed
    item["source_document_ids"] = sorted({member.get("doc_id") for member in members if member.get("doc_id")})
    mentions = []
    for member in members:
        mention = mention_of(member, method if len(members) > 1 else "unlinked")
        mention["linked_event_instance_id"] = item["id"]
        mentions.append(mention)
    item["mentions"] = mentions
    return item
