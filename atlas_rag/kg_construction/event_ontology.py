# -*- coding: utf-8 -*-
"""Event-type ontology and causal-trigger check.

The event schema follows the LNG event pipeline: a cause edge is kept only
when the source sentence contains a causal trigger, and each event can sit
under one event type. Grade-to-commodity links stay on 属于, so event types
use 类型 and do not take that relation.
"""
import re

# 「受」单独出现会命中「受阻」。因果触发必须是完整说法。
TRIGGER_RE = re.compile(r"导致|由于|因为|因此|所以|使得|造成|引发|促使|带动|受[^。\n]{0,16}影响")
ORDER_CUE = re.compile(r"之前|之后|此前|先前|先于|随后|而后|然后|继而|接着|前期|后续|此后")
CAUSAL_RELATIONS = {"结果", "因为", "导致"}
ORDER_RELATIONS = {"之前", "之后"}
KEEP_EVENT_RELATIONS = {"同时", "促进", "抑制", "必要条件", "对比", "组成"}

# Earlier rows win. Supply words are checked after weather and policy so
# “检修” stays a supply event.
EVENT_TYPES = (
    ("天气事件", ("台风", "冷空气", "寒潮", "降雨", "天气", "梅雨")),
    ("政策事件", ("政策", "反倾销", "关税", "限产", "调控")),
    ("供应事件", ("供应", "减产", "停产", "停车", "检修", "复产", "重启", "发运", "到港", "库存", "开工", "产能", "产量")),
    ("需求事件", ("需求", "采购", "下游", "成交", "消费")),
    ("国际事件", ("美联储", "地缘", "海外", "进口", "出口", "美元指数")),
)


def trigger_word(text):
    found = TRIGGER_RE.search(text or "")
    return found.group(0) if found else ""


def event_type_of(event):
    raw = event or ""
    for name, words in EVENT_TYPES:
        if any(word in raw for word in words):
            return name
    return ""


def event_status(event):
    """Forecast wording stays marked. A report date is not the event time."""
    raw = event or ""
    if any(word in raw for word in ("预计", "预测", "将", "未来", "有望", "或将")):
        return "预测中"
    return "已发生"


def _compact(text):
    return re.sub(r"\s+", "", text or "")


_ARG_TRIGGER = re.compile(r"导致|造成|致使|因此|所以|使得|因为|由于|促进|推动|带动|推高|推升|抑制|压制|限制|拖累|挤压|削弱")


def _named(region, event):
    """The event text itself must occur. A four-character slice is not a mention."""
    compact = _compact(region)
    target = _compact(event)
    return len(target) >= 2 and target in compact


def _causal_clauses(sentence):
    """A comma starts a new cause only when that side has its own trigger.

    「A下降，导致B下降」 stays one clause. 「A导致B，C推高D」 stays two.
    """
    pieces = [piece for piece in re.split(r"[，,]", sentence or "") if piece]
    merged, index = [], 0
    while index < len(pieces):
        piece = pieces[index]
        if index + 1 < len(pieces) and not _ARG_TRIGGER.search(piece) and _ARG_TRIGGER.match(pieces[index + 1]):
            merged.append(piece + "，" + pieces[index + 1])
            index += 2
        else:
            merged.append(piece)
            index += 1
    return merged


def _covers(sentence, event):
    """A non-causal mention still has to occur as a whole string, not as character slices."""
    return _named(sentence, event)


def _family_trigger(clause, relation):
    """The trigger has to belong to this relation, not to any causal word in the clause."""
    from eventkg.support import _triggers
    family = relation if _triggers(relation) else ("导致" if relation in CAUSAL_RELATIONS else "")
    allowed = _triggers(family)
    if not allowed:
        return None
    pattern = re.compile("|".join(re.escape(word) for word in sorted(allowed, key=len, reverse=True)))
    return pattern.search(clause or "")


def accept_causal(head, relation, tail, source):
    """Keep a cause only when one clause names this cause, this relation's trigger and this effect."""
    typed = relation in CAUSAL_RELATIONS or relation in ("促进", "抑制", "拖累")
    if not typed:
        return relation
    for sentence in re.split(r"[。！？\n]", source or ""):
        for clause in _causal_clauses(sentence):
            found = _family_trigger(clause, relation)
            if not found:
                continue
            if _named(clause[:found.start()], head) and _named(clause[found.end():], tail):
                return "导致" if relation in CAUSAL_RELATIONS else relation
    return None


def _same_sentence_with_order_cue(head, tail, source):
    """之前/之后 follow the order of the two mentions, not any time word in the sentence."""
    for sentence in re.split(r"[。！？\n]", source or ""):
        if not ORDER_CUE.search(sentence):
            continue
        compact = _compact(sentence)
        left, right = _compact(head), _compact(tail)
        if left not in compact or right not in compact:
            continue
        return compact.find(left) < compact.find(right)
    return False


def normalize_event_relation(head, relation, tail, source):
    """Map an event-event link onto the closed relation set, or drop it."""
    relation = relation or ""
    if relation in CAUSAL_RELATIONS or relation in ("促进", "抑制", "拖累"):
        return accept_causal(head, relation, tail, source)
    if relation in ORDER_RELATIONS:
        if not _same_sentence_with_order_cue(head, tail, source):
            return None
        return "之前" if relation == "之前" else "之后"
    if relation in KEEP_EVENT_RELATIONS - {"促进", "抑制", "拖累"}:
        if _covers(source or "", head) and _covers(source or "", tail):
            return relation
        return None
    return None
