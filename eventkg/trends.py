# -*- coding: utf-8 -*-
"""Trend reading bound to one indicator mention. Market quantities stay out of the query label."""
import re

from eventkg.ontology import _alias_indicator

_LEVEL = {
    "上涨": "上升", "上升": "上升", "增加": "上升", "回升": "上升", "累库": "上升",
    "走强": "上升", "走高": "上升", "增长": "上升", "持涨": "上升",
    "下跌": "下降", "下降": "下降", "减少": "下降", "回落": "下降", "去库": "下降",
    "走弱": "下降", "走低": "下降", "下滑": "下降",
    "持平": "持平", "持稳": "持平",
}
_PRED = re.compile(
    r"上涨|下跌|上升|下降|增加|减少|回升|回落|累库|去库|持涨|持平|持稳|走强|走弱|增长|下滑|收窄|扩大|走扩")
_SPLIT = re.compile(r"但是|但|而|[，、。；;\n]")
_NEG = re.compile(r"(?:并未|并没有|没有|不是|并非|并不|未|无)$")
_FORECAST = re.compile(r"展望|预计|或将|料将|预期将")
_WHEN = re.compile(r"下周|下期|上周|上期|本周|本期|本月|上月")
_ADVERB = re.compile(r"^(仍然|仍旧|依旧|依然|还是|继续|进一步|明显|大幅)+")
_GRADE_TAIL = re.compile(
    r"^(?:铁精粉|澳粉|精粉|球团粉|球团|块矿|粉矿|超特粉|混合粉|PB粉|纽曼粉|金布巴|卡粉|品位|[粉块矿品])")
_CHANGE_BEFORE = re.compile(r"(上涨|下跌|上升|下降|增加|减少|回升|回落|累库|去库|涨幅|跌幅|降幅|幅度|变动|变化|持涨|下滑)$")
_MARKET_UNIT = r"(?:美元/干吨|美元/吨|美元|元/吨|元|万吨|吨|个百分点|点)"
_MARKET_QTY = re.compile(r"[+\-−－]?\d+(?:\.\d+)?\s*" + _MARKET_UNIT)
_PURE_MARKET = re.compile(r"^[+\-−－]?\d+(?:\.\d+)?(?:" + _MARKET_UNIT + r"|%)$")
_BARE_DIRECTION = {
    "上涨", "下跌", "上升", "下降", "增加", "减少", "持平", "累库", "去库", "回升", "回落", "持涨",
}
_FILLER = re.compile(
    r"较去年同期|较年初|周环比|月环比|同比|环比|展望|预计|或将|去年|今年|同期|本周|上周|下周|本期|上期|下期|本月|上月|当日|昨日")


def _clauses(text):
    return [part.strip() for part in _SPLIT.split(text or "") if part and part.strip()]


def _clean_subject(text):
    cleaned = _FILLER.sub("", text or "")
    cleaned = _MARKET_QTY.sub("", cleaned)
    cleaned = re.sub(r"\d+(?:\.\d+)?%", "", cleaned)
    return cleaned.strip(" 的约为近较比")


def _negated(prefix):
    tail = re.sub(r"(明显|大幅|继续|进一步)$", "", (prefix or "")[-8:])
    return bool(_NEG.search(tail))


def _basis_of(span):
    """The comparison word that belongs to this predicate, not to a later one in the same sentence."""
    if "同比" in span or "去年同期" in span:
        return "同比"
    if "较年初" in span:
        return "较年初"
    if "环比" in span or "较上周" in span or "较上期" in span or "较上月" in span or "比上周" in span or "比上期" in span:
        return "环比"
    return ""


def _when_of(span):
    kept = []
    for match in _WHEN.finditer(span or ""):
        if span[max(0, match.start() - 1):match.start()] in ("较", "比"):
            continue
        kept.append(match.group(0))
    return kept[-1] if kept else ""


def _bucket(text):
    raw = text or ""
    if "下周" in raw or "下期" in raw:
        return "下期"
    if "上周" in raw or "上期" in raw:
        return "上期"
    if raw:
        return "本期"
    return ""


def percent_role(text, match):
    """A percent before 澳粉 or 精粉 is a grade. A percent after a direction is a change."""
    before = (text or "")[max(0, match.start() - 8):match.start()]
    after = (text or "")[match.end():].lstrip()
    if _CHANGE_BEFORE.search(before):
        return "change"
    if _GRADE_TAIL.match(after):
        return "grade"
    return "change"


def _momentum_level(clause):
    if "跌幅" in clause or "降幅" in clause:
        return "下降", "收窄" if "收窄" in clause else ("扩大" if "扩大" in clause or "走扩" in clause else "")
    if "涨幅" in clause:
        widened = "扩大" in clause or "走扩" in clause
        return "上升", "收窄" if "收窄" in clause else ("扩大" if widened else "")
    if "收窄" in clause:
        return "", "收窄"
    if "扩大" in clause or "走扩" in clause:
        return "", "扩大"
    return "", ""


def _one(subject, level, momentum, basis, negated, forecast, when):
    return {
        "subject": subject, "level": level, "momentum": momentum, "basis": basis,
        "negated": negated, "forecast": forecast,
        "forecast_time": when if when in ("下周", "下期") else "",
        "when": when,
    }


def _readings(text):
    """One reading per predicate. Time and comparison basis stay with that predicate."""
    found = []
    pending = ""
    for clause in _clauses(text):
        when = _when_of(clause)
        forecast = bool(_FORECAST.search(clause) or when in ("下周", "下期"))
        if re.search(r"(涨幅|跌幅|降幅|幅度).{0,6}(收窄|扩大|走扩)", clause):
            pred = _PRED.search(clause)
            if pred and pred.group(0) in ("收窄", "扩大", "走扩"):
                level, momentum = _momentum_level(clause)
                subject = pending
                if found and (not subject or found[-1]["subject"] == subject) and found[-1]["when"] == when:
                    found[-1]["momentum"] = momentum or found[-1]["momentum"]
                    if level and not found[-1]["level"]:
                        found[-1]["level"] = level
                    found[-1]["forecast"] = found[-1]["forecast"] or forecast
                else:
                    found.append(_one(subject, level, momentum, _basis_of(clause[:pred.start()]), False, forecast, when))
                continue
        predicates = list(_PRED.finditer(clause))
        if not predicates:
            noun = _ADVERB.sub("", _clean_subject(clause))
            if noun:
                pending = noun
            continue
        previous = 0
        for index, pred in enumerate(predicates):
            local = clause[previous:pred.start()]
            previous = pred.end()
            if pred.group(0) in ("收窄", "扩大", "走扩") and re.search(r"涨幅|跌幅|降幅|幅度", clause):
                level, momentum = _momentum_level(clause)
                if found and found[-1]["subject"] == (pending or found[-1]["subject"]) and found[-1]["when"] == when:
                    found[-1]["momentum"] = momentum or found[-1]["momentum"]
                    if level and not found[-1]["level"]:
                        found[-1]["level"] = level
                continue
            bare = re.sub(r"(明显|大幅|继续|进一步)+$", "", local)
            bare = re.sub(r"(并未|并没有|没有|不是|并非|并不|未|无)+$", "", bare)
            subject = _ADVERB.sub("", _clean_subject(bare)) if index == 0 else ""
            if subject:
                pending = subject
            else:
                subject = pending
            word = pred.group(0)
            if word in ("收窄", "扩大", "走扩"):
                level, momentum = "", ("收窄" if word == "收窄" else "扩大")
            else:
                level, momentum = _LEVEL.get(word, ""), ""
            if re.search(r"(涨幅|跌幅|降幅).{0,4}收窄", local + word):
                momentum = "收窄"
            elif re.search(r"(涨幅|跌幅|降幅).{0,4}(扩大|走扩)", local + word):
                momentum = "扩大"
            found.append(_one(subject, level, momentum, _basis_of(local), _negated(local), forecast, when))
    return found


def _same_mention(subject, indicator, commodity, scope):
    if not subject or not indicator:
        return False
    core = subject
    if scope and core.startswith(scope):
        core = core[len(scope):]
    core = re.sub(r"^\d+港", "", core)
    if commodity and core.startswith(commodity):
        core = core[len(commodity):]
    core = core.strip("的 ")
    if not core:
        return False
    return _alias_indicator(core, scope) == _alias_indicator(indicator, scope)


def _assemble(readings):
    affirmed = [item for item in readings if item["level"] in ("上升", "下降", "持平") and not item["negated"]]
    subjects = {item["subject"] for item in affirmed if item["subject"]}
    levels = {item["level"] for item in affirmed}
    if len(subjects) > 1 and len(levels) > 1:
        level = "UNKNOWN"
        primary = None
    else:
        primary = next((item for item in affirmed if item["basis"] == "环比"), None)
        if primary is None and affirmed:
            primary = affirmed[0]
        level = primary["level"] if primary else "UNKNOWN"
    signals = []
    seen = set()
    for item in affirmed:
        key = (item["basis"], item["level"], item["momentum"])
        if key in seen:
            continue
        seen.add(key)
        signals.append({
            "comparison_basis": item["basis"], "level_direction": item["level"], "momentum": item["momentum"],
        })
    bases = []
    for item in signals:
        if item["comparison_basis"] and item["comparison_basis"] not in bases:
            bases.append(item["comparison_basis"])
    momentum = next((item["momentum"] for item in affirmed if item["momentum"]), "")
    if not momentum:
        momentum = next((item["momentum"] for item in readings if item["momentum"]), "")
    forecast_rows = affirmed or readings
    current = [item for item in affirmed if not item["forecast"]]
    forecast = bool(affirmed) and not current
    forecast_time = ""
    if forecast:
        forecast_time = next((item["forecast_time"] for item in affirmed if item["forecast_time"]), "")
    negated = []
    for item in readings:
        if item["negated"] and item["level"] in ("上升", "下降", "持平") and item["level"] not in negated:
            negated.append(item["level"])
    return {
        "level_direction": level,
        "momentum": momentum,
        "comparison_basis": "；".join(bases),
        "trend_signals": signals,
        "negated": negated,
        "forecast": forecast,
        "forecast_time": forecast_time,
    }


def interpret_trend(text, indicator="", commodity="", scope="", when=""):
    """Read the mention of this indicator at this time, not every predicate in the sentence.

    本周价格上涨 and 下周价格下跌 stay apart. 环比 and 同比 are read from the words
    next to each predicate, including when a顿号 joins them.
    """
    readings = _readings(text)
    if indicator:
        readings = [item for item in readings if _same_mention(item["subject"], indicator, commodity, scope)]
        if when:
            bucket = _bucket(when)
            matched = [item for item in readings if item["when"] and _bucket(item["when"]) == bucket]
            untimed = [item for item in readings if not item["when"]]
            other = [item for item in readings if item["when"] and _bucket(item["when"]) != bucket]
            if matched or not other:
                readings = matched + untimed
            elif len({_bucket(item["when"]) for item in other}) == 1:
                readings = other
            else:
                readings = []
        else:
            buckets = {_bucket(item["when"]) for item in readings if item["when"]}
            if len(buckets) > 1:
                readings = [item for item in readings if not item["when"]]
    return _assemble(readings)


def same_basis_conflict(signals):
    """Opposite directions conflict only inside one comparison basis."""
    by_basis = {}
    for item in signals or []:
        basis = item.get("comparison_basis") or ""
        direction = item.get("level_direction") or ""
        if not basis or direction not in ("上升", "下降"):
            continue
        previous = by_basis.get(basis)
        if previous and previous != direction:
            return True
        by_basis[basis] = direction
    return False


def _strip_quantities(raw):
    """Remove change amounts. Keep a grade percent that names the product in front of it."""
    protected = []

    def protect(match):
        if percent_role(raw, match) == "grade":
            token = "\x00%d\x00" % len(protected)
            protected.append(match.group(0))
            return token
        return ""

    cleaned = re.sub(r"\d+(?:\.\d+)?%", protect, raw)
    cleaned = _MARKET_QTY.sub("", cleaned)
    for index, grade in enumerate(protected):
        cleaned = cleaned.replace("\x00%d\x00" % index, grade)
    return cleaned


def rewrite_market_label(text):
    """Drop a market quantity and keep the object, grade, scope and date.

    价格上涨1.65% becomes 价格上涨. 65%精粉价格上涨1.75美元/吨 keeps 65%精粉.
    A bare 上涨 or 下降 is not an event node.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    compact = re.sub(r"[\s,，]", "", raw)
    if _PURE_MARKET.fullmatch(compact):
        return ""
    cleaned = re.sub(r"\s+", "", _strip_quantities(raw))
    cleaned = re.sub(r"^[，,、；;约近]+", "", cleaned)
    cleaned = re.sub(r"[，,、；;至到约近]+$", "", cleaned)
    if cleaned in _BARE_DIRECTION or not cleaned:
        return ""
    return cleaned
