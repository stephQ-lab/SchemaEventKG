# -*- coding: utf-8 -*-
"""Keep a model answer only when it is anchored in the source text."""
import re
from difflib import SequenceMatcher

from eventkg.corpus import sentences_of
from eventkg.observations import parse_signed_number as parse_signed_number
from eventkg.ontology import tidy_commodity, tidy_indicator, tidy_scope
from eventkg.trends import interpret_trend, percent_role, rewrite_market_label
from eventkg.lexicon import (
    CAUSAL,
    DIMENSIONS,
    INTRA_RELATIONS,
    PERIODS,
    TRIGGERS,
    canonical_commodity,
    compact,
    match_commodities,
    numbers,
)

DIRECTION_ALIASES = {
    "上升": "上升", "上涨": "上升", "增加": "上升", "回升": "上升", "累库": "上升", "走扩": "上升",
    "扩大": "上升", "走强": "上升", "走高": "上升", "抬升": "上升", "增长": "上升",
    "下降": "下降", "下跌": "下降", "减少": "下降", "回落": "下降", "去库": "下降",
    "走弱": "下降", "走低": "下降", "下滑": "下降", "下调": "下降",
    "持平": "持平", "持稳": "持平", "企稳": "持平", "稳定": "持平", "震荡": "持平", "基本持平": "持平",
    "中断": "中断", "检修": "中断", "停车": "中断", "停产": "中断", "减产": "下降",
    "恢复": "恢复", "复产": "恢复", "重启": "恢复", "增产": "上升",
    "不明": "不明", "": "不明",
}
ELLIPSIS = re.compile(r"\.{2,}|…+|。{2,}")
SAME_FACT = re.compile(r"同一(事实|事件|数据|指标|个事)|描述的是同|重复描述|即同一")
CONCESSION = re.compile(r"虽然|尽管|理论上|潜在|被.{0,8}(压倒|抵消|覆盖)")
FORECAST = re.compile(r"预计|预测|或将|有望|料将|将会|下期|下周|下月|展望|后市|短期料|预期将|或累库|或去库")
DIM_KEYWORDS = (
    ("价差", ("价差", "基差", "溢价", "折扣", "升水", "贴水")),
    ("利润", ("利润", "盈利", "毛利", "亏损")),
    ("成本", ("成本",)),
    ("物流", ("运费", "海运", "船", "压港", "物流")),
    ("库存", ("库存", "去库", "累库")),
    ("进口", ("进口", "到港")),
    ("出口", ("出口",)),
    ("需求", ("需求", "消费", "疏港", "日耗", "铁水", "采购", "成交", "订单")),
    ("供应", ("发运", "产量", "产能", "开工", "供应", "检修", "复产", "停车")),
    ("价格", ("价格", "指数", "价", "期货", "合约")),
    ("宏观金融", ("美联储", "降息", "美元", "汇率", "流动性", "货币")),
    ("政策", ("政策", "反倾销", "关税", "限产")),
    ("地缘", ("谈判", "冲突", "制裁", "地缘")),
    ("天气", ("天气", "高温", "雨季", "台风", "降雨", "寒潮")),
)


def repair_quote(quote, source, sentences=None, threshold=0.72):
    """The quote itself when it is in the source; else the closest source sentence(s)."""
    text = (quote or "").strip()
    if not text:
        return ""
    target = compact(text)
    if target and target in compact(source):
        return text
    sentences = sentences if sentences is not None else sentences_of(source)
    pieces = [piece for piece in re.split(r"(?<=[。；！？])|\.{2,}|…+|。{2,}", text) if compact(piece)]
    repaired = []
    for piece in pieces or [text]:
        found = piece.strip() if compact(piece) in compact(source) else _closest(compact(piece), sentences, threshold)
        if not found:
            return ""
        if found not in repaired:
            repaired.append(found)
    return "……".join(repaired) if len(pieces) > 1 and ELLIPSIS.search(text) else "".join(repaired)


def _closest(target, sentences, threshold):
    best, score = "", 0.0
    for sentence in sentences:
        candidate = compact(sentence)
        if target in candidate:
            return sentence
        if not candidate:
            continue
        if abs(len(candidate) - len(target)) > max(len(candidate), len(target)):
            continue
        ratio = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
        if ratio > score:
            best, score = sentence, ratio
    return best if score >= threshold else ""


def _sentence_containing(fragment, sentences):
    target = compact(fragment)
    for sentence in sentences:
        if target and target in compact(sentence):
            return sentence
    return ""


def normalize_direction(value):
    raw = compact(value)
    if raw in DIRECTION_ALIASES:
        return DIRECTION_ALIASES[raw]
    for word, mapped in DIRECTION_ALIASES.items():
        if word and word in raw:
            return mapped
    return ""


def normalize_dim(value, indicator, name):
    raw = compact(value)
    if raw in DIMENSIONS:
        return raw
    text = compact(indicator) + compact(name)
    for dim, words in DIM_KEYWORDS:
        if any(word in text for word in words):
            return dim
    return "UNKNOWN"


def normalize_period(value, time_text):
    """A day is not a month. 4月17日 stays a date, not the monthly frequency."""
    raw = compact(value)
    if raw in PERIODS:
        return raw
    text = raw + compact(time_text)
    dated = re.sub(r"20\d{2}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日", "", text)
    if re.search(r"月均|月度|当月|上月|\d+月份", dated):
        return "月度"
    if re.search(r"年内|今年|累计|年度|全年", dated):
        return "年度"
    return "当期"


def explicit_date(text):
    """2026-04-17 and 2026年4月17日 are the same day. A relative week is not a date."""
    raw = compact(text or "")
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", raw)
    if not match:
        return ""
    return "%04d-%02d-%02d" % (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def time_bucket(text):
    """A known period stays itself. An unrecognized expression is UNKNOWN, not 本期.

    本周 is the current report period. 近期, 上月, 下月, 本季度, 年初以来 and
    同期 are different windows and must not collapse into that period.
    """
    raw = compact(text)
    if not raw:
        return "UNKNOWN"
    dated = explicit_date(raw)
    if dated:
        return dated
    relative = re.sub(r"较上周|较上期|较上月|比上周|比上期", "", raw)
    if re.search(r"下期|下周", relative):
        return "下期"
    if re.search(r"上期|上周", relative):
        return "上期"
    if re.search(r"本周|本期|当周", relative):
        return "本期"
    if "下月" in relative:
        return "下月"
    if "上月" in relative:
        return "上月"
    if re.search(r"本月|当月", relative):
        return "本月"
    if "下季度" in relative:
        return "下季度"
    if "上季度" in relative:
        return "上季度"
    if "本季度" in relative or "当季" in relative:
        return "本季度"
    if re.search(r"年初以来|今年以来|年初至今", relative):
        return "年初以来"
    if "同期" in relative:
        return "同期"
    if re.search(r"近期|近来|近日|过去几周|未来一段时间|未来几周", relative):
        return "UNKNOWN"
    found = re.search(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d{1,2}月\d{1,2}日|\d{1,2}月份?", relative)
    if found and not re.fullmatch(r"\d{1,2}月", found.group(0)):
        return found.group(0)
    if found and "月" in relative and not re.search(r"本周|本期|当周", relative):
        return found.group(0)
    return "UNKNOWN"


def _mention_clause(evidence, indicator, name):
    """The clause that names this event. Another mention's 预计 or 下周 stays outside it."""
    clauses = [part.strip() for part in re.split(r"[。！？；;\n，,]", evidence or "") if part.strip()]
    needles = []
    for raw in (indicator, name):
        folded = compact(raw)
        if len(folded) >= 2:
            needles.append(folded)
    hits = []
    for clause in clauses:
        folded = compact(clause)
        if any(needle in folded for needle in needles):
            hits.append(clause)
    if len(hits) == 1:
        return hits[0]
    indicator_text = compact(indicator)
    own = [clause for clause in hits if indicator_text and indicator_text in compact(clause)]
    if len(own) == 1:
        return own[0]
    return ""


def _time_in(text):
    named = re.findall(r"下周|下期|上周|上期|下月|上月|本周|本期|本月|本季度", text or "")
    buckets = []
    for item in named:
        bucket = time_bucket(item)
        if bucket not in buckets:
            buckets.append(bucket)
    if len(buckets) == 1 and buckets[0] != "UNKNOWN":
        return named[-1]
    return ""


def _unique_mention_date(evidence, indicator):
    """Use a date only when the indicator's own clause contains exactly one."""
    text = evidence or ""
    clauses = re.split(r"[。！？；;\n]", text)
    host = next((clause for clause in clauses if indicator and indicator in clause), text)
    found = re.findall(r"20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}|\d{1,2}月\d{1,2}日", host)
    unique = []
    for item in found:
        normalized = explicit_date(item) or item
        if normalized not in unique:
            unique.append(normalized)
    if len(unique) != 1:
        return ""
    return unique[0]


def trend_name(event):
    """Node name: time, scope, object and direction. Comparison figures stay in the evidence, not the name."""
    when = time_bucket(event.get("time") or "")
    if when == "UNKNOWN":
        when = ""
    if when == "本期" and event.get("status") == "预测中" and not compact(event.get("time") or ""):
        when = "下期"
    scope = tidy_scope(event.get("scope") or "")
    indicator = event.get("indicator") or ""
    commodity = tidy_commodity(event.get("commodity") or "", indicator, scope)
    indicator = tidy_indicator(commodity, indicator, scope)
    if commodity in ("", "宏观") or commodity in indicator or commodity in scope:
        commodity = ""
    if scope and scope in indicator:
        scope = ""
    trend = {"上升": "上升", "下降": "下降", "持平": "持平", "中断": "中断", "恢复": "恢复"}.get(event.get("dir") or "", "")
    return "".join((when, scope, commodity, indicator, trend))


def clean_event(raw, source, sentences, categories, reasons=None):
    def drop(why):
        if reasons is not None:
            reasons[why] += 1
        return None

    if not isinstance(raw, dict):
        return drop("格式错误")
    sentences = sentences if sentences is not None else sentences_of(source)
    name = str(raw.get("name") or "").strip()
    if len(compact(name)) < 6:
        return drop("名称过短")
    evidence = repair_quote(str(raw.get("evidence") or ""), source, sentences)
    if not evidence:
        return drop("依据不在原文")
    scope_figures = numbers(raw.get("scope") or "") & numbers(source)
    if numbers(name) - numbers(evidence) - scope_figures:
        wider = _sentence_containing(evidence, sentences)
        if wider and not (numbers(name) - numbers(wider) - scope_figures):
            evidence = wider
        else:
            return drop("名称数字不在依据")
    direction = normalize_direction(raw.get("dir") or raw.get("direction") or "")
    if not direction:
        direction = "不明"
    indicator = rewrite_market_label(str(raw.get("indicator") or "")) or str(raw.get("indicator") or "").strip()
    if len(compact(indicator)) < 2:
        return drop("指标为空")
    scope = _field(str(raw.get("scope") or ""))
    subject = compact(str(raw.get("subject") or ""))
    separators = len(re.findall(r"[、，,；;/]", scope))
    if separators >= 2 or (separators and direction in ("中断", "恢复")):
        subject = subject or scope
        scope = ""
    commodity = canonical_commodity(str(raw.get("commodity") or ""), categories)
    if not commodity:
        hits = match_commodities(name + evidence)
        if len(hits) == 1:
            commodity = hits[0]
    host = _mention_clause(evidence, indicator, name)
    mention_time = _time_in(name) or _time_in(host)
    time_text = str(raw.get("time") or "").strip()
    time_conflict = False
    if not time_text:
        time_text = mention_time
    elif mention_time and time_bucket(time_text) not in ("", "UNKNOWN") and time_bucket(mention_time) not in ("", "UNKNOWN"):
        if time_bucket(time_text) != time_bucket(mention_time):
            time_conflict = True
            time_text = ""
    trend = interpret_trend(host or name, indicator=indicator, commodity=commodity, scope=scope, when=time_text)
    bound = trend["level_direction"]
    if direction in set(trend.get("negated") or []):
        direction = "不明"
        trend = dict(trend, level_direction="UNKNOWN", trend_signals=[])
    elif bound in ("上升", "下降", "持平") and direction in ("上升", "下降", "持平") and direction != bound:
        direction = "不明"
        trend = dict(trend, level_direction="UNKNOWN", trend_signals=[])
    elif bound in ("上升", "下降", "持平") and direction == "不明":
        direction = bound
    host_forecast = bool(FORECAST.search((host or "") + name) or time_bucket(time_text) == "下期")
    if trend.get("forecast") or host_forecast:
        status = "预测中"
        forecast_time = trend.get("forecast_time") or ""
        if forecast_time and time_text in ("", "本期", "本周"):
            time_text = forecast_time
    else:
        status = "已发生"
    value = str(raw.get("value") or "").strip()
    change = str(raw.get("change") or "").strip()
    if numbers(value) - numbers(evidence):
        value = ""
    if numbers(change) - numbers(evidence):
        change = ""
    salience = "核心" if str(raw.get("salience") or "").strip() == "核心" else "一般"
    cleaned = {
        "local_id": str(raw.get("id") or "").strip(),
        "name": name,
        "commodity": _field(commodity),
        "indicator": _field(indicator),
        "scope": scope,
        "dim": normalize_dim(raw.get("dim"), indicator, name),
        "dir": direction,
        "value": value,
        "change": change,
        "level_direction": trend["level_direction"],
        "momentum": trend["momentum"],
        "comparison_basis": trend["comparison_basis"],
        "trend_signals": trend["trend_signals"],
        "time": time_text,
        "time_conflict": time_conflict,
        "event_time": explicit_date(time_text) or explicit_date(_unique_mention_date(evidence, indicator)),
        "mention_date": _unique_mention_date(evidence, indicator),
        "period": normalize_period(raw.get("period"), time_text),
        "status": status,
        "subject": subject,
        "evidence": evidence,
        "salience": salience,
    }
    cleaned["name"] = trend_name(cleaned) or name
    _keep_identity_marks(cleaned, name + str(raw.get("indicator") or "") + str(raw.get("commodity") or "") + scope)
    cleaned["name"] = trend_name(cleaned) or cleaned["name"]
    return cleaned


def _keep_identity_marks(event, blob):
    """A grade or a sample size is part of the object, even after the display name is rebuilt."""
    text = "".join(event.get(key) or "" for key in ("commodity", "indicator", "scope", "name"))
    grades = []
    for match in re.finditer(r"\d+(?:\.\d+)?%", blob or ""):
        grade = match.group(0)
        if percent_role(blob or "", match) == "grade" and grade not in grades:
            grades.append(grade)
    if len(grades) == 1 and grades[0] not in text:
        event["commodity"] = grades[0] + (event.get("commodity") or "")
        text += grades[0]
    for sample in re.findall(r"\d+港", blob or ""):
        if sample not in text:
            event["scope"] = sample if not event.get("scope") else event["scope"]
            text += sample


def _field(text):
    return compact(text).replace("|", "/")


def anchors(event):
    """Short strings that identify an event inside a quote."""
    found = []
    for field in ("indicator", "scope", "subject"):
        value = compact(event.get(field) or "")
        if len(value) >= 2:
            found.append(value)
    commodity = event.get("commodity") or ""
    if len(commodity) >= 2 and commodity != "宏观":
        found.append(commodity)
    name = compact(event.get("name") or "")
    for size in (6, 4):
        for start in range(0, max(0, len(name) - size + 1), 2):
            found.append(name[start:start + size])
    return found


def covers(quote, event):
    text = compact(quote)
    if not text:
        return False
    evidence = compact(event.get("evidence") or "")
    if evidence and (evidence[:12] in text or text[:12] in evidence):
        return True
    indicator = compact(event.get("indicator") or "")
    if len(indicator) >= 2 and indicator in text:
        return True
    figures = numbers(event.get("name") or "")
    if figures and figures <= numbers(quote):
        return True
    hits = [piece for piece in anchors(event) if piece in text]
    return len(hits) >= 2


def _positions(text, sentences):
    found = set()
    for piece in ELLIPSIS.split(text or ""):
        target = compact(piece)
        if not target:
            continue
        for index, sentence in enumerate(sentences):
            candidate = compact(sentence)
            if candidate and (candidate in target or target in candidate):
                found.add(index)
    return found


def near(quote, event, sentences, distance=1):
    """The event's own evidence sits in the quote or in a sentence next to it."""
    quoted = _positions(quote, sentences)
    own = _positions(event.get("evidence") or "", sentences)
    return any(abs(left - right) <= distance for left in quoted for right in own)


def has_trigger(relation, quote, trigger):
    pattern = TRIGGERS[relation]
    text = quote or ""
    if pattern.search(text):
        return True
    word = compact(trigger)
    return bool(word) and word in compact(text) and bool(pattern.search(word) or TRIGGERS["导致"].search(word))


def clean_relation(raw, events_by_id, source, sentences, inferred_allowed=False, reasons=None):
    """A checked relation, or None. reasons (a Counter) collects why relations were dropped."""
    def drop(why):
        if reasons is not None:
            reasons[why] += 1
        return None

    if not isinstance(raw, dict):
        return drop("格式错误")
    relation = str(raw.get("type") or raw.get("relation") or "").strip()
    head = str(raw.get("head") or "").strip()
    tail = str(raw.get("tail") or "").strip()
    if relation not in INTRA_RELATIONS:
        return drop("关系类型不在词表")
    if head == tail:
        return drop("自环")
    sentences = sentences if sentences is not None else sentences_of(source)
    if head not in events_by_id or tail not in events_by_id:
        return drop("端点事件未通过校验")
    basis = str(raw.get("basis") or "原文明示").strip()
    if basis not in ("原文明示", "综合推断") or (basis == "综合推断" and not inferred_allowed):
        basis = "原文明示"
    quote = repair_quote(str(raw.get("evidence") or raw.get("quote") or ""), source, sentences)
    if not quote:
        return drop("依据不在原文")
    trigger = str(raw.get("trigger") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    if relation in CAUSAL and "受阻" in quote and not TRIGGERS["导致"].search(quote.replace("受阻", "")):
        return drop("受阻不是因果")
    left, right = events_by_id[head], events_by_id[tail]
    if SAME_FACT.search(reason) or (relation == "构成" and numbers(left["name"]) and
                                    numbers(left["name"]) == numbers(right["name"])):
        return drop("同一事实")
    if relation in ("之前", "之后") and left["status"] != right["status"]:
        return drop("本期与预测的时序")
    if CONCESSION.search(reason):
        return drop("推断理由自相矛盾")
    left_ok = covers(quote, left) or near(quote, left, sentences)
    right_ok = covers(quote, right) or near(quote, right, sentences)
    triggered = relation == "构成" or has_trigger(relation, quote, trigger)
    if basis == "原文明示":
        if not triggered:
            if not (inferred_allowed and reason):
                return drop("缺触发词")
            basis = "综合推断"
        elif not (left_ok and right_ok):
            if not (left_ok or right_ok):
                return drop("依据定位不到两端")
            basis = "综合推断"
            reason = reason or "依据句含触发词，另一端在上下文中"
    if basis == "综合推断":
        if not reason:
            return drop("推断缺理由")
        if not (left_ok or right_ok):
            return drop("推断依据定位不到任一端")
    confidence = 0.9 if basis == "原文明示" else 0.6
    return {"head": head, "tail": tail, "type": relation, "trigger": trigger, "evidence": quote,
            "basis": basis, "reason": reason, "confidence": confidence}
