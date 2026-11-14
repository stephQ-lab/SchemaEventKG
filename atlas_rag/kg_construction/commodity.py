# -*- coding: utf-8 -*-
"""Commodity names and label-merge guards for document-level linking.

This is part of the extraction pipeline. It does not build a graph by itself.
"""
import re

# name, parent, aliases. Longer aliases are matched first.
LEXICON = [
    ("铁矿石", "", ["铁矿石", "铁矿"]),
    ("PB粉", "铁矿石", ["PB粉"]),
    ("金布巴粉", "铁矿石", ["金布巴粉", "金布巴"]),
    ("麦克粉", "铁矿石", ["麦克粉"]),
    ("纽曼粉", "铁矿石", ["纽曼粉"]),
    ("卡拉加斯粉", "铁矿石", ["卡拉加斯粉", "卡拉加斯", "卡粉"]),
    ("巴混", "铁矿石", ["巴混"]),
    ("罗伊山粉", "铁矿石", ["罗伊山粉", "罗伊山"]),
    ("62%澳粉", "铁矿石", ["62%澳粉"]),
    ("65%巴西粉", "铁矿石", ["65%巴西粉"]),
    ("铁水", "铁矿石", ["铁水"]),
    ("球团", "铁矿石", ["球团"]),
    ("块矿", "铁矿石", ["块矿"]),
    ("螺纹钢", "", ["螺纹钢", "螺纹"]),
    ("热轧板卷", "", ["热轧板卷", "热轧"]),
    ("冷轧板卷", "", ["冷轧板卷", "冷轧"]),
    ("钢坯", "", ["钢坯"]),
    ("焦炭", "", ["焦炭"]),
    ("炼焦煤", "", ["炼焦煤", "焦煤"]),
    ("动力煤", "", ["动力煤"]),
    ("喷吹煤", "", ["喷吹煤"]),
    ("兰炭", "", ["兰炭"]),
    ("电石", "", ["电石"]),
    ("烧碱", "", ["烧碱"]),
    ("氯乙烯", "", ["氯乙烯"]),
    ("电解铜", "", ["电解铜", "废铜"]),
    ("碳酸锂", "", ["碳酸锂"]),
    ("锂辉石", "", ["锂辉石"]),
    ("磷酸铁锂", "", ["磷酸铁锂"]),
    ("磷酸铁", "", ["磷酸铁"]),
    ("甲醇", "", ["甲醇"]),
    ("PVC", "", ["PVC", "聚氯乙烯"]),
    ("电石法PVC", "PVC", ["电石法PVC", "电石法"]),
    ("乙烯法PVC", "PVC", ["乙烯法PVC", "乙烯法"]),
    ("丙烯", "", ["丙烯"]),
    ("乙烯", "", ["乙烯"]),
    ("苯乙烯", "", ["苯乙烯"]),
    ("辛醇", "", ["辛醇"]),
    ("正丁醇", "", ["正丁醇"]),
    ("苯酐", "", ["苯酐"]),
    ("DOP", "", ["DOP"]),
    ("DOTP", "", ["DOTP"]),
    ("尿素", "", ["尿素"]),
    ("豆粕", "", ["豆粕"]),
    ("菜粕", "", ["菜粕"]),
    ("棉粕", "", ["棉粕"]),
    ("豆油", "", ["豆油"]),
    ("菜油", "", ["菜油"]),
    ("棕榈油", "", ["棕榈油"]),
    ("玉米", "", ["玉米"]),
    ("玉米淀粉", "玉米", ["玉米淀粉"]),
    ("玉米蛋白粉", "玉米", ["玉米蛋白粉"]),
    ("玉米胚芽粕", "玉米", ["玉米胚芽粕"]),
    ("木薯淀粉", "", ["木薯淀粉"]),
    ("马铃薯淀粉", "", ["马铃薯淀粉"]),
    ("棉花", "", ["新疆棉", "棉花"]),
    ("生猪", "", ["生猪", "外三元"]),
    ("锰矿", "", ["锰矿"]),
    ("锰铁", "", ["锰铁", "高碳锰铁", "中碳锰铁", "低碳锰铁"]),
    ("钼", "", ["钼铁", "钼精矿"]),
    ("稀土", "", ["稀土", "镨钕", "镝铽"]),
    ("水泥", "", ["水泥"]),
    ("石油树脂", "", ["石油树脂"]),
]

BLOCK_GROUPS = [
    ["DOP", "DOTP"],
    ["散装", "袋装"],
    ["电石法", "乙烯法"],
    ["冷轧", "热轧"],
    ["高碳", "中碳", "低碳"],
    ["PB粉", "金布巴", "麦克粉", "纽曼", "卡拉加斯", "罗伊山", "巴混"],
    ["62%澳粉", "65%巴西粉", "PB粉"],
]

PLACES = [
    "东莞", "钦州", "天津", "青岛", "日照", "上海", "宁波", "福州", "厦门",
    "唐山", "邯郸", "石家庄", "北京", "广州", "成都", "重庆",
]

_ALIAS_ROWS = []
for _name, _parent, _aliases in LEXICON:
    for _alias in _aliases:
        _ALIAS_ROWS.append((len(_alias), _alias, _name))
_ALIAS_ROWS.sort(reverse=True)
ALIASES = {name: aliases for name, _, aliases in LEXICON}
PARENT = {name: parent for name, parent, _ in LEXICON}


def norm(text):
    return re.sub(r"[\s，,。；;、：:（）()\[\]【】\"'“”]+", "", text or "")


def outline_core(text):
    """Drop a contents label such as 「库存：」 so the claim can match the body sentence."""
    return re.sub(r"^(供应|需求|库存|价格|成本|利润)", "", norm(text))


def numbers(text):
    return set(re.findall(r"-?\d+(?:\.\d+)?", text or ""))


_TIME = re.compile(
    r"\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|截至\d{1,2}月\d{1,2}日|\d{1,2}月份|\d{1,2}月"
)
_IDENTITY = re.compile(r"\d+(?:\.\d+)?(?:%|港|家)")
_MEASURE = re.compile(r"-?\d+(?:\.\d+)?(?:万吨/天|万吨|吨/天|美元/干吨|美元/吨|美元|元/吨|元|个百分点|个基点)?")
_COMPARE = re.compile(r"周环比|月环比|环比|同比|较年初|较去年同期|较同期|较上期|较上周")
_FILLER = re.compile(r"截止目前|截至目前|目前|总量|合计|整体|大约|约")
_TREND_ONLY = {"上涨", "下跌", "上升", "下降", "增加", "减少", "回升", "回落", "累库", "去库", "持平",
               "走弱", "走强", "扩大", "收窄", "增长", "下滑", "持稳"}


def identity_marks(text):
    """Sample and grade tokens that distinguish a thing: 47港, 247家, 62%."""
    return set(_IDENTITY.findall(norm(text)))


def strip_measures(text):
    """Drop comparison figures. Keep time, and keep 47港 / 247家 / 62% because those name the thing."""
    raw = norm(text)
    held = []

    def hold(match):
        held.append(match.group(0))
        return "〔%s〕" % ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(held) - 1])

    raw = _TIME.sub(hold, raw)
    raw = _IDENTITY.sub(hold, raw)
    raw = _MEASURE.sub("", raw)
    raw = _COMPARE.sub("", raw)
    raw = _FILLER.sub("", raw)
    raw = re.sub(r"(至|达到|升至|降至)+", "", raw)
    for index, value in enumerate(held):
        raw = raw.replace("〔%s〕" % "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index], value)
    return raw


def concrete_label(text):
    """A node name: the thing, the time and the trend. A bare number or a bare direction is not a node."""
    stripped = re.sub(r"^[的之至到约为]+|[的左右附近上下]+$", "", strip_measures(text))
    if len(stripped) < 2 or stripped in _TREND_ONLY:
        return ""
    return stripped


# Longer sources first. A source is replaced only when its span is still free, so 日均铁水产量 is not rewritten twice.
MENTION_ALIASES = (
    ("远期现货价格指数", "远期价格指数"),
    ("远期现货指数", "远期价格指数"),
    ("现货价格指数", "价格指数"),
    ("港口库存总量", "港口库存"),
    ("进口铁矿库存", "港口库存"),
    ("铁水日均铁水产量", "日均铁水产量"),
    ("铁水日均产量", "日均铁水产量"),
    ("钢厂盈利率", "盈利率"),
    ("高炉炼铁产能利用率", "高炉产能利用率"),
    ("日均疏港量", "疏港量"),
    ("样本钢厂", "钢厂"),
    ("周环比", ""),
    ("月环比", ""),
    ("环比", ""),
    ("同比", ""),
    ("较年初", ""),
    ("较去年同期", ""),
    ("较同期", ""),
    ("较上周", ""),
)
_SOURCE_PREFIXES = ("Mysteel", "mysteel", "我的钢铁网", "我的钢铁", "钢联")
_UP = "累库|上涨|上升|增加|回升|走强|增长"
_DOWN = "去库|下跌|下降|减少|回落|走弱|下滑"
_FLAT = "持稳|企稳|持平"


def _apply_aliases(text):
    taken = [False] * len(text)
    hits = []
    for source, target in sorted(MENTION_ALIASES, key=lambda item: -len(item[0])):
        start = 0
        while True:
            found = text.find(source, start)
            if found < 0:
                break
            end = found + len(source)
            if not any(taken[found:end]):
                hits.append((found, end, target))
                for index in range(found, end):
                    taken[index] = True
            start = found + 1
    if not hits:
        return text
    parts, cursor = [], 0
    for start, end, target in sorted(hits):
        parts.append(text[cursor:start])
        parts.append(target)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _trend_suffix(text):
    matched = re.search(r"(?:%s|%s|%s)+$" % (_UP, _DOWN, _FLAT), text)
    if not matched:
        return text
    word = matched.group(0)
    last = max(((word.rfind(piece), piece) for piece in re.findall(r"(?:%s|%s|%s)" % (_UP, _DOWN, _FLAT), word)),
               default=(-1, ""))[1]
    if re.search(_DOWN, last):
        trend = "下降"
    elif re.search(_FLAT, last):
        trend = "持平"
    else:
        trend = "上升"
    return text[:matched.start()] + trend


def canonical_mention(text):
    """One name for a thing: time, identity (47港、62%) and a trend. Comparison figures are not part of the name."""
    label = concrete_label(text)
    if not label:
        return ""
    for prefix in _SOURCE_PREFIXES:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = _trend_suffix(_apply_aliases(label))
    label = re.sub(r"^(的|之|至|到|约为)+|(的|左右|附近)$", "", label)
    if len(label) < 2 or label in _TREND_ONLY or label in {"上升", "下降", "持平"}:
        return ""
    return label


def collapse_mentions(labels):
    """Map every raw label to one canonical name.

    Exact wording, a shorter cut of the same sentence, and an alias of the same
    indicator share a name. A short label that fits two different longer labels is left as it is.
    """
    canon = {label: canonical_mention(label) for label in labels}
    forms = sorted({value for value in canon.values() if value}, key=len)
    redirect = {form: form for form in forms}
    for short in forms:
        hosts = [longer for longer in forms
                 if longer != short and short in longer and len(short) >= 6
                 and len(short) / len(longer) >= 0.34 and block_reason(short, longer) == ""]
        if len(hosts) == 1:
            redirect[short] = hosts[0]

    def resolve(form):
        seen = set()
        while form and redirect.get(form, form) != form and form not in seen:
            seen.add(form)
            form = redirect[form]
        return form

    return {label: resolve(form) for label, form in canon.items()}


def match_commodities(text):
    """Return commodity names whose aliases occur in text. Longer aliases win a span."""
    raw = norm(text)
    taken = [False] * len(raw)
    found = []
    for _, alias, name in _ALIAS_ROWS:
        start = 0
        while True:
            i = raw.find(alias, start)
            if i < 0:
                break
            end = i + len(alias)
            if not any(taken[i:end]):
                for k in range(i, end):
                    taken[k] = True
                if name not in found:
                    found.append(name)
                break
            start = i + 1
    return found


def mentions(text, commodity):
    raw = norm(text)
    return any(alias in raw for alias in ALIASES.get(commodity, [commodity]))


_HEADLINE = ("日报", "周报", "周刊", "月报")


def category_names():
    """Kinds named by a report title, such as 铁矿石, PVC, 甲醇. Grades are not kinds."""
    return [name for name, parent, _ in LEXICON if not parent]


def match_categories(text):
    """Like match_commodities, but only names with no parent."""
    raw = norm(text)
    taken = [False] * len(raw)
    found = []
    rows = []
    for name, parent, aliases in LEXICON:
        if parent:
            continue
        for alias in aliases:
            rows.append((len(alias), alias, name))
    rows.sort(reverse=True)
    for _, alias, name in rows:
        start = 0
        while True:
            i = raw.find(alias, start)
            if i < 0:
                break
            end = i + len(alias)
            if not any(taken[i:end]):
                for k in range(i, end):
                    taken[k] = True
                if name not in found:
                    found.append(name)
                break
            start = i + 1
    return found


def document_categories(doc_id="", title="", file_name="", text=""):
    """Categories of one article: kind names in the title and the masthead.

    A grade such as PB粉 stays out of this list even when the body is full of it.
    """
    pieces = []
    for part in (doc_id, title, file_name):
        if part:
            pieces.append(str(part))
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        pieces.append(lines[0])
    for line in lines:
        if any(mark in line for mark in _HEADLINE):
            pieces.append(line)
    found = []
    for name in match_categories("\n".join(pieces)):
        if name not in found:
            found.append(name)
    return found


def classify_event(event, categories):
    """Map one event onto the article's categories and the grades named in it.

    Returns (category names to link with 涉及, grade names to keep as entities).
    With one article category, any lexicon hit belongs to that category.
    """
    categories = [name for name in (categories or []) if name in PARENT and not PARENT[name]]
    hits = match_commodities(event)
    if not hits or not categories:
        return [], []
    if len(categories) == 1:
        category = categories[0]
        return [category], [hit for hit in hits if hit != category]
    chosen = []
    members = []
    for hit in hits:
        parent = PARENT.get(hit, "")
        if hit in categories:
            if hit not in chosen:
                chosen.append(hit)
        elif parent in categories:
            if parent not in chosen:
                chosen.append(parent)
            if hit not in members:
                members.append(hit)
        elif hit not in members:
            members.append(hit)
    if not chosen:
        return [], []
    return chosen, [hit for hit in members if hit not in chosen]


_ORG_SUFFIX = ("石化", "化工", "集团", "公司", "能源", "矿业", "钢铁")


def belongs_edge(member, target):
    """属于 is only the variety-to-category link, such as PB粉 -> 铁矿石."""
    if not member or not target:
        return False
    return PARENT.get(member, "") == target


def member_targets(member, categories):
    """Grades belong to their lexicon parent when that parent is this article's category."""
    categories = list(categories or [])
    if not member or member not in PARENT or member in categories:
        return []
    parent = PARENT.get(member, "")
    if parent and parent in categories:
        return [parent]
    return []


def shared_scope(name, ntype, file_scope):
    """One node for a category, a variety, a place, or a named organization.

    Events, evidence and devices without a company stay on the document that
    mentioned them, so two articles can still be told apart.
    """
    file_scope = file_scope or ""
    if ntype in ("commodity", "event_type", "concept", "document"):
        return ""
    if ntype == "event":
        return file_scope
    if ntype == "passage":
        return file_scope or name or ""
    if not name:
        return file_scope
    if name in PARENT or name in PLACES:
        return ""
    if "装置" in name and not any(suffix in name.split("装置", 1)[0] for suffix in _ORG_SUFFIX):
        return file_scope
    if len(name) >= 3 and any(name.endswith(suffix) for suffix in _ORG_SUFFIX):
        return ""
    return file_scope


def block_reason(a, b):
    """Empty string means the two labels may be the same fact.

    A figure of a price or a volume is a trend, not a different thing. 47港 and 45港 stay apart.
    """
    marks_a, marks_b = identity_marks(a), identity_marks(b)
    if marks_a and marks_b and marks_a != marks_b:
        return "口径不同"
    pa = [p for p in PLACES if p in a]
    pb = [p for p in PLACES if p in b]
    if pa != pb and (pa or pb):
        return "地点不同"
    for group in BLOCK_GROUPS:
        ha = [g for g in group if g in a]
        hb = [g for g in group if g in b]
        if ha != hb and (ha or hb):
            return "规格或品种不同"
    return ""
