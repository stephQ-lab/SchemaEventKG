# -*- coding: utf-8 -*-
"""Map raw (commodity, indicator, scope, period) strings to canonical state variables."""
import hashlib
import json
import re
from collections import Counter

from eventkg.lexicon import PERIODS, category_of, compact, numbers
from eventkg.llm import parallel_map
from eventkg.prompts import ONTOLOGY_SYSTEM

BATCH = 60
PREFIXES = ("Mysteel", "mysteel", "我的钢铁", "钢联")
GENERIC_SCOPES = {"全国", "国内", "中国", "我国", "国内市场", "全国范围"}
MILL_INDICATORS = re.compile(r"盈利率|高炉开工率|高炉产能利用率|高炉检修")
STEEL = {"螺纹钢", "热轧板卷", "冷轧板卷", "钢坯", "钢材", "成材", "五大材", "钢厂"}


def tidy_scope(scope):
    text = compact(scope)
    if text in GENERIC_SCOPES:
        return "全国"
    mill = re.fullmatch(r"(\d+)(?:家|个)?(?:样本|主流|重点)?(?:钢厂|钢企|钢铁企业)", text)
    if mill:
        return mill.group(1) + "家钢厂"
    port = re.fullmatch(r"(?:全国)?(\d+)(?:个)?(?:主要)?(?:港口|港)", text)
    if port:
        return port.group(1) + "港"
    parts = re.split(r"[、+＋和与及/]", text)
    if 2 <= len(parts) <= 3 and all(1 <= len(part) <= 4 for part in parts):
        return "+".join(parts)
    return text


def tidy_commodity(commodity, indicator, scope):
    """Mill-level measures belong to the mill, not to whichever steel product the sentence named."""
    if "铁水" in indicator or ("高炉" in indicator and not re.search(r"盈利|开工|产能利用", indicator)):
        return "铁水"
    if MILL_INDICATORS.search(indicator) and (commodity in STEEL or "钢厂" in scope):
        return "钢厂"
    return commodity


def tidy_mill(commodity, indicator, scope):
    """Mysteel's 247-mill sample is the default for hot metal and mill measures; its spellings are one indicator."""
    if commodity == "铁水" and re.fullmatch(r"(日均)?(铁水)?(日均)?(铁水)?(产量|日产|日均产量)(日均)?", indicator):
        indicator = "日均铁水产量"
    return indicator, scope


def tidy_indicator(commodity, indicator, scope):
    text = compact(indicator)
    for prefix in PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    scope_text = compact(scope)
    if scope_text and scope_text in text and text != scope_text:
        text = text.replace(scope_text, "", 1)
    if commodity and commodity != "宏观" and text.startswith(commodity) and len(text) > len(commodity) + 1:
        text = text[len(commodity):]
    text = re.sub(r"^(的|之)", "", text)
    text = _alias_indicator(text, scope)
    return text or compact(indicator)


EXACT_INDICATORS = {
    "铁水日均产量": "日均铁水产量",
    "铁水日均": "日均铁水产量",
    "日均铁水": "日均铁水产量",
    "铁水产量": "日均铁水产量",
    "港存": "港口库存",
    "港口库存总量": "港口库存",
    "远期现货价格指数": "远期价格指数",
    "远期现货指数": "远期价格指数",
    "远期指数": "远期价格指数",
    "钢厂盈利率": "盈利率",
    "高炉炼铁产能利用率": "高炉产能利用率",
    "日均疏港量": "疏港量",
    "矿价": "价格",
}


def _alias_indicator(indicator, scope):
    """One indicator for the wordings reports use interchangeably. 45港 and 47港 are not touched here."""
    text = EXACT_INDICATORS.get(indicator, indicator)
    if text in ("库存", "库存量", "库存总量") and "港" in (scope or ""):
        return "港口库存"
    return text


def split_grade(commodity, scope):
    """「47港/澳矿」: the grade named in the scope is the commodity, the rest is the scope."""
    parts = [part for part in re.split(r"[+＋\-/、]", scope) if part]
    grades = [part for part in parts if part != commodity and (
        category_of(part) == category_of(commodity) != "" or
        commodity == "铁矿石" and re.fullmatch(r"\D{1,4}(矿|粉|块)", part))]
    if len(grades) == 1 and len(parts) == 2:
        return grades[0], next(part for part in parts if part != grades[0])
    return commodity, scope


def raw_key(event):
    scope = tidy_scope(event["scope"])
    commodity, scope = split_grade(event["commodity"], scope)
    commodity = tidy_commodity(commodity, event["indicator"], scope)
    indicator = tidy_indicator(commodity, event["indicator"], scope)
    indicator, scope = tidy_mill(commodity, indicator, scope)
    return "|".join((commodity, indicator, scope, event["period"]))


def family_of(commodity):
    if commodity == "宏观":
        return "宏观"
    return category_of(commodity) or commodity or "其他"


def _fallback(raw, dim):
    commodity, indicator, scope, period = raw.split("|")
    return {"key": raw, "commodity": commodity, "indicator": indicator, "scope": scope,
            "period": period, "dim": dim, "parent": "", "source": "原始"}


def _accept(raw, row, dim):
    commodity, indicator, scope, period = raw.split("|")
    key = str(row.get("key") or "").strip()
    parts = key.split("|")
    if len(parts) != 4:
        return None
    new_commodity, new_indicator, new_scope, new_period = (compact(part) for part in parts)
    if not new_indicator or new_commodity != commodity or new_period != period or new_period not in PERIODS:
        return None
    if numbers(new_indicator) or re.search(r"上升|下降|增加|减少|回落|上涨|下跌|去库|累库", new_indicator):
        return None
    new_scope = tidy_scope(new_scope)
    if bool(scope) != bool(new_scope) or numbers(scope) != numbers(new_scope):
        return None
    same_place = scope == new_scope or scope in new_scope or new_scope in scope
    if scope and not same_place and not numbers(scope):
        return None
    parent = str(row.get("parent") or "").strip()
    if parent and (len(parent.split("|")) != 4 or parent == key):
        parent = ""
    return {"key": "|".join((new_commodity, new_indicator, new_scope, new_period)),
            "commodity": new_commodity, "indicator": new_indicator, "scope": new_scope,
            "period": new_period, "dim": dim, "parent": parent, "source": "归一"}


def induce(chat, events, workers):
    counts = Counter()
    dims, examples = {}, {}
    for event in events:
        raw = raw_key(event)
        counts[raw] += 1
        dims.setdefault(raw, Counter())[event["dim"]] += 1
        examples.setdefault(raw, event["name"])
    families = {}
    for raw in counts:
        families.setdefault(family_of(raw.split("|")[0]), []).append(raw)

    def dim_of(raw):
        return dims[raw].most_common(1)[0][0]

    def run(job):
        family, start, batch, existing = job
        ids = {"r%d" % (start + index): raw for index, raw in enumerate(batch, 1)}
        payload = json.dumps({
            "family": family,
            "existing": existing,
            "items": [{"id": rid, "commodity": raw.split("|")[0], "indicator": raw.split("|")[1],
                       "scope": raw.split("|")[2], "period": raw.split("|")[3],
                       "dim": dim_of(raw), "example": examples[raw]}
                      for rid, raw in ids.items()],
        }, ensure_ascii=False)
        parsed = chat.complete_json(ONTOLOGY_SYSTEM, payload, max_tokens=6144,
                                    tag="ontology:%s:%d" % (family, start))
        if isinstance(parsed, dict):
            parsed = parsed.get("items") or parsed.get("data") or []
        found = {}
        for row in parsed if isinstance(parsed, list) else []:
            if not isinstance(row, dict):
                continue
            raw = ids.get(str(row.get("id") or "").strip())
            if raw and raw not in found:
                accepted = _accept(raw, row, dim_of(raw))
                if accepted:
                    found[raw] = accepted
        for raw in batch:
            found.setdefault(raw, _fallback(raw, dim_of(raw)))
        return found

    # The most frequent forms of every family go first and become the anchors;
    # the remaining batches of all families then run in parallel against them.
    ordered = {family: sorted(raws, key=lambda raw: (-counts[raw], raw)) for family, raws in families.items()}
    mapping = {}
    first = [(family, 0, raws[:BATCH], []) for family, raws in sorted(ordered.items())]
    for part in parallel_map(run, first, workers, "指标归一（锚点）"):
        mapping.update(part)
    anchors = {}
    for family, raws in ordered.items():
        anchors[family] = sorted({mapping[raw]["key"] for raw in raws[:BATCH]})
    rest = [(family, start, raws[start:start + BATCH], anchors[family])
            for family, raws in sorted(ordered.items()) for start in range(BATCH, len(raws), BATCH)]
    for part in parallel_map(run, rest, workers, "指标归一（其余）"):
        mapping.update(part)
    variables = {}
    for raw, variable in mapping.items():
        current = variables.setdefault(variable["key"], dict(variable, raw_forms=[], dims=Counter()))
        current["raw_forms"].append(raw)
        current["dims"][variable["dim"]] += counts[raw]
        if variable["parent"] and not current["parent"]:
            current["parent"] = variable["parent"]
    for variable in variables.values():
        variable["dim"] = variable["dims"].most_common(1)[0][0]
        variable["dims"] = dict(variable["dims"])
        if variable["parent"] not in variables:
            variable["parent"] = ""
    stats = {"raw_variables": len(counts), "variables": len(variables),
             "normalized": sum(1 for item in mapping.values() if item["source"] == "归一")}
    return mapping, variables, stats


def assign(events, mapping):
    for event in events:
        variable = mapping[raw_key(event)]
        event["var"] = variable["key"]
        event["abstract"] = abstract_key(variable["key"], event["dir"])


def abstract_key(var, direction):
    return var + "|" + direction


# Role names differ by event kind. An entity keeps one identity; the role is only how it participates.
ROLE_SCHEMAS = {
    "MAINTENANCE": ("agent", "facility", "product", "location", "start_time",
                    "expected_recovery_time", "actual_recovery_time", "status"),
    "POLICY_CHANGE": ("publisher", "policy_measure", "affected_region", "affected_entity",
                      "effective_time", "modality"),
    "LOGISTICS_DISRUPTION": ("transport_route", "transport_mode", "disruption_cause", "affected_cargo",
                             "origin_region", "destination_region", "start_time", "end_time"),
    "LOGISTICS_CHANGE": ("origin_region", "destination_region", "start_time", "end_time"),
    "PRODUCTION_CHANGE": ("product", "location", "start_time"),
}
MAINTENANCE_STATUS = ("计划", "进行中", "已完成")
POLICY_MODALITY = ("已发布", "拟议", "预测")


def role_binding(entity_name, role_name):
    """The entity id is the name's identity. The role is not a second entity."""
    name = (entity_name or "").strip()
    return {"entity_id": "ent_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12],
            "name": name, "role": role_name}
