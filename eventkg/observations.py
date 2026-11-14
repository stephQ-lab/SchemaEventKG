# -*- coding: utf-8 -*-
"""Structured observations parsed from table headers, not from co-occurring numbers."""
import re
from datetime import date

CELL = re.compile(r"列(\d+)\[([^\]]*)\]=([^；\n]*)")
ROW = re.compile(r"原表第(\d+)行：(.+)")
DATE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
LOOSE = re.compile(r"[+\-−－]?\d+(?:\.\d+)?%?")


VALUE_CELL = re.compile(
    r"^([+\-−－])?(\d+(?:\.\d+)?)(%|％|万吨|吨|元/吨|元|美元/干吨|点)?$")


def parse_signed_number(raw):
    """A value cell is only a number. A name such as 47港库存 or 62%澳粉 is not a value."""
    text = (raw or "").strip()
    if not text or "空白" in text:
        return None
    cleaned = text.replace(",", "").replace("，", "").replace(" ", "")
    match = VALUE_CELL.fullmatch(cleaned)
    if not match:
        return None
    value = float(match.group(2))
    if match.group(1) in ("-", "−", "－"):
        value = -value
    unit = match.group(3) or ""
    if unit == "％":
        unit = "%"
    return {"value_raw": text, "value_numeric": value, "unit": unit}


def _header_kind(header):
    text = header or ""
    if "表头未确认" in text or not text.strip():
        return "ambiguous"
    if "预测" in text:
        return "forecast"
    if "同比" in text or "去年同期" in text:
        return "yoy"
    if "较年初" in text or "年初" in text:
        return "ytd"
    if "环比" in text:
        return "mom"
    if "涨跌" in text or "变化" in text:
        return "change"
    if DATE.search(text):
        return "level"
    return "unknown"


def _period(header):
    match = DATE.search(header or "")
    if not match:
        return ""
    return "%s-%02d-%02d" % (match.group(1), int(match.group(2)), int(match.group(3)))


def _direction(value):
    if value is None:
        return ""
    if value < 0:
        return "下降"
    if value > 0:
        return "上升"
    return "持平"


def _measure_unit(indicator):
    """The unit in a trailing parenthesis. A grade such as 62% inside the name is not a unit."""
    match = re.search(r"（([^）]{1,20})）\s*$", indicator or "")
    if not match:
        return ""
    unit = match.group(1)
    if unit in ("%", "％", "百分比"):
        return "%"
    return unit


def _close(left, right):
    return abs(left - right) <= max(0.05, abs(right) * 0.002)


def _row_indicator(cells):
    """The indicator is the label column, decided by the header and the cell position."""
    for cell in cells:
        if any(word in (cell["header"] or "") for word in ("指标", "项目", "名称")):
            if cell["raw"] and "空白" not in cell["raw"]:
                return cell["raw"], cell["column"]
    if not cells:
        return "", None
    first = cells[0]
    if (parse_signed_number(first["raw"]) is None and first["raw"] and "空白" not in first["raw"]
            and _header_kind(first["header"]) in ("unknown", "ambiguous")):
        return first["raw"], first["column"]
    return "", None


def _pair_change(levels, change_value, change_raw, header_kind):
    """Pick current and base by the arithmetic, not by which date column is last.

    108.10 - 106.35 = 1.75 selects 4月17日 as the current period even when that
    column sits on the left. A percent-level gap such as 22.29 - 40.22 = -17.93
    is a percentage-point change, distinct from the relative rate -44.58%.
    """
    candidates = []
    for current in levels:
        for base in levels:
            if current["observation_period"] == base["observation_period"]:
                continue
            if current["value_numeric"] is None or base["value_numeric"] is None:
                continue
            diff = current["value_numeric"] - base["value_numeric"]
            if _close(diff, change_value):
                candidates.append((current, base, "absolute"))
            if base["value_numeric"]:
                relative = diff / base["value_numeric"] * 100
                if _close(relative, change_value):
                    candidates.append((current, base, "relative"))
    if not candidates:
        return None
    if header_kind == "change":
        preferred = [item for item in candidates if item[2] == "absolute"]
    elif header_kind in ("mom", "yoy") and "%" in (change_raw or ""):
        preferred = [item for item in candidates if item[2] == "relative"]
    else:
        preferred = candidates
    pool = preferred or candidates
    orientations = {(item[0]["observation_period"], item[1]["observation_period"]) for item in pool}
    if len(orientations) != 1:
        return None
    return pool[0]


def _basis(kind):
    return {"yoy": "同比", "mom": "环比", "change": "涨跌", "ytd": "较年初", "forecast": "预测"}.get(kind, "")


def _span_ok(kind, current, base):
    """同比 needs a year-ago column. 环比 needs the adjacent period, not a matching subtraction alone."""
    if kind not in ("yoy", "mom") or not current or not base:
        return True
    try:
        gap = abs((date.fromisoformat(current) - date.fromisoformat(base)).days)
    except ValueError:
        return False
    if kind == "yoy":
        return 300 <= gap <= 400
    return 1 <= gap <= 45


def _blank():
    return {
        "observation_id": "", "document_id": "", "indicator_id": "", "commodity_id": "",
        "subject_id": "", "scope": "", "region": "", "observation_period": "",
        "comparison_period": "", "value_numeric": None, "value_raw": "", "unit": "",
        "change_numeric": None, "change_unit": "", "direction": "", "base_period": "",
        "source_table_id": "", "source_page": None, "source_section": "",
        "row_header": "", "column_header": "", "source_text": "", "evidence_id": "",
        "status": "confirmed", "measure_kind": "",
    }


def parse_table_measures(text, document_id="", source_table_id="", source_page=None,
                         source_section="", scope="", region="", commodity_id="", subject_id=""):
    """Pair a cell with the indicator and period named by its own row and column headers.

    Several numbers in one table, or in one unlabeled string, do not become a
    current value, a previous value and a change just because they co-occur.
    """
    text = text or ""
    found = []
    title = re.search(r"\[表格：([^\]]+)\]", text)
    section = source_section or (title.group(1) if title else "")
    for row_match in ROW.finditer(text):
        cells = [{"column": int(item.group(1)), "header": item.group(2), "raw": item.group(3).strip()}
                 for item in CELL.finditer(row_match.group(2))]
        cells.sort(key=lambda item: item["column"])
        indicator, indicator_column = _row_indicator(cells)
        named_unit = _measure_unit(indicator)
        levels = []
        changes = []
        for cell in cells:
            if cell["column"] == indicator_column:
                continue
            kind = _header_kind(cell["header"])
            number = parse_signed_number(cell["raw"])
            if number is None:
                continue
            item = _blank()
            item.update({
                "observation_id": "%s:r%s:c%s" % (source_table_id or document_id or "table",
                                                  row_match.group(1), cell["column"]),
                "document_id": document_id, "indicator_id": indicator, "commodity_id": commodity_id,
                "subject_id": subject_id, "scope": scope, "region": region or scope,
                "source_table_id": source_table_id, "source_page": source_page,
                "source_section": section, "row_header": indicator, "column_header": cell["header"],
                "source_text": row_match.group(0), "evidence_id": source_table_id,
                "unit": number["unit"] or named_unit,
                "measure_kind": kind, "value_raw": number["value_raw"],
                "value_numeric": number["value_numeric"],
            })
            if not indicator:
                item["status"] = "AMBIGUOUS"
            if kind == "level":
                item["observation_period"] = _period(cell["header"]) or ""
                item["status"] = "confirmed" if indicator and item["observation_period"] else "AMBIGUOUS"
                item["record_kind"] = "TrendSignal"
                item["level_direction"] = "UNKNOWN"
                item["comparison_basis"] = ""
                item["direction"] = ""
                if item["status"] == "confirmed":
                    levels.append(item)
            elif kind in ("mom", "yoy", "change", "forecast"):
                changes.append((item, kind, number))
            else:
                item["status"] = "AMBIGUOUS"
                item["measure_kind"] = "ambiguous"
            found.append(item)
        percent_levels = bool(levels) and all((level.get("unit") or named_unit) == "%" for level in levels)
        for item, kind, number in changes:
            paired = _pair_change(levels, number["value_numeric"], number["value_raw"], kind) if indicator else None
            item["change_numeric"] = number["value_numeric"]
            item["record_kind"] = "TrendSignal"
            item["comparison_basis"] = _basis(kind)
            item["direction"] = ""
            item["level_direction"] = "UNKNOWN"
            item["momentum"] = ""
            if not paired:
                item["status"] = "AMBIGUOUS"
                item["change_unit"] = ""
                continue
            current, base, how = paired
            if not _span_ok(kind, current["observation_period"], base["observation_period"]):
                item["status"] = "UNKNOWN"
                item["change_unit"] = ""
                continue
            item["observation_period"] = current["observation_period"]
            item["comparison_period"] = base["observation_period"]
            item["base_period"] = base["observation_period"]
            item["status"] = "confirmed"
            item["direction"] = _direction(number["value_numeric"])
            item["level_direction"] = item["direction"]
            if how == "relative":
                item["change_unit"] = "%"
                item["measure_kind"] = "forecast" if kind == "forecast" else kind
            else:
                item["change_unit"] = "百分点" if percent_levels else (current.get("unit") or number["unit"] or "")
                item["measure_kind"] = "forecast" if kind == "forecast" else (
                    "percentage_point" if item["change_unit"] == "百分点" else kind)
    if found:
        return found
    loose = [token for token in LOOSE.findall(text) if parse_signed_number(token)]
    if len(loose) >= 2 and not ROW.search(text):
        item = _blank()
        item.update({
            "observation_id": "%s:ambiguous" % (source_table_id or document_id or "table"),
            "document_id": document_id, "source_table_id": source_table_id, "source_page": source_page,
            "source_section": section, "source_text": text, "evidence_id": source_table_id,
            "value_raw": text.strip(), "status": "AMBIGUOUS", "measure_kind": "ambiguous",
        })
        return [item]
    return []
