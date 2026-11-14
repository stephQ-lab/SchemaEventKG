import hashlib
import json
import re
from pathlib import Path


SCHEMA_VERSION = "document-blocks-1"


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def table_units(block):
    rows = block["rows"]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    header_count = block.get("header_rows", 0)
    headers = []
    for col in range(width):
        parts = []
        for row in rows[:header_count]:
            value = row[col] if col < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts) or "列%d（表头未确认）" % (col + 1))
    prefix = block.get("caption", "")
    header = "[表格：%s]" % (prefix or "原文未提供标题")
    if block.get("notes"):
        header += "\n原文附注：" + "；".join(block["notes"])
    result = []
    for row_index in range(header_count, len(rows)):
        row = rows[row_index]
        if not any(row):
            continue
        fields = []
        for col in range(width):
            value = row[col] if col < len(row) else ""
            fields.append("列%d[%s]=%s" % (col + 1, headers[col], value or "（空白）"))
        source_row = block.get("source_rows", list(range(1, len(rows) + 1)))[row_index]
        text = "原表第%d行：%s" % (source_row, "；".join(fields))
        result.append({"text": header + "\n" + text, "block_id": block["id"],
                       "row": source_row, "kind": "table", "page": block.get("page"),
                       "bbox": block.get("bbox"), "coordinate_system": block.get("coordinate_system"),
                       "section": block.get("section", "")})
    return result


def finalize_document(document, allow_empty=False):
    texts = [document["title"]]
    cursor = len(texts[0])
    units = []
    for block in document["blocks"]:
        if block["kind"] in ("page_header", "page_footer", "picture"):
            continue
        if block["kind"] == "table":
            parts = table_units(block)
        else:
            parts = [{"text": block.get("text", ""), "block_id": block["id"],
                      "kind": block["kind"], "page": block.get("page"),
                      "bbox": block.get("bbox"), "coordinate_system": block.get("coordinate_system"),
                      "section": block.get("section", "")}]
        for unit in parts:
            unit["text"] = unit["text"].replace("\v", "\n").replace("\r\n", "\n").replace("\r", "\n")
            unit["section"] = unit.get("section", "").replace("\v", "\n")
            if not unit["text"].strip():
                continue
            texts.append(unit["text"])
            unit["start"] = cursor + 2
            unit["end"] = unit["start"] + len(unit["text"])
            cursor = unit["end"]
            units.append(unit)
    document["text"] = "\n\n".join(texts)
    document["units"] = units
    document["text_sha256"] = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
    document["schema_version"] = SCHEMA_VERSION
    if not units and not allow_empty:
        raise ValueError("文档没有可抽取内容：" + document["title"])
    for unit in units:
        if document["text"][unit["start"]:unit["end"]] != unit["text"]:
            raise ValueError("文档字符位置验证失败：" + document["id"])
    return document


def split_unit(unit, limit):
    text = unit["text"]
    if len(text) <= limit:
        return [dict(unit)]
    if unit["kind"] == "table":
        raise ValueError("完整表格行超过片段限制，请增加 --chunk-size：" + unit["block_id"])
    result = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"[。！？!?\n]", text[start:end]))
            if boundaries and boundaries[-1].end() >= limit // 2:
                end = start + boundaries[-1].end()
        piece = dict(unit)
        piece.update(text=text[start:end], start=unit["start"] + start, end=unit["start"] + end)
        result.append(piece)
        start = end
    return result


_CONTENTS = re.compile(r"目录|CONTENTS", re.I)


def contents_pages(units):
    """Pages that are only a table of contents: a 目录 heading and no prose sentence."""
    grouped = {}
    for unit in units or []:
        page = unit.get("page")
        if page is None or unit.get("kind") in ("picture", "page_header", "page_footer", "table"):
            continue
        grouped.setdefault(page, []).append(unit.get("text") or "")
    found = set()
    for page, texts in grouped.items():
        blob = "\n".join(texts)
        if _CONTENTS.search(blob) and not re.search(r"[。！？]", blob):
            found.add(page)
    return found


def make_chunks(document, limit=5000, merge_sections=True):
    """Narrative units of neighbouring pages and sections share a chunk up to limit.

    Splitting at every page or section produced hundreds of tiny chunks (one
    model call each). Tables still get chunks of their own.
    """
    if limit < 512:
        raise ValueError("chunk_size 至少为512字符")
    chunks = []
    current = []
    size = 0

    def flush():
        nonlocal current, size
        if current:
            chunks.append({"text": "\n\n".join(unit["text"] for unit in current), "units": current})
        current, size = [], 0

    hidden = contents_pages(document["units"])
    for original in document["units"]:
        if original.get("page") in hidden and original.get("kind") != "table":
            continue
        for unit in split_unit(original, limit):
            if current:
                previous = current[-1]
                separate = (not merge_sections and
                            (unit.get("page"), unit.get("section")) != (previous.get("page"), previous.get("section")))
                separate = separate or (unit["kind"] == "table") != (previous["kind"] == "table")
                separate = separate or (unit["kind"] == "table" and unit["block_id"] != previous["block_id"])
                if separate or size + 2 + len(unit["text"]) > limit:
                    flush()
            current.append(unit)
            size += len(unit["text"]) + (2 if len(current) > 1 else 0)
    flush()
    for index, chunk in enumerate(chunks):
        chunk.update(index=index, document_id=document["id"], text_sha256=document["text_sha256"])
        position = 0
        for unit in chunk["units"]:
            unit["chunk_start"] = position
            unit["chunk_end"] = position + len(unit["text"])
            position = unit["chunk_end"] + 2
        if len(chunk["text"]) > limit:
            raise ValueError("片段超过字符限制")
    return chunks


def document_prompt_context(document):
    from atlas_rag.kg_construction.commodity import document_categories, match_commodities

    return {"categories": document_categories(title=document["title"], text=document["title"]),
            "varieties": match_commodities(document["text"])}


def structured_prompt(document, chunk, total, context=None):
    context = document_prompt_context(document) if context is None else context
    categories, varieties = context["categories"], context["varieties"]
    sections = list(dict.fromkeys(unit.get("section", "") for unit in chunk["units"] if unit.get("section")))
    pages = sorted({unit["page"] for unit in chunk["units"] if unit.get("page") is not None})
    return (
        "[文档标题]\n%s\n[标题品类]\n%s\n[全文品种索引，仅用于理解指代]\n%s\n"
        "[章节]\n%s\n[页面或幻灯片]\n%s\n[片段位置]\n%d/%d\n"
        "只抽取本片段证据支持的事实。标题与品种索引不构成事件证据。"
        "文档编号、块编号、原表行号、列号仅用于定位，不作为实体、事件或概念。"
        "表格每个数值必须同时保留同行指标、该列表头日期、单位和统计范围。"
        "不同日期的数据分别描述；涨跌值与环比只能属于对应本期指标。"
        "空白、#VALUE!、#N/A、#REF!、表头未确认、横轴未确认的数据不得补写。成本、利润分别处理。"
        "图片图表中没有明确读取的数据不得根据趋势推算。\n[原文]\n%s"
    ) % (document["title"], "、".join(categories), "、".join(varieties),
         " / ".join(sections), str(pages), chunk["index"] + 1, total, chunk["text"])


def load_structured_sample(sample, data_directory, limit, merge_sections=True):
    relative = sample.get("metadata", {}).get("structure_file")
    if not relative:
        return None
    root = Path(data_directory).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("structure_file 必须位于输入目录内")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["schema_version"] != SCHEMA_VERSION or document["id"] != sample["id"]:
        raise ValueError("结构文件与输入记录不一致：" + str(path))
    if document["text"] != sample["text"]:
        raise ValueError("结构文件与输入文本不一致：" + str(path))
    return document, make_chunks(document, limit, merge_sections)
