import re

import pdfplumber

from atlas_rag.document_input.structure import finalize_document


def intersection(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def top_bbox(block, page_height):
    bbox = block.get("bbox")
    if not isinstance(bbox, dict):
        return None
    if bbox.get("coord_origin") == "BOTTOMLEFT":
        return (bbox["l"], page_height - bbox["t"], bbox["r"], page_height - bbox["b"])
    return (bbox["l"], bbox["t"], bbox["r"], bbox["b"])


def extract_cells(page, table, words):
    rows = []
    columns = [column.bbox for column in table.columns]
    for row in table.rows:
        values = []
        for column_index, cell in enumerate(row.cells):
            if cell is None:
                column = columns[column_index]
                center = (column[0] + column[2]) / 2
                merged = any(existing is not None and existing[0] <= center <= existing[2] for existing in row.cells)
                if merged:
                    values.append("")
                    continue
                cell = (column[0], row.bbox[1], column[2], row.bbox[3])
            left, top, right, bottom = cell
            left, top = max(page.bbox[0], left), max(page.bbox[1], top)
            right, bottom = min(page.bbox[2], right), min(page.bbox[3], bottom)
            if right - left < 0.5 or bottom - top < 0.5:
                values.append("")
                continue
            if abs(right - table.bbox[2]) < 1:
                # 仅扩展跨越表格外边界的完整文字，内部列边界保持原值。
                crossing = [word["x1"] for word in words if left <= word["x0"] < right < word["x1"]
                            and top <= (word["top"] + word["bottom"]) / 2 < bottom]
                right = min(page.width, max([right, *crossing]))
            values.append((page.crop((left, top, right, bottom)).extract_text(x_tolerance=1, y_tolerance=3) or "").strip())
        rows.append(values)
    return rows


def is_header(row):
    filled = [value for value in row if value]
    if len(filled) < 2:
        return False
    first = "".join(row[0].split())
    labels = {"", "标签", "指标", "项目", "地区", "市场", "品种", "矿种", "规格", "名称", "日期", "月份", "年份", "类别", "类型", "分类", "企业", "企业名称", "工厂名称"}
    if (first in labels or "更新" in first) and sum(bool(re.search(r"20\d{2}[/年.-]\d{1,2}", value)) for value in row) >= 2:
        return True
    return first in labels and first != "" and (any(not re.fullmatch(r"[-+\d.,%]+", value) for value in filled[1:])
                                               or all(re.fullmatch(r"(?:19|20)\d{2}", value) for value in filled[1:]))


def divide_table(original, rows):
    parts = []
    caption = original.get("caption", "")
    header = None
    active = []
    source_rows = []

    def flush():
        nonlocal active, source_rows
        if not active:
            return
        payload = dict(original)
        payload.update(id=original["id"] + ":native%d" % (len(parts) + 1), rows=([header] if header else []) + active,
                       header_rows=int(header is not None), caption=caption,
                       source_rows=([0] if header else []) + source_rows,
                       table_method="pdfplumber原生单元格与Docling版面共同定位")
        parts.append(payload)
        active, source_rows = [], []

    for index, row in enumerate(rows):
        filled = [value for value in row if value]
        if not filled:
            continue
        native_cells = original.get("native_cells", [])[index]
        table_width = original["native_bbox"][2] - original["native_bbox"][0]
        spanning = any(cell is not None and cell[2] - cell[0] >= table_width * 0.8 for cell in native_cells)
        if len(filled) == 1 and spanning:
            flush()
            text = filled[0]
            caption = text
            payload = {key: value for key, value in original.items() if key not in ("rows", "cells", "header_rows")}
            payload.update(id=original["id"] + ":native%d" % (len(parts) + 1), kind="text", text=text,
                           table_method="pdfplumber原生单元格", source_row=index + 1)
            parts.append(payload)
        elif is_header(row):
            flush()
            header = row
        else:
            active.append(row)
            source_rows.append(index + 1)
    flush()
    return parts


def refine_pdf_tables(document):
    if not document["source_file"].lower().endswith(".pdf"):
        return document
    replacements = {}
    removed = set()
    with pdfplumber.open(document["source_file"]) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            blocks = [block for block in document["blocks"] if block.get("page") == number]
            modeled = [block for block in blocks if block["kind"] == "table"]
            if not modeled:
                continue
            words = page.extract_words()
            for native in page.find_tables():
                native_box = native.bbox
                candidates = []
                for block in modeled:
                    box = top_bbox(block, page.height)
                    if box is None:
                        continue
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > 0 and intersection(box, native_box) / area >= 0.8:
                        candidates.append(block)
                if not candidates or (len(candidates) > 1 and any(block["rows"] for block in candidates)):
                    continue
                block = candidates[0]
                rows = extract_cells(page, native, words)
                old_values = set(re.findall(r"-?\d+(?:\.\d+)?%?", str(block["rows"])))
                new_values = set(re.findall(r"-?\d+(?:\.\d+)?%?", str(rows)))
                coverage = len(old_values & new_values) / len(old_values) if old_values else 1.0
                native_columns = sum(any(column < len(row) and row[column] for row in rows)
                                     for column in range(max((len(row) for row in rows), default=0)))
                model_columns = sum(any(column < len(row) and row[column] for row in block["rows"])
                                    for column in range(max((len(row) for row in block["rows"]), default=0)))
                extra_columns = native_columns > model_columns
                if (native_columns < max(2, model_columns) or coverage < 0.95
                        or not (extra_columns or len(new_values) > len(old_values))):
                    continue
                for other in candidates[1:]:
                    removed.add(other["id"])
                region = dict(block)
                region.pop("cells", None)
                region["native_cells"] = [row.cells for row in native.rows]
                region["native_bbox"] = native_box
                replacements[block["id"]] = divide_table(region, rows)
                all_text = "".join("".join(value.split()) for row in rows for value in row)
                expanded_right = max([native_box[2], *(word["x1"] for word in words
                                      if native_box[0] <= word["x0"] < native_box[2] < word["x1"]
                                      and native_box[1] <= word["top"] <= native_box[3])])
                expanded_box = (native_box[0], native_box[1], expanded_right, native_box[3])
                for other in blocks:
                    if other["kind"] in ("table", "picture"):
                        continue
                    text = "".join(other.get("text", "").split())
                    box = top_bbox(other, page.height)
                    if text and box and text in all_text:
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > 0 and intersection(box, expanded_box) / area > 0.7:
                            removed.add(other["id"])
                document["issues"].append({"kind": "native_table_reconciliation", "page": number,
                                           "block_id": block["id"], "numeric_agreement": coverage,
                                           "reason": "表格按原生单元格分配，保留边界处的完整数字和表内分节标题"})
    blocks = []
    for block in document["blocks"]:
        if block["id"] in replacements:
            blocks.extend(replacements[block["id"]])
        elif block["id"] not in removed:
            blocks.append(block)
    with pdfplumber.open(document["source_file"]) as pdf:
        for block in blocks:
            if block["kind"] != "table" or block.get("rows"):
                continue
            page = pdf.pages[block["page"] - 1]
            box = top_bbox(block, page.height)
            clipped = (max(0, box[0]), max(0, box[1]), min(page.width, box[2]), min(page.height, box[3]))
            block["kind"] = "unstructured_table"
            block["text"] = "[表格行列尚未确认，数值不得组合为事件]\n" + (page.crop(clipped).extract_text(layout=True) or "")
            document["issues"].append({"kind": "table_structure_requires_review", "page": block["page"],
                                       "block_id": block["id"], "bbox": box,
                                       "reason": "保留区域文字和位置，行列尚未确认"})
    document["blocks"] = blocks
    document["table_refinement_version"] = 1
    return finalize_document(document, allow_empty=True)


def attach_table_notes(document):
    blocks = document["blocks"]
    for index, block in enumerate(blocks):
        if block["kind"] != "table":
            continue
        notes = []
        for following in blocks[index + 1:index + 4]:
            if following["kind"] in ("table", "section_header"):
                break
            text = following.get("text", "").strip()
            if text.startswith(("备注", "单位", "注：", "注:", "数据说明", "数据来源", "统计口径")):
                if following.get("page") == block.get("page"):
                    notes.append(text)
        block["notes"] = notes
    return finalize_document(document, allow_empty=True)
