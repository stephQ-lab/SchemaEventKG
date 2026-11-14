# -*- coding: utf-8 -*-
"""Parsed reports -> cleaned narrative windows and table observations.

Noise that made earlier extractions messy is removed before any model call:
table-of-contents pages, chart axis ticks, data-source notes, repeated running
heads, disclaimer lines, and space characters left inside words by PDF line
wrapping. Table rows stay observations; they are never sent as narrative.
"""
import hashlib
import json
import re
from pathlib import Path

from atlas_rag.document_input.structure import contents_pages
from eventkg.lexicon import NOISE_LINE, compact, document_categories
from eventkg.observations import parse_table_measures

SKIP_KINDS = {"picture", "page_header", "page_footer", "table"}
BOILERPLATE_SECTIONS = ("版权声明", "免责声明", "目录", "CONTENTS")
WINDOW_CHARS = 2200
DOCUMENT_CHARS = 14000
SUMMARY_WORDS = ("综述", "回顾", "展望", "观点", "核心", "总结", "小结", "摘要", "概览", "解读", "后市")

_CJK = r"\u4e00-\u9fff"
_JOIN_LEFT = re.compile(r"(?<=[%s，。；：、（）《》“”%%])\s+(?=[%s0-9，。；：、（）《》“”])" % (_CJK, _CJK))
_JOIN_DIGIT = re.compile(r"(?<=[0-9%%])\s+(?=[%s])" % _CJK)


def normalize_line(line):
    line = (line or "").replace("\u3000", " ").replace("\xa0", " ").strip()
    line = _JOIN_LEFT.sub("", line)
    line = _JOIN_DIGIT.sub("", line)
    return re.sub(r"[ \t]{2,}", " ", line)


def _cjk_count(text):
    return len(re.findall("[%s]" % _CJK, text))


CHANGE_WORDS = re.compile(r"上涨|下跌|上升|下降|增加|减少|回升|回落|去库|累库|走扩|走缩|收窄|扩大|走强|走弱|"
                          r"持平|企稳|触顶|见底|检修|复产|停产|减产|增产|偏强|偏弱|震荡|涨跌|高于|低于")


def is_noise(line):
    if NOISE_LINE.search(line) or "=" in line:
        return True
    if _cjk_count(line) < 4:
        return True
    if re.search(r"[。；，,]", line) and len(compact(line)) < 10:
        return True
    if not re.search(r"[。；，,]", line):
        if len(line) > 40:
            return True
        digits = len(re.findall(r"[0-9.\-+%/↑↓]", line))
        if digits / max(1, len(line)) > 0.35:
            return True
    return False


def is_heading(line):
    return len(line) <= 40 and not re.search(r"[。；，,]", line)


def keep_heading(line):
    """Chart and table labels are dropped; a heading that states a change is kept for context."""
    return bool(CHANGE_WORDS.search(line))


def report_date(*sources):
    for source in sources:
        match = re.search(r"(20\d{2})[-_.年]?(\d{2})[-_.月]?(\d{2})", source or "")
        if match:
            year, month, day = match.groups()
            if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                return "%s-%s-%s" % (year, month, day)
    return ""


def load_rows(data_dir, name):
    data_dir = Path(data_dir)
    source = data_dir / (name + ".json")
    if not source.is_file():
        candidates = sorted(path for path in data_dir.glob("*.json")
                            if path.name not in {"document_registry.json", "preparation_report.json",
                                                 "preparation_status.json"})
        if not candidates:
            raise ValueError("输入目录里没有文档清单 JSON：" + str(data_dir))
        source = candidates[0]
    return json.loads(source.read_text(encoding="utf-8")), source


def load_corpus(data_dir, name="unified10"):
    data_dir = Path(data_dir)
    rows, _ = load_rows(data_dir, name)
    documents = []
    for row in rows:
        relative = row.get("metadata", {}).get("structure_file")
        if not relative:
            raise ValueError("缺少结构文件：" + row.get("id", ""))
        path = (data_dir / relative).resolve()
        if not str(path).startswith(str(data_dir.resolve())):
            raise ValueError("结构文件必须在输入目录内")
        structure = json.loads(path.read_text(encoding="utf-8"))
        if structure.get("id") != row["id"] or structure.get("text") != row.get("text"):
            raise ValueError("结构文件与输入记录不一致：" + row["id"])
        documents.append(prepare_document(row, structure))
    documents.sort(key=lambda doc: (doc["report_date"] or "9999", doc["id"]))
    return documents


def prepare_document(row, structure):
    meta = row.get("metadata", {})
    title = structure.get("title") or meta.get("file") or row["id"]
    file_name = meta.get("file") or ""
    categories = document_categories(doc_id=row["id"], title=title, file_name=file_name,
                                     text=structure.get("text") or "")
    groups = []
    observations = []
    hidden = contents_pages(structure.get("units") or [])
    for unit in structure.get("units") or []:
        kind = unit.get("kind") or ""
        if unit.get("page") in hidden and kind != "table":
            continue
        section = normalize_line((unit.get("section") or "").replace("\n", " "))
        if kind == "table":
            text = (unit.get("text") or "").strip()
            if text:
                item = {
                    "id": "%s:obs:%d" % (row["id"], len(observations) + 1),
                    "doc_id": row["id"], "section": section, "page": unit.get("page"), "text": text,
                }
                item["measures"] = parse_table_measures(
                    text, document_id=row["id"], source_table_id=item["id"],
                    source_page=unit.get("page"), source_section=section)
                observations.append(item)
            continue
        if kind in SKIP_KINDS:
            continue
        if not groups or groups[-1]["section"] != section:
            groups.append({"section": section, "units": []})
        groups[-1]["units"].append(unit)

    seen = set()
    sections = []
    for group in groups:
        if any(mark in group["section"] for mark in BOILERPLATE_SECTIONS):
            continue
        lines = []
        body = 0
        title_key = compact(group["section"])
        for unit in group["units"]:
            for raw in (unit.get("text") or "").split("\n"):
                line = normalize_line(raw)
                if not line or is_noise(line):
                    continue
                key = compact(line)
                if key in seen or key == title_key:
                    continue
                seen.add(key)
                heading = is_heading(line)
                if heading and not keep_heading(line):
                    continue
                lines.append(("### " + line) if heading else line)
                if not heading:
                    body += len(key)
        if body < 30:
            continue
        pages = sorted({unit.get("page") for unit in group["units"] if unit.get("page") is not None})
        sections.append({"section": group["section"], "lines": lines, "pages": pages})

    windows = build_windows(row["id"], sections)
    narrative = "\n\n".join(window["text"] for window in windows)
    return {
        "id": row["id"],
        "title": title,
        "file_name": file_name,
        "source_file": meta.get("source_file") or "",
        "report_date": report_date(file_name, title),
        "categories": categories,
        "sections": sections,
        "windows": windows,
        "narrative": narrative,
        "observations": observations,
    }


def _section_block(section):
    return "【章节】%s\n%s" % (section["section"] or "（无标题）", "\n".join(section["lines"]))


def build_windows(doc_id, sections):
    windows = []
    current = []
    size = 0

    def flush():
        nonlocal current, size
        if current:
            text = "\n\n".join(block for _, block in current)
            windows.append({
                "id": "%s:w%d" % (doc_id, len(windows) + 1),
                "doc_id": doc_id,
                "sections": [name for name, _ in current],
                "text": text,
                "sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
        current, size = [], 0

    for section in sections:
        block = _section_block(section)
        if len(block) > WINDOW_CHARS:
            flush()
            for piece in _split_block(section):
                current.append((section["section"], piece))
                size = len(piece)
                flush()
            continue
        if size and size + len(block) > WINDOW_CHARS:
            flush()
        current.append((section["section"], block))
        size += len(block) + 2
    flush()
    return windows


def _split_block(section):
    head = "【章节】%s（续）\n" % (section["section"] or "（无标题）")
    pieces = []
    buffer = "【章节】%s\n" % (section["section"] or "（无标题）")
    for line in section["lines"]:
        sentences = re.split(r"(?<=[。；！？])", line) if len(line) > WINDOW_CHARS else [line]
        for sentence in sentences:
            if len(buffer) + len(sentence) + 1 > WINDOW_CHARS and len(buffer) > len(head):
                pieces.append(buffer.rstrip())
                buffer = head
            buffer += sentence + ("\n" if sentence is sentences[-1] else "")
    if buffer.strip() and len(buffer) > len(head):
        pieces.append(buffer.rstrip())
    return pieces


def document_view(document, limit=DOCUMENT_CHARS):
    """Full cleaned narrative when it fits; otherwise summary sections first, others shortened."""
    text = document["narrative"]
    if len(text) <= limit:
        return text
    blocks = [_section_block(section) for section in document["sections"]]
    summary = [block for block, section in zip(blocks, document["sections"])
               if any(word in section["section"] for word in SUMMARY_WORDS)]
    others = [block for block, section in zip(blocks, document["sections"])
              if not any(word in section["section"] for word in SUMMARY_WORDS)]
    budget = max(0, limit - sum(len(block) for block in summary))
    share = budget // max(1, len(others))
    trimmed = [block[:share] for block in others]
    return "\n\n".join(summary + trimmed)[:limit]


def sentences_of(text):
    """Sentence spans used to repair a quote the model slightly rewrote."""
    found = []
    for line in (text or "").split("\n"):
        line = line.strip()
        if line.startswith("### ") or line.startswith("【章节】"):
            continue
        for piece in re.split(r"(?<=[。；！？])", line):
            piece = piece.strip()
            if len(compact(piece)) >= 6:
                found.append(piece)
    return found
