# -*- coding: utf-8 -*-
"""Processing views over a whole document.

A chunk is only the span sent as evidence. The prompt view also carries the
document title, where the span sits, and the commodity names found in the
full document. Those names are a registry for coreference, not facts.
"""
import re

from atlas_rag.kg_construction.commodity import document_categories, match_commodities


def split_paragraphs(text, chunk_size):
    """Split on paragraph boundaries. A short document stays one span."""
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if not paras:
        return []
    chunks = []
    buf = ""
    for para in paras:
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i:i + chunk_size])
            continue
        if buf and len(buf) + 1 + len(para) > chunk_size:
            chunks.append(buf)
            buf = para
        else:
            buf = para if not buf else buf + "\n" + para
    if buf:
        chunks.append(buf)
    return chunks


def document_title(text):
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


def prompt_view(source_text, span, index, total):
    """LLM-facing text. The stored passage must remain `span`."""
    title = document_title(source_text)
    categories = document_categories(title=title, text=source_text)
    varieties = [name for name in match_commodities(source_text) if name not in categories]
    catalog = "、".join(categories) if categories else "（标题里没有词典内的品类）"
    grades = "、".join(varieties) if varieties else "（无）"
    return (
        "[文档标题]\n%s\n"
        "[文档品类]\n%s\n"
        "文档品类来自标题。下面的文内品种是该品类下的实体，不要把它们再写成一个品类。\n"
        "[文内品种]\n%s\n"
        "原文里没有写出的名称，不要写成实体、事件或关系。\n"
        "[片段位置]\n第%d段，共%d段\n"
        "[原文]\n%s"
    ) % (title, catalog, grades, index, total, span)
