# -*- coding: utf-8 -*-
"""Full-text passages and table observations. Positions stay null when they are not known."""
import re

from eventkg.lexicon import compact
from eventkg.retrieval import search_texts


def _place(text, document, window=None, page=None, section=None, paragraph_id=None):
    narrative = document.get("narrative") or ""
    char_start = char_end = None
    if text and narrative and narrative.count(text) == 1:
        char_start = narrative.find(text)
        char_end = char_start + len(text)
    sentence_id = None
    if window and window.get("id") and text and text in (window.get("text") or ""):
        offset = window["text"].find(text)
        sentence_id = "%s:s%d" % (window["id"], 1 + len(re.findall(r"[。；！？]", window["text"][:offset])))
        if not paragraph_id:
            paragraph_id = window.get("id")
    return {
        "page": page if page not in ("", None) else None,
        "section": section or None,
        "paragraph_id": paragraph_id or None,
        "sentence_id": sentence_id,
        "char_start": char_start,
        "char_end": char_end,
    }


def index_corpus(documents):
    """Index narrative paragraphs and table rows. Cited event text is not required."""
    by_text = {}
    items = []
    for document in documents or []:
        for window in document.get("windows") or []:
            section = ""
            names = window.get("sections") or []
            if names:
                section = names[0]
            parts = [part.strip() for part in re.split(r"\n\s*\n", window.get("text") or "") if part.strip()]
            for index, text in enumerate(parts):
                _add(items, by_text, "ev:passage:%s:%d" % (window.get("id") or document.get("id"), index),
                     text, document, "paragraph", window=window, section=section, paragraph_id=window.get("id"))
        for observation in document.get("observations") or []:
            _add(items, by_text, observation.get("id") or "", observation.get("text") or "", document,
                 "observation", page=observation.get("page"), section=observation.get("section") or "")
    return items


def _add(items, by_text, evidence_id, text, document, source_type, window=None, page=None, section=None,
         paragraph_id=None):
    key = compact(text)
    if not key or not evidence_id:
        return
    place = _place(text, document, window, page, section, paragraph_id)
    source = {
        "evidence_id": evidence_id, "document_id": document.get("id"), "source_type": source_type,
        "report_date": document.get("report_date") or "",
        "report_series_id": document.get("report_series_id") or "",
        "page": place["page"], "section": place["section"],
        "char_start": place["char_start"], "char_end": place["char_end"],
        "place_status": "UNIQUE" if place["char_start"] is not None else "AMBIGUOUS",
    }
    if key in by_text:
        primary = by_text[key]
        primary["sources"].append(source)
        if evidence_id not in primary["evidence_ids"]:
            primary["evidence_ids"].append(evidence_id)
        return
    item = {
        "evidence_id": evidence_id, "document_id": document.get("id") or "",
        "source_type": source_type, "text": text, "counted": True,
        "sources": [source], "evidence_ids": [evidence_id],
        "page": place["page"], "section": place["section"], "paragraph_id": place["paragraph_id"],
        "sentence_id": place["sentence_id"], "char_start": place["char_start"], "char_end": place["char_end"],
    }
    by_text[key] = item
    items.append(item)


def retrieve_passages(questions, passages, method="bm25", encoder=None, top_k=4, cache=None,
                      max_passages=8, max_chars=2000):
    """Bounded lookup. Duplicate text is already one row in the index."""
    cache = cache if cache is not None else {}
    chosen = []
    seen = set()
    used = 0
    for question in questions:
        if len(chosen) >= max_passages or used >= max_chars:
            break
        hits = search_texts(question, passages, method="bm25" if method == "jaccard" else method,
                            encoder=encoder, top_k=top_k, cache=cache)
        by_id = {item["evidence_id"]: item for item in passages}
        for hit in hits:
            item = by_id.get(hit["id"])
            if item is None or item["evidence_id"] in seen:
                continue
            if used + len(item["text"]) > max_chars and chosen:
                return chosen
            seen.add(item["evidence_id"])
            chosen.append(dict(item, retrieved=True, retrieval_query=question))
            used += len(item["text"])
            if len(chosen) >= max_passages:
                break
    return chosen
