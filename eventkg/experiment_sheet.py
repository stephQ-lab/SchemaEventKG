# -*- coding: utf-8 -*-
"""Annotation sheet for one real Qwen run. Constructed-slice F1 is not an accuracy."""

EXPLICIT = "explicit_textual_cause"
SEGMENTED = "segmented_cause"
IMPLICIT = "implicit_without_trigger"
UNCLASSIFIED = "UNCLASSIFIED"
KIND_NOTE = {
    EXPLICIT: "有明确文本因果陈述的跨文档关系",
    SEGMENTED: "分段因果组合",
    IMPLICIT: "完全没有显式因果陈述的隐式关系",
    UNCLASSIFIED: "缺少证据类型，不能根据方法名称猜测，交给人工标注",
}


def edge_kind(edge):
    """Label an already scored edge. This does not rescore it."""
    components = edge.get("components") or {}
    path = components.get("inference_path") or ""
    if path == "implicit_segment" or components.get("segment"):
        return SEGMENTED
    if path in ("asserted", "explicit_third_document") or edge.get("relation_class") == "ASSERTED_CAUSE":
        return EXPLICIT
    if path == "implicit" or edge.get("relation_class") == "INFERRED_SUPPORTED":
        return IMPLICIT
    return UNCLASSIFIED


def _evidence(edge):
    return {
        "evidence_ids": list(edge.get("evidence_ids") or edge.get("support_ids") or []),
        "counter_evidence_ids": list(edge.get("counter_evidence_ids") or []),
        "rule_evidence": edge.get("rule_evidence") or "",
        "head_evidence": edge.get("head_evidence") or "",
        "tail_evidence": edge.get("tail_evidence") or "",
        "reason": edge.get("reason") or edge.get("why") or edge.get("rejection_reason") or "",
        "relation_class": edge.get("relation_class") or "",
        "inference_path": (edge.get("components") or {}).get("inference_path") or "",
    }


def _pack(edge, status):
    return {
        "head": edge.get("head"),
        "tail": edge.get("tail"),
        "relation": edge.get("relation") or edge.get("type") or "",
        "status": status,
        "kind": edge_kind(edge),
        "kind_note": KIND_NOTE[edge_kind(edge)],
        "evidence": _evidence(edge),
    }


def _coverage(target_ids, covered_ids):
    targets = list(target_ids)
    covered = [item for item in targets if item in set(covered_ids)]
    return {"targets": len(targets), "with_candidates": len(covered),
            "coverage": round(len(covered) / len(targets), 4) if targets else None}


def candidate_recall_at_k(per_target, gold_pairs, k=8):
    """Recall of human positives inside each target's candidate list. No gold means no recall."""
    positives = [item for item in (gold_pairs or []) if item.get("label") == 1 and item.get("scope") == "cross"]
    if not positives:
        return {"k": k, "positives": 0, "hit": 0, "recall": None,
                "note": "没有人工正例，不能计算 Candidate Recall@K。构造切片 F1 不是这次的模型准确率。"}
    by_tail = {}
    for row in per_target:
        by_tail.setdefault(row["target"], []).append(row["candidates"])
    hit = 0
    for item in positives:
        lists = by_tail.get(item.get("tail")) or []
        found = False
        for candidates in lists:
            if item.get("head") in candidates[:k]:
                found = True
        if found:
            hit += 1
    return {"k": k, "positives": len(positives), "hit": hit,
            "recall": round(hit / len(positives), 4)}


def sheet_for_method(method, manifest, target_ids, per_target, confirmed, uncertain, rejected, gold_pairs=None, k=8):
    covered = [row["target"] for row in per_target if row.get("candidates")]
    return {
        "method": method,
        "model_accuracy": None,
        "constructed_slice_f1_used": False,
        "manifest": manifest,
        "target_coverage": _coverage(target_ids, covered),
        "candidate_recall_at_k": candidate_recall_at_k(per_target, gold_pairs, k),
        "candidates": per_target,
        "confirmed": [_pack(edge, "confirmed") for edge in confirmed],
        "uncertain": [_pack(edge, "uncertain") for edge in uncertain],
        "rejected": [_pack(edge, "rejected") for edge in rejected],
        "counts": {"confirmed": len(confirmed), "uncertain": len(uncertain), "rejected": len(rejected)},
        "kind_counts": _kind_counts(confirmed + uncertain + rejected),
    }


def _kind_counts(edges):
    counts = {EXPLICIT: 0, SEGMENTED: 0, IMPLICIT: 0, UNCLASSIFIED: 0}
    for edge in edges:
        counts[edge_kind(edge)] += 1
    return counts


def write_sheet(path, sheet):
    from pathlib import Path
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sheet, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path
