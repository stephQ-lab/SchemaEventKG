# -*- coding: utf-8 -*-
"""Placeholders for a human gold run. Missing labels stay null."""


def empty_gold_report():
    """No gold annotation has been scored. These nulls are not zeros."""
    return {
        "candidate_recall": {"10": None, "20": None, "50": None},
        "event_coreference": {"precision": None, "recall": None, "f1": None},
        "relation": {"precision": None, "recall": None, "f1": None},
        "relation_evidence_support_rate": None,
        "strict_path_accuracy": None,
        "retrieval_seconds": None,
        "token_cost": None,
        "qwen": "NOT_RUN",
        "dense_hybrid": "NOT_RUN",
        "modes": ["legacy", "abductive+bm25", "abductive+dense", "abductive+hybrid"],
    }
