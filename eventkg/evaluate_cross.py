# -*- coding: utf-8 -*-
"""Hand-labeled comparison of legacy and abductive cross-document relations.

Unreviewed predictions are listed for adjudication. They are not treated as
negatives, and a headline F1 is withheld until that list is empty. Scripted
proposals exercise the scorer only; they are not a model-accuracy report.
"""
import json
from pathlib import Path

from eventkg.graph_induction import index_evidence, induce, retrieve_neighborhood, select_graph
from eventkg.link import abstract_rules

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "tests" / "fixtures" / "cross_gold.json"
LEGACY_PAIRS = (
    ("r1:hot", "r2:price"),
    ("r1:ship", "r2:arrive"),
    ("r2:arrive", "r3:stock"),
    ("r1:price", "r4:cost"),
    ("r1:coke", "r4:cost"),
    ("r1:price", "r2:price"),
    ("r1:price", "r4:margin"),
    ("r1:inv", "r2:price"),
    ("r1:price", "r3:mood"),
)


def iron_ore_slice(extended=False):
    """Weekly dates from the ten-report set, with one hand-checked passage per labeled pair."""
    documents = [
        {"id": "r1", "report_date": "2026-04-17", "title": "铁矿石周刊20260417"},
        {"id": "r2", "report_date": "2026-04-24", "title": "铁矿石周刊20260424"},
        {"id": "r3", "report_date": "2026-05-15", "title": "铁矿石周刊20260515"},
        {"id": "r4", "report_date": "2026-05-22", "title": "铁矿石周刊20260522"},
    ]

    def event(event_id, doc, commodity, indicator, direction, evidence, salience, dim=""):
        var = "%s|%s||当期" % (commodity, indicator)
        return {"id": event_id, "doc_id": doc, "commodity": commodity, "indicator": indicator, "dim": dim,
                "var": var, "abstract": var + "|" + direction, "dir": direction, "status": "已发生",
                "name": event_id, "evidence": evidence, "salience": salience, "scope": "", "sections": ["综述"],
                "page": 1}

    events = [
        event("r1:hot", "r1", "铁水", "日均铁水产量", "上升", "日均铁水产量上升", "一般"),
        event("r1:price", "r1", "铁矿石", "价格", "上升", "本期矿价上涨", "一般", "价格"),
        event("r1:ship", "r1", "铁矿石", "发运量", "下降", "澳洲发运下降", "一般", "供应"),
        event("r1:inv", "r1", "铁矿石", "港口库存", "上升", "港口库存上升", "一般", "库存"),
        event("r1:coke", "r1", "焦炭", "价格", "上升", "焦炭价格上涨", "一般", "价格"),
        event("r1:px", "r1", "铁矿石", "价格", "下降", "库存上升导致价格下降", "一般", "价格"),
        event("r2:price", "r2", "铁矿石", "价格", "上升", "2026-04-17日均铁水产量上升带动本期矿价上涨", "核心", "价格"),
        event("r2:arrive", "r2", "铁矿石", "到港量", "下降", "2026-04-17发运下降导致本期澳矿到港下降", "核心", "供应"),
        event("r2:margin", "r2", "钢厂", "盈利率", "上升", "钢厂盈利率回升至64%", "一般", "利润"),
        event("r2:drain", "r2", "铁矿石", "疏港量", "下降", "疏港量下降", "一般", "物流"),
        event("r2:build", "r2", "铁矿石", "港口库存", "上升", "港口库存累积", "一般", "库存"),
        event("r3:stock", "r3", "铁矿石", "港口库存", "下降", "2026-04-24到港下降带动本期港口库存下降", "核心", "库存"),
        event("r3:mood", "r3", "铁矿石", "市场情绪", "上升", "黑色系情绪回暖", "核心", "市场情绪"),
        event("r4:margin", "r4", "钢厂", "盈利率", "下降", "矿价上涨挤压钢厂利润，盈利率下降", "核心", "利润"),
        event("r4:cost", "r4", "钢厂", "成本", "上升",
              "2026-04-17矿价上涨与2026-04-17焦炭价格上涨共同推高本期钢厂成本", "核心", "成本"),
    ]
    if extended:
        events.extend([
            event("r3:freight", "r3", "铁矿石", "海运费", "上升", "2026-05-15海运费上涨。", "核心", "物流"),
            event("r4:push", "r4", "铁矿石", "发运量", "下降", "2026-04-17发运下降推动2026-05-15海运费上涨。", "核心", "供应"),
        ])
    relations = [
        {"head": "r1:inv", "tail": "r1:px", "type": "导致", "evidence": "库存上升导致价格下降", "confidence": 0.9},
        {"head": "r2:drain", "tail": "r2:build", "type": "导致", "evidence": "疏港下降导致库存累积", "confidence": 0.9},
    ]
    return documents, events, relations


class SliceChat:
    """Scripted hypotheses for the scorer. Not a stand-in for measured model accuracy."""

    def __init__(self):
        self.users = []
        self.stats = {"calls": 0, "cached": 0, "failed": 0}

    def complete_json(self, system, user, max_tokens=4096, tag=""):
        self.stats["calls"] += 1
        self.users.append(user)
        target = json.loads(user)["target"]["id"]
        table = {
            "r2:price": [
                {"id": "h-hot", "kind": "单一直接原因", "why": "后文写明铁水带动矿价", "missing": "", "needs_retrieval": False,
                 "nodes": ["r1:hot", "r2:price"],
                 "edges": [{"head": "r1:hot", "tail": "r2:price", "relation": "促进", "evidence_ids": ["ev:r2:price"],
                            "counter_evidence_ids": [], "why": "该句同时点名铁水和矿价", "missing": "", "needs_retrieval": False}]},
                {"id": "h-series", "kind": "时间或主题关联", "why": "两期价格", "missing": "", "needs_retrieval": False,
                 "edges": [{"head": "r1:price", "tail": "r2:price", "relation": "导致", "evidence_ids": ["ev:r1:price"],
                            "counter_evidence_ids": [], "why": "只有前后两期价格", "missing": "", "needs_retrieval": False}]},
                {"id": "h-rule", "kind": "单一直接原因", "why": "套用库存压价", "missing": "", "needs_retrieval": False,
                 "edges": [{"head": "r1:inv", "tail": "r2:price", "relation": "抑制", "evidence_ids": ["ev:rel:r1:inv:r1:px:0"],
                            "counter_evidence_ids": [], "why": "历史句式不能替代本期原文", "missing": "", "needs_retrieval": False}]},
            ],
            "r2:arrive": [
                {"id": "h-ship", "kind": "单一直接原因", "why": "后文写明发运导致到港下降", "missing": "", "needs_retrieval": False,
                 "edges": [{"head": "r1:ship", "tail": "r2:arrive", "relation": "导致", "evidence_ids": ["ev:r2:arrive"],
                            "counter_evidence_ids": [], "why": "该句点名发运和到港", "missing": "", "needs_retrieval": False}]},
            ],
            "r3:stock": [
                {"id": "h-arrive", "kind": "间接影响", "why": "到港下降再带动库存", "missing": "", "needs_retrieval": False,
                 "edges": [{"head": "r2:arrive", "tail": "r3:stock", "relation": "促进", "evidence_ids": ["ev:r3:stock"],
                            "counter_evidence_ids": [], "why": "该句点名到港和库存", "missing": "", "needs_retrieval": False}]},
            ],
            "r4:margin": [
                {"id": "h-skip", "kind": "单一直接原因", "why": "后文写挤压利润", "missing": "中间盈利率曾经回升", "needs_retrieval": False,
                 "edges": [{"head": "r1:price", "tail": "r4:margin", "relation": "抑制", "evidence_ids": ["ev:r4:margin"],
                            "counter_evidence_ids": [], "why": "该句点名矿价和利润", "missing": "", "needs_retrieval": False}]},
            ],
            "r4:cost": [
                {"id": "h-both", "kind": "多个共同原因", "why": "同一句点名矿价和焦炭", "missing": "", "needs_retrieval": False,
                 "nodes": ["r1:price", "r1:coke", "r4:cost"],
                 "edges": [
                     {"head": "r1:price", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                      "counter_evidence_ids": [], "why": "该句点名矿价和成本", "missing": "", "needs_retrieval": False},
                     {"head": "r1:coke", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                      "counter_evidence_ids": [], "why": "该句点名焦炭和成本", "missing": "", "needs_retrieval": False},
                 ]},
            ],
            "r3:mood": [
                {"id": "h-none", "kind": "证据不足", "why": "情绪回暖没有对应的原因句", "missing": "关系支持证据", "needs_retrieval": False,
                 "edges": []},
            ],
        }
        return {"hypotheses": table.get(target, [{"id": "h0", "kind": "证据不足", "why": "没有提案", "edges": []}])}


class AblationChat(SliceChat):
    """Same scripted proposer, plus the pairs that separate the four selection modes."""

    def complete_json(self, system, user, max_tokens=4096, tag=""):
        parsed = super().complete_json(system, user, max_tokens=max_tokens, tag=tag)
        target = json.loads(user)["target"]["id"]
        if target == "r4:cost":
            parsed["hypotheses"].insert(0, {
                "id": "h-only", "kind": "单一直接原因", "why": "只保留矿价这一条", "missing": "", "needs_retrieval": False,
                "nodes": ["r1:price", "r4:cost"],
                "edges": [{"head": "r1:price", "tail": "r4:cost", "relation": "促进", "evidence_ids": ["ev:r4:cost"],
                           "counter_evidence_ids": [], "why": "该句点名矿价和成本", "missing": "", "needs_retrieval": False}],
            })
            parsed["hypotheses"].append({
                "id": "h-chain", "kind": "间接影响", "why": "铁水位于矿石上游", "missing": "没有把铁水写成成本的原因",
                "needs_retrieval": False, "nodes": ["r1:hot", "r4:cost"],
                "edges": [{"head": "r1:hot", "tail": "r4:cost", "relation": "导致",
                           "evidence_ids": ["ev:r1:hot", "ev:r4:cost"], "counter_evidence_ids": [],
                           "why": "两句分别写了铁水和成本", "missing": "", "needs_retrieval": False}],
            })
        elif target == "r3:freight":
            parsed["hypotheses"] = [{
                "id": "h-effect", "kind": "单一直接原因", "why": "本期只写了海运费和发运走弱", "missing": "原因侧的触发词在另一篇",
                "needs_retrieval": False, "nodes": ["r1:ship", "r3:freight"],
                "edges": [{"head": "r1:ship", "tail": "r3:freight", "relation": "促进", "evidence_ids": ["ev:r3:freight"],
                           "counter_evidence_ids": [], "why": "这句点到发运和海运费，但没有因果触发词",
                           "missing": "", "needs_retrieval": False}],
            }]
        elif target == "r4:push":
            parsed["hypotheses"] = [{
                "id": "h-cause", "kind": "单一直接原因", "why": "这句写了发运下降并带有触发词", "missing": "结果侧的海运费在另一篇",
                "needs_retrieval": False, "nodes": ["r1:ship", "r3:freight"],
                "edges": [{"head": "r1:ship", "tail": "r3:freight", "relation": "促进", "evidence_ids": ["ev:r4:push"],
                           "counter_evidence_ids": [], "why": "这句把触发词落在发运上，没有写出海运费",
                           "missing": "", "needs_retrieval": False}],
            }]
        return parsed


def ablation_gold():
    return load_gold() + [
        {"head": "r1:ship", "tail": "r3:freight", "label": 1, "category": "cross_document_implicit",
         "expected_class": "INFERRED_SUPPORTED", "direction": "head_causes_tail", "reviewed": True, "scope": "cross",
         "note": "构造对照：第三篇写明 2026-04-17 发运下降推动 2026-05-15 海运费上涨，结果所在的那篇只写海运费上涨。单独一篇不能确认，合在一起才是跨文档隐式关系。这条不在 cross_gold.json 里。"},
        {"head": "r1:hot", "tail": "r4:cost", "label": 0, "category": "industry_chain_no_cause",
         "expected_class": "NONE", "direction": "none", "reviewed": True, "scope": "cross",
         "note": "铁水事实和成本句拼在一起。成本句的触发词说的是矿价和焦炭，不能借产业链确认铁水导致成本。"},
    ]


def implicit_scores(edges, gold_pairs):
    """Cross-document implicit causes. A false reviewed pair counts; a true explicit pair does not dilute recall."""
    positives = {(item["head"], item["tail"]) for item in gold_pairs
                 if item.get("reviewed") and item.get("scope") == "cross" and item.get("label") == 1
                 and item.get("category") == "cross_document_implicit"}
    false = {(item["head"], item["tail"]) for item in gold_pairs
             if item.get("reviewed") and item.get("scope") == "cross" and item.get("label") == 0}
    predicted = {(edge["head"], edge["tail"]) for edge in edges}
    metrics = _metrics(len(predicted & positives), len(predicted & false), len(positives - predicted))
    pending = pending_reviews(predicted, gold_pairs)
    metrics["pending"] = pending
    metrics["complete"] = not pending
    if pending:
        metrics["f1"] = None
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["withheld"] = "存在未复核预测，不能计算正式 F1"
    return metrics


def load_gold(path=GOLD_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["pairs"]


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def pending_reviews(predicted, gold_pairs):
    """Predictions the annotators have not judged. They are not negatives and are not dropped."""
    reviewed = {(item["head"], item["tail"]) for item in gold_pairs if item.get("reviewed") and item.get("scope") == "cross"}
    return [{"head": head, "tail": tail, "reviewed": False} for head, tail in sorted(set(predicted) - reviewed)]


def score_relations(predicted, gold_pairs):
    gold = [item for item in gold_pairs if item.get("reviewed") and item.get("scope") == "cross"]
    table = {(item["head"], item["tail"]): item for item in gold}
    predicted = set(predicted)
    pending = pending_reviews(predicted, gold_pairs)
    judged = predicted & set(table)
    tp = sum(1 for pair in judged if table[pair]["label"] == 1)
    fp = sum(1 for pair in judged if table[pair]["label"] == 0)
    fn = sum(1 for pair, item in table.items() if item["label"] == 1 and pair not in predicted)
    metrics = _metrics(tp, fp, fn)
    metrics["pending"] = pending
    metrics["complete"] = not pending
    if pending:
        metrics["f1"] = None
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["withheld"] = "存在未复核预测，不能计算正式 F1"
    return metrics


def candidate_recall_at_k(documents, events, relations, gold_pairs, k=8):
    rules = abstract_rules(events, relations)
    by_id = {event["id"]: event for event in events}
    positives = [item for item in gold_pairs if item.get("label") == 1 and item.get("scope") == "cross"]
    hit = 0
    for item in positives:
        target = by_id.get(item["tail"])
        if target is None:
            continue
        neighborhood = retrieve_neighborhood(target, events, documents, relations, rules, top_k=k)
        if item["head"] in {row["id"] for row in neighborhood["candidates"]}:
            hit += 1
    return {"k": k, "positives": len(positives), "hit": hit,
            "recall": round(hit / len(positives), 4) if positives else 0.0}


def direction_accuracy(edges, gold_pairs):
    table = {(item["head"], item["tail"]): item for item in gold_pairs if item.get("scope") == "cross" and item.get("reviewed")}
    matched = [edge for edge in edges if (edge["head"], edge["tail"]) in table and table[(edge["head"], edge["tail"])]["label"] == 1]
    correct = sum(1 for edge in matched if table[(edge["head"], edge["tail"])].get("direction") == "head_causes_tail")
    return {"matched": len(matched), "correct": correct,
            "accuracy": round(correct / len(matched), 4) if matched else None}


def evidence_precision(edges):
    supported = [edge for edge in edges if edge.get("relation_class") in ("ASSERTED_CAUSE", "INFERRED_SUPPORTED") and edge.get("evidence_ids")]
    return {"confirmed": len(edges), "with_support_id": len(supported),
            "precision": round(len(supported) / len(edges), 4) if edges else None}


def path_precision(edges, gold_pairs):
    """A path is precise only when every hop is a reviewed positive cause. No path is not a perfect score."""
    gold = {(item["head"], item["tail"]) for item in gold_pairs if item.get("label") == 1 and item.get("scope") == "cross"}
    forward = {}
    for edge in edges:
        forward.setdefault(edge["head"], []).append(edge["tail"])
    paths = []

    def walk(node, trail):
        extended = False
        for nxt in forward.get(node, []):
            if nxt in trail:
                continue
            extended = True
            walk(nxt, trail + [nxt])
        if not extended and len(trail) >= 3:
            paths.append(trail)

    for start in forward:
        if all(start not in children for children in forward.values()):
            walk(start, [start])
    if not paths:
        return {"paths": 0, "precise": 0, "precision": None, "note": "没有长度不少于 2 的因果路径"}
    precise = 0
    for trail in paths:
        hops = list(zip(trail, trail[1:]))
        if all(hop in gold for hop in hops):
            precise += 1
    return {"paths": len(paths), "precise": precise, "precision": round(precise / len(paths), 4)}


def compare_slice(gold_path=GOLD_PATH):
    documents, events, relations = iron_ore_slice()
    chat = SliceChat()
    edges, payload = induce(chat, events, relations, documents, workers=1)
    gold = load_gold(gold_path)
    predicted = [(edge["head"], edge["tail"]) for edge in edges]
    return {
        "report_type": "functional_test",
        "model_accuracy": None,
        "note": "提案来自脚本，不是 Qwen 的准确率。正式 F1 只在人工复核完所有预测后给出。",
        "legacy": score_relations(LEGACY_PAIRS, gold),
        "abductive": score_relations(predicted, gold),
        "candidate_recall_at_8": candidate_recall_at_k(documents, events, relations, gold, 8),
        "direction_accuracy": direction_accuracy(edges, gold),
        "evidence_supported_edge_precision": evidence_precision(edges),
        "path_precision": path_precision(edges, gold),
        "edges": edges,
        "payload": payload,
        "prompts": chat.users,
    }


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value))


def write_functional_report(scores, path):
    """Candidate lists, competing graphs, evidence ids and scores. This file is not graph.json."""
    payload = scores["payload"]
    targets = []
    for row in payload["targets"]:
        targets.append({
            "target": row["target"],
            "decision": row["decision"],
            "rejection_reason": row["rejection_reason"],
            "recall_stats": row["recall_stats"],
            "candidates": [{
                "id": item["id"], "time_relation": item["time_relation"], "causal_use": item["causal_use"],
                "rank": item["rank"], "recall": item["recall"],
            } for item in row["candidates"]],
            "hypotheses": [{
                "id": item["id"], "kind": item["kind"], "why": item["why"], "missing": item["missing"],
                "needs_retrieval": item["needs_retrieval"], "ready": item["ready"], "reasons": item["reasons"],
                "graph_scores": item["graph_scores"], "edge_scores": item["edge_scores"],
            } for item in row["hypotheses"]],
            "confirmed_edges": row["edges"],
            "possible_edges": row["possible_edges"],
        })
    report = {
        "report_type": scores["report_type"],
        "model_accuracy": None,
        "note": scores["note"],
        "legacy": scores["legacy"],
        "abductive": scores["abductive"],
        "candidate_recall_at_8": scores["candidate_recall_at_8"],
        "direction_accuracy": scores["direction_accuracy"],
        "evidence_supported_edge_precision": scores["evidence_supported_edge_precision"],
        "path_precision": scores["path_precision"],
        "confirmed_edges": scores["edges"],
        "global": payload["global"],
        "targets": targets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    return path


def run_ablation():
    """Four modes on one constructed slice. The numbers describe this slice only."""
    documents, events, relations = iron_ore_slice(extended=True)
    gold = ablation_gold()
    modes = (
        ("A", False, False),
        ("B", True, False),
        ("C", False, True),
        ("D", True, True),
    )
    report = {
        "report_type": "functional_ablation",
        "model_accuracy": None,
        "note": "A/B/C/D 的数字来自脚本提案和构造切片，不是十篇研报上的模型效果。构造样本即使出现 1.0，也不能当作方法结论。",
        "samples": {
            "real_report": {"status": "not_run", "method_f1": None,
                            "note": "没有对十篇原文调用 Qwen，也没有真实报告样本的 F1。"},
            "constructed": {"status": "scripted", "gold_file_cross_positives": 5,
                            "ablation_cross_positives": 6,
                            "why_counts_differ": "tests/fixtures/cross_gold.json 里跨文档正例是 5 条。本报告的关系召回分母是 6，因为 ablation_gold() 在运行时追加了构造的 ship→freight，这条不在打包的 Gold 文件里。"},
        },
        "modes": {},
    }
    for name, multi_source, joint in modes:
        edges, payload = induce(AblationChat(), events, relations, documents, workers=1,
                                multi_source=multi_source, joint=joint)
        global_result = payload["global"]
        entry = {
            "mode": name,
            "multi_source": multi_source,
            "joint": joint,
            "relation": score_relations([(edge["head"], edge["tail"]) for edge in edges], gold),
            "implicit": implicit_scores(edges, gold),
            "path_precision": path_precision(edges, gold),
            "confirmed": [{"head": edge["head"], "tail": edge["tail"],
                           "relation_class": edge.get("relation_class"),
                           "evidence_ids": edge.get("evidence_ids")} for edge in edges],
        }
        if joint:
            entry["joint_score"] = global_result.get("joint_score")
            entry["competitions"] = global_result.get("competitions")
            entry["uncertain"] = [{"head": edge["head"], "tail": edge["tail"],
                                   "rejection_reason": edge.get("rejection_reason")}
                                  for edge in global_result.get("possible") or []]
            entry["rejected"] = [{"head": edge["head"], "tail": edge["tail"],
                                  "relation": edge.get("relation"),
                                  "rejection_reason": edge.get("rejection_reason")}
                                 for edge in global_result.get("rejected") or []]
            entry["edges"] = global_result.get("edges")
        report["modes"][name] = entry
    return report


def write_ablation_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=_jsonable), encoding="utf-8")
    return path


def score_human_review(path=None):
    """Check real-report sentences. The result is a review record, not a method F1."""
    path = Path(path) if path else ROOT / "tests" / "fixtures" / "human_implicit_review.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["items"]
    checked = []
    for row in rows:
        events = []
        for spec in row["events"]:
            events.append({
                "id": spec["id"], "doc_id": spec["doc_id"], "commodity": spec["commodity"],
                "indicator": spec["indicator"], "direction": spec["direction"],
                "evidence": spec["evidence"], "salience": "核心", "status": "已发生",
                "var": "%s|%s||当期" % (spec["commodity"], spec["indicator"]), "scope": "",
            })
        evidence = index_evidence(events, row["documents"], [])
        choice = select_graph(next(event for event in events if event["id"] == row["target"]), [{
            "id": row["hypothesis_id"], "kind": "单一直接原因", "why": row["reason"], "edges": row["edges"],
        }], [{"id": edge["head"]} for edge in row["edges"]], events, row["documents"], evidence)
        confirmed = bool(choice["edges"])
        scored = choice["hypotheses"][0]["edge_scores"][0] if choice["hypotheses"] and choice["hypotheses"][0]["edge_scores"] else {}
        checked.append({
            "id": row["id"], "source": "real_report_sentence", "document": row["document"],
            "quote": row["quote"], "human_label": row["label"], "human_reason": row["reason"],
            "system_confirmed": confirmed, "system_class": scored.get("relation_class", "NONE"),
            "model_score": scored.get("model_score"), "system_reasons": choice["hypotheses"][0]["reasons"] if choice["hypotheses"] else [],
            "agrees_with_human": confirmed == bool(row["label"]),
        })
    return {
        "kind": "human_review", "method_f1": None,
        "note": "句子来自十篇研报原文。人工只判断能不能当成跨文档隐式因果。这里不计算 F1，也不并进构造切片的分数。",
        "reviewed": len(checked), "agreements": sum(1 for item in checked if item["agrees_with_human"]),
        "items": checked,
    }


def main():
    scores = compare_slice()
    public = {key: scores[key] for key in (
        "report_type", "model_accuracy", "note", "legacy", "abductive", "candidate_recall_at_8",
        "direction_accuracy", "evidence_supported_edge_precision", "path_precision")}
    destination = ROOT / "example" / "generated" / "abductive_functional_report.json"
    write_functional_report(scores, destination)
    ablation = run_ablation()
    ablation["human_review"] = score_human_review()
    ablation["fakechat"] = {
        "kind": "scripted_functional_test", "method_f1": None,
        "note": "compare_slice 使用脚本提案。下面的分数只说明打分器在手工切片上的行为，不是 Qwen 准确率。",
        "legacy": scores["legacy"], "abductive": scores["abductive"],
        "path_precision": scores["path_precision"],
    }
    experiment = ROOT / "example" / "experiment" / "joint_ablation_report.json"
    write_ablation_report(ablation, experiment)
    ablation_path = ROOT / "example" / "generated" / "joint_ablation_report.json"
    write_ablation_report(ablation, ablation_path)
    public["detail_report"] = str(destination)
    public["ablation_report"] = str(ablation_path)
    public["ablation"] = {name: {"relation": item["relation"], "implicit": item["implicit"],
                                 "path_precision": item["path_precision"], "confirmed": item["confirmed"]}
                          for name, item in ablation["modes"].items()}
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
