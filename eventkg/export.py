# -*- coding: utf-8 -*-
"""graph.json, event_layer.cypher and a readable report."""
import json
from collections import Counter
from pathlib import Path

from atlas_rag.kg_construction.commodity import belongs_edge
from atlas_rag.kg_construction.cypher_writer import write_cypher
from atlas_rag.kg_construction.neo4j_export import _uid
from eventkg.lexicon import DIMENSIONS, PARENT, shared_scope


def variable_title(key):
    commodity, indicator, scope, period = key.split("|")
    text = commodity + ("·" + scope if scope else "") + "·" + indicator
    return text + ("（%s）" % period if period != "当期" else "")


def abstract_title(key):
    var, direction = key.rsplit("|", 1)
    return variable_title(var) + " " + direction


def instance_query_props(event, documents=()):
    """Query view: trend, time, scope and identity. Market quantities stay on the audit node."""
    return {
        "名称": event["name"], "文档": event["doc_id"], "报告日期": _date_of(documents, event["doc_id"]),
        "品种": event["commodity"], "指标": event["indicator"], "口径": event["scope"], "维度": event["dim"],
        "方向": event["dir"], "趋势方向": event.get("level_direction") or event["dir"],
        "动量": event.get("momentum") or "", "比较维度": event.get("comparison_basis") or "",
        "时间": event.get("time") or "",
        "事件时间": event.get("event_time") or "",
        "观测时期": event.get("observation_period") or "",
        "趋势信号": event.get("trend_signals") or [],
        "周期": event.get("period") or "", "状态": event.get("status") or "", "主体": event.get("subject") or "",
        "同义写法": event["names"], "章节": event["sections"],
        "显著性": event["salience"], "变量键": event["var"], "抽象键": event["abstract"],
        "合并成员": event["members"], "实例身份": event.get("event_instance_id") or event["id"],
        "事件类型身份": event.get("event_type_id") or "",
        "事件种类": event.get("event_kind") or "",
        "事件角色": event.get("roles") or {},
    }


class Builder:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def node(self, label, scope, name, props, extra_labels=()):
        uid = _uid(label, scope, name)
        current = self.nodes.get(uid)
        if current is None:
            self.nodes[uid] = {"labels": [label, *extra_labels], "props": dict({"名称": name, "键": name}, **props)}
        else:
            for key, value in props.items():
                if value not in (None, "", []) and current["props"].get(key) in (None, "", []):
                    current["props"][key] = value
        return uid

    def edge(self, head, relation, tail, props=None):
        if not head or not tail or head == tail:
            return
        props = dict(props or {})
        current = self.edges.get((head, relation, tail))
        if current is None:
            self.edges[(head, relation, tail)] = props
            return
        for key, value in props.items():
            if key in ("证据编号", "evidence_ids") and isinstance(value, list):
                merged = list(current.get(key) or [])
                for item in value:
                    if item not in merged:
                        merged.append(item)
                current[key] = merged
            elif current.get(key) in (None, "", []) and value not in (None, "", []):
                current[key] = value


def _is_title_category(name):
    """铁矿石、PVC 这一层：词表里的最大模块，没有上级。"""
    return bool(name) and name in PARENT and not PARENT[name]


def _category_uid(builder, name):
    if not _is_title_category(name):
        return ""
    return builder.node("品类", "", name, {})


_UNCONFIRMED = {
    "UNVERIFIED", "DISPUTED", "MISSING", "FORECASTS", "REJECTED",
    "LEGACY_CANDIDATE", "GENERIC_RELATION_ONLY", "INSTANCE_AMBIGUOUS",
}


def _confirmed_label(record, default):
    status = (record or {}).get("verification_status") or ""
    kind = (record or {}).get("relation_class") or ""
    if status in _UNCONFIRMED or kind in _UNCONFIRMED:
        return "候选关系"
    return default


def _resolve_category(commodity, doc_categories):
    """这篇标题上的品类。宏观事件不因为报告只有一个标题品类就变成该商品。"""
    if commodity == "宏观":
        return ""
    roots = [name for name in (doc_categories or []) if _is_title_category(name)]
    if commodity and commodity != "宏观" and commodity in roots:
        return commodity
    parent = PARENT.get(commodity, "") if commodity and commodity != "宏观" else ""
    if parent and parent in roots:
        return parent
    if len(roots) == 1:
        return roots[0]
    return ""


def _variety_uid(builder, name):
    if not name or name == "宏观":
        return ""
    scope = shared_scope(name, "entity", "")
    return builder.node("实体", scope, name, {"文档": scope})


def _link_commodity(builder, owner, commodity, doc_categories):
    """涉及品类只指向标题品类。品种是实体；词表里的下级才写「属于」。"""
    category = _resolve_category(commodity, doc_categories)
    category_uid = _category_uid(builder, category)
    builder.edge(owner, "涉及品类", category_uid)
    if not commodity or commodity == "宏观" or commodity == category:
        return
    variety = _variety_uid(builder, commodity)
    builder.edge(owner, "品种", variety)
    if category_uid and belongs_edge(commodity, category):
        builder.edge(variety, "属于", category_uid)


def document_successors(documents):
    """Next issue inside one report series. A skipped week is not the next period."""
    from datetime import date as _date
    from eventkg.graph_induction import report_series_id
    series = {}
    for doc in documents or []:
        text = doc.get("report_date") or ""
        if not text:
            continue
        try:
            year, month, day = (int(part) for part in text.split("-"))
            stamped = _date(year, month, day)
        except (TypeError, ValueError):
            continue
        series.setdefault(report_series_id(doc) or doc["id"], []).append((stamped, doc))
    edges = []
    for rows in series.values():
        rows.sort(key=lambda item: (item[0], item[1]["id"]))
        for (left_date, left), (right_date, right) in zip(rows, rows[1:]):
            gap = (right_date - left_date).days
            if gap <= 0 or gap > 42:
                continue
            label = "下一期" if gap <= 10 else "下一份已收录报告"
            edges.append((left["id"], label, right["id"]))
    return edges


def build(result):
    documents = result["documents"]
    events = result["events"]
    by_id = {event["id"]: event for event in events}
    builder = Builder()
    doc_uid, event_uid = {}, {}
    doc_categories = {doc["id"]: doc.get("categories") or [] for doc in documents}
    var_categories = {}
    for event in events:
        var_categories.setdefault(event["var"], []).append(doc_categories.get(event["doc_id"], []))
    for doc in documents:
        doc_uid[doc["id"]] = builder.node("文档", "", doc["id"], {
            "标题": doc["title"], "报告日期": doc["report_date"] or "未给出年份", "文件名": doc["file_name"],
            "来源文件": doc["source_file"], "品类": doc["categories"]})
        for category in doc["categories"]:
            builder.edge(doc_uid[doc["id"]], "涉及品类", _category_uid(builder, category))
    for left_id, label, right_id in document_successors(documents):
        builder.edge(doc_uid[left_id], label, doc_uid[right_id])

    for key, variable in result["variables"].items():
        var_uid = builder.node("指标", "", key, {
            "名称": variable_title(key), "品种": variable["commodity"], "指标": variable["indicator"],
            "口径": variable["scope"], "周期": variable["period"], "维度": variable["dim"],
            "原始写法": variable["raw_forms"], "来源": variable["source"]})
        dim = variable["dim"]
        code, kind = DIMENSIONS.get(dim, ("", ""))
        builder.edge(var_uid, "所属维度", builder.node("维度", "", dim, {"编码": code, "事件类型": kind}))
        lists = var_categories.get(key, [])
        agreed = []
        if lists and all(item == lists[0] for item in lists):
            agreed = lists[0]
        _link_commodity(builder, var_uid, variable["commodity"], agreed)
        if variable["parent"]:
            builder.edge(var_uid, "分项属于", builder.node("指标", "", variable["parent"], {
                "名称": variable_title(variable["parent"])}))

    for key, node in result["abstracts"].items():
        abstract_uid = builder.node("抽象事件", "", key, {
            "名称": abstract_title(key), "方向": node["dir"], "维度": node["dim"],
            "实例数": len(node["instances"]), "文档数": len(node["docs"]), "文档": node["docs"],
            "状态分布": node["statuses"]})
        builder.edge(abstract_uid, "状态变量", _uid("指标", "", node["var"]))

    for event in events:
        uid = builder.node("实例事件", event["doc_id"], event["id"], instance_query_props(event, documents),
                           extra_labels=("事件",))
        event_uid[event["id"]] = uid
        builder.node("证据审计", event["doc_id"], event["id"] + ":audit", {
            "原文": event["evidence"], "全部原文": event["evidences"], "文档": event["doc_id"]})
        builder.edge(uid, "审计", _uid("证据审计", event["doc_id"], event["id"] + ":audit"))
        builder.edge(uid, "来自", doc_uid[event["doc_id"]])
        builder.edge(uid, "实例化", _uid("抽象事件", "", event["abstract"]))
        builder.edge(uid, "状态变量", _uid("指标", "", event["var"]))
        _link_commodity(builder, uid, event["commodity"], doc_categories.get(event["doc_id"], []))
        kind = DIMENSIONS.get(event["dim"], ("", ""))[1]
        if kind:
            builder.edge(uid, "类型", builder.node("事件类型", "", kind, {}))
        for section in event["sections"]:
            section_uid = builder.node("章节", event["doc_id"], section, {"文档": event["doc_id"]})
            builder.edge(uid, "位于", section_uid)
            builder.edge(section_uid, "收录于", doc_uid[event["doc_id"]])
        if event["subject"]:
            scope = shared_scope(event["subject"], "entity", event["doc_id"])
            builder.edge(uid, "主体", builder.node("实体", scope, event["subject"], {"文档": scope}))
        roles = event.get("roles") or {}
        if isinstance(roles, dict):
            for role_name, value in roles.items():
                if isinstance(value, str) and value and role_name not in ("status", "modality"):
                    role_scope = shared_scope(value, "entity", event["doc_id"])
                    builder.edge(uid, role_name, builder.node("实体", role_scope, value, {"文档": role_scope}))

    for relation in result["relations"]:
        label = _confirmed_label(relation, relation["type"])
        builder.edge(event_uid[relation["head"]], label, event_uid[relation["tail"]], {
            "依据": relation["basis"], "原文": relation["evidence"], "全部原文": relation.get("evidences", []),
            "触发词": relation["trigger"], "理由": relation["reason"], "置信度": relation["confidence"],
            "来源": relation["source"],
            "验证状态": relation.get("verification_status") or "",
            "证据编号": relation.get("evidence_ids") or []})
    for link in result["continuity"]:
        builder.edge(event_uid[link["head"]], "事态延续", event_uid[link["tail"]], {
            "变化": link["change"], "间隔天数": link["gap_days"], "置信度": 1.0})
    for link in result["expectations"]:
        builder.edge(event_uid[link["head"]], "预期验证", event_uid[link["tail"]], {
            "结果": link["outcome"], "间隔天数": link["gap_days"], "置信度": 1.0})
    for link in result["cross"]:
        builder.edge(event_uid[link["head"]], _confirmed_label(link, "跨期传导"), event_uid[link["tail"]], {
            "关系": link.get("relation", ""), "理由": link.get("reason", ""),
            "规律原文": link.get("rule_evidence", ""), "规律支持数": link.get("rule_support", 0),
            "原文": link.get("evidence", ""), "假设": link.get("hypothesis", ""),
            "关系类别": link.get("relation_class", ""), "证据": link.get("evidence_ids", []),
            "推断状态": link.get("status", ""), "评分": link.get("components", {}),
            "模型评分": link.get("model_score", link.get("confidence")),
            "评分类型": link.get("score_kind", "model_score"),
            "证据来源": link.get("evidence_sources", []), "替代解释": link.get("alternatives", []),
            "拒绝理由": link.get("rejection_reason", ""),
            "间隔天数": link.get("gap_days", ""), "置信度": link.get("confidence", 0)})
    for link in result["observation_links"]:
        observation = result["observation_index"][link["tail"]]
        obs_uid = builder.node("趋势信号", observation["doc_id"], observation["id"], {
            "章节": observation.get("section") or "", "页码": observation.get("page"),
            "文档": observation["doc_id"],
            "指标": observation.get("indicator_id") or observation.get("row_header") or "",
            "趋势方向": observation.get("level_direction") or observation.get("direction") or "",
            "比较维度": observation.get("comparison_basis") or "",
            "时期": observation.get("observation_period") or "",
            "比较期": observation.get("comparison_period") or "",
            "证据": observation.get("evidence_id") or ""}, extra_labels=("观测",))
        builder.edge(event_uid[link["head"]], "数据支撑", obs_uid)

    for edge in result["abstract_edges"]:
        builder.edge(_uid("抽象事件", "", edge["head"]), edge["type"], _uid("抽象事件", "", edge["tail"]), {
            "支持数": edge["support"], "文档数": len(edge["docs"]), "文档": edge["docs"],
            "跨期支持": edge["cross_period"], "置信度": edge["confidence"], "示例原文": edge["evidence"]})

    for chain in result["reasoning_chains"] + result["local_chains"]:
        names = [by_id[node]["name"] for node in chain["nodes"]]
        chain_uid = builder.node(chain["kind"], "", chain["id"], {
            "路径": _path_text(names, chain["edges"]), "长度": len(chain["edges"]), "文档": chain["docs"],
            "因果边数": chain["causal_edges"], "得分": chain["score"], "起点": names[0], "终点": names[-1]},
            extra_labels=("推理链",))
        for index, node in enumerate(chain["nodes"]):
            builder.edge(chain_uid, "包含", event_uid[node], {"序号": index})
    for chain in result["transmission_chains"]:
        names = [abstract_title(node) for node in chain["nodes"]]
        chain_uid = builder.node("传导链", "", chain["id"], {
            "路径": _path_text(names, chain["edges"]), "长度": len(chain["edges"]), "文档": chain["docs"],
            "支持数": chain["support"], "得分": chain["score"]})
        for index, node in enumerate(chain["nodes"]):
            builder.edge(chain_uid, "包含", _uid("抽象事件", "", node), {"序号": index})
    for index, line in enumerate(result["storylines"], 1):
        line_uid = builder.node("文档主线", line["doc_id"], "%s:主线%d" % (line["doc_id"], index), {
            "标题": line["title"], "文档": line["doc_id"],
            "路径": " → ".join(by_id[node]["name"] for node in line["events"] if node in by_id)})
        builder.edge(line_uid, "来自", doc_uid[line["doc_id"]])
        for order, node in enumerate(line["events"]):
            if node in event_uid:
                builder.edge(line_uid, "包含", event_uid[node], {"序号": order})
    return builder


def _date_of(documents, doc_id):
    for doc in documents:
        if doc["id"] == doc_id:
            return doc["report_date"] or "未给出年份"
    return ""


def _path_text(names, edges):
    parts = [names[0]]
    for name, edge in zip(names[1:], edges):
        label = edge.get("relation") if edge["type"] == "跨期传导" else edge["type"]
        if edge["type"] == "跨期传导":
            label = "跨期" + label
        parts.append("-[%s]→ %s" % (label, name))
    return " ".join(parts)


def _trend_rows(result):
    rows = []
    for event in result.get("events") or []:
        signals = event.get("trend_signals") or []
        if not signals and event.get("dir"):
            signals = [{"level_direction": event.get("dir"), "momentum": event.get("momentum") or "",
                        "comparison_basis": event.get("comparison_basis") or ""}]
        for signal in signals:
            rows.append({
                "event_id": event.get("id"),
                "indicator": event.get("indicator") or "",
                "direction": signal.get("level_direction") or event.get("dir") or "",
                "momentum": signal.get("momentum") or "",
                "comparison_basis": signal.get("comparison_basis") or "",
                "observation_period": event.get("observation_period") or event.get("event_time") or "",
                "comparison_period": signal.get("comparison_period") or "",
                "document_id": event.get("doc_id") or "",
                "evidence_id": "ev:" + (event.get("id") or ""),
            })
    return rows


def _mention_rows(result):
    rows = []
    for event in result.get("events") or []:
        indicator = event.get("indicator") or ""
        rows.append({
            "base_mention_id": event.get("local_id") or event.get("id"),
            "event_instance_id": event.get("event_instance_id") or event.get("id"),
            "document_id": event.get("doc_id") or "",
            "evidence_id": "ev:" + (event.get("id") or ""),
            "alignment_status": "UNIQUE" if indicator else "AMBIGUOUS",
            "alignment_confidence": 1.0 if indicator else 0.0,
        })
    return rows


def write_all(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    builder = build(result)
    edges = [(head, relation, tail, props) for (head, relation, tail), props in builder.edges.items()]
    abductive = result.get("relation_method") == "abductive"
    cypher_name = "event_layer_abductive.cypher" if abductive else "event_layer.cypher"
    graph_name = "graph_abductive.json" if abductive else "graph.json"
    report_json = "report_abductive.json" if abductive else "report.json"
    report_md = "report_abductive.md" if abductive else "report.md"
    node_count, edge_count = write_cypher(str(output / cypher_name), builder.nodes, edges, header=(
        "事件层：实例事件按文档区分；指标=品种|指标|口径|周期；抽象事件=指标+方向。",
        "品类只来自文章标题（铁矿石、PVC 这一层）。正文里的品种是实体，词表下级用「属于」连到标题品类。",
        "跨文档边：事态延续（同指标相邻两期）、预期验证（预测→下一期实测）、跨期传导（经核验的跨期因果）。",
    ))
    serializable = {key: value for key, value in result.items() if key not in ("documents", "observation_index")}
    serializable["trend_signals"] = _trend_rows(result)
    serializable["mention_links"] = _mention_rows(result)
    serializable["documents"] = [{key: doc[key] for key in ("id", "title", "file_name", "report_date", "categories")}
                                 for doc in result["documents"]]
    if not abductive or not (output / "graph.json").exists():
        (output / graph_name).write_text(json.dumps(serializable, ensure_ascii=False, indent=1, default=_default),
                                         encoding="utf-8")
    elif abductive:
        (output / graph_name).write_text(json.dumps(serializable, ensure_ascii=False, indent=1, default=_default),
                                         encoding="utf-8")
    if result.get("abduction") is not None:
        (output / "abduction.json").write_text(
            json.dumps(result["abduction"], ensure_ascii=False, indent=1, default=_default), encoding="utf-8")
    report = summarize(result, node_count, edge_count)
    (output / report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / report_md).write_text(render_report(result, report), encoding="utf-8")
    return report


def _default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Counter):
        return dict(value)
    raise TypeError(type(value))


def summarize(result, node_count, edge_count):
    by_id = {event["id"]: event for event in result["events"]}
    cross_doc_edges = sum(1 for relation in result["continuity"] + result["expectations"] + result["cross"])
    lengths = Counter(len(chain["edges"]) for chain in result["reasoning_chains"])
    multi_doc_abstract = sum(1 for node in result["abstracts"].values() if len(node["docs"]) >= 2)
    return {
        "timing_seconds": result.get("timing", {}),
        "llm": result.get("llm_stats", {}),
        "stages": result.get("stats", {}),
        "cypher_nodes": node_count,
        "cypher_edges": edge_count,
        "instance_events": len(result["events"]),
        "core_events": sum(1 for event in result["events"] if event["salience"] == "核心"),
        "forecast_events": sum(1 for event in result["events"] if event["status"] == "预测中"),
        "variables": len(result["variables"]),
        "abstract_events": len(result["abstracts"]),
        "abstract_events_in_2plus_docs": multi_doc_abstract,
        "intra_relations": Counter(relation["type"] for relation in result["relations"]),
        "intra_relations_by_source": Counter(relation["source"] + "/" + relation["basis"]
                                             for relation in result["relations"]),
        "cross_doc_edges": cross_doc_edges,
        "continuity": Counter(link["change"] for link in result["continuity"]),
        "expectations": Counter(link["outcome"] for link in result["expectations"]),
        "cross_period": {"candidates": result.get("cross_candidates", 0), "accepted": len(result["cross"])},
        "abstract_edges": len(result["abstract_edges"]),
        "reasoning_chains": len(result["reasoning_chains"]),
        "reasoning_chain_lengths": dict(sorted(lengths.items())),
        "local_chains": len(result["local_chains"]),
        "transmission_chains": len(result["transmission_chains"]),
        "typed_paths": {name: len((result.get("typed_paths") or {}).get(name) or [])
                        for name in ("PURE_CAUSAL", "TEMPORAL", "COMPOSITION", "MIXED_EXPLANATION")},
        "storylines": len(result["storylines"]),
        "events_without_relation": sum(1 for event_id in by_id if event_id not in _touched(result)),
    }


def _touched(result):
    touched = set()
    for group in ("relations", "continuity", "expectations", "cross"):
        for edge in result[group]:
            touched.add(edge["head"])
            touched.add(edge["tail"])
    return touched


def render_report(result, report):
    by_id = {event["id"]: event for event in result["events"]}
    dates = {doc["id"]: doc["report_date"] or "未给出年份" for doc in result["documents"]}
    lines = ["# 事件层运行报告", ""]
    lines.append("## 规模")
    for key in ("instance_events", "core_events", "forecast_events", "variables", "abstract_events",
                "abstract_events_in_2plus_docs", "cross_doc_edges", "abstract_edges", "reasoning_chains",
                "local_chains", "transmission_chains", "storylines", "events_without_relation",
                "cypher_nodes", "cypher_edges"):
        lines.append("- %s: %s" % (key, report[key]))
    lines.append("- 文档内关系: %s" % json.dumps(report["intra_relations"], ensure_ascii=False))
    lines.append("- 关系来源: %s" % json.dumps(report["intra_relations_by_source"], ensure_ascii=False))
    lines.append("- 事态延续: %s" % json.dumps(report["continuity"], ensure_ascii=False))
    lines.append("- 预期验证: %s" % json.dumps(report["expectations"], ensure_ascii=False))
    lines.append("- 跨期传导: %s" % json.dumps(report["cross_period"], ensure_ascii=False))
    lines.append("- 链长分布: %s" % json.dumps(report["reasoning_chain_lengths"], ensure_ascii=False))
    lines.append("- 类型化路径: %s" % json.dumps(report.get("typed_paths") or {}, ensure_ascii=False))
    lines.append("- cross_doc_edges 含连续观测、预测验证和跨文档核验，不是已确认因果条数。")
    lines.append("- 耗时（秒）: %s" % json.dumps(report["timing_seconds"], ensure_ascii=False))
    lines.append("- 模型调用: %s" % json.dumps(report["llm"], ensure_ascii=False))
    stages = report.get("stages", {})
    extract = stages.get("extract", {})
    lines.append("- 窗口抽取: 事件 %s→%s，关系 %s→%s" % (extract.get("raw_events"), extract.get("kept_events"),
                                                  extract.get("raw_relations"), extract.get("kept_relations")))
    lines.append("- 事件被拒原因: %s" % json.dumps(extract.get("dropped_events", {}), ensure_ascii=False))
    lines.append("- 窗口关系被拒原因: %s" % json.dumps(extract.get("dropped_relations", {}), ensure_ascii=False))
    lines.append("- 整篇关系被拒原因: %s" % json.dumps(stages.get("document", {}).get("dropped_relations", {}),
                                                ensure_ascii=False))
    lines.append("- 指标归一: %s" % json.dumps(stages.get("ontology", {}), ensure_ascii=False))
    lines.append("- 合并: %s" % json.dumps(stages.get("merge", {}), ensure_ascii=False))
    lines.append("")
    lines.append("## 跨文档推理链（前 15 条）")
    for chain in result["reasoning_chains"][:15]:
        lines.append("")
        lines.append("### %s（%d 条边，%d 篇文档）" % (chain["id"], len(chain["edges"]), len(chain["docs"])))
        for index, node in enumerate(chain["nodes"]):
            event = by_id[node]
            prefix = "%d. " % (index + 1)
            if index:
                edge = chain["edges"][index - 1]
                prefix += "[%s] " % _edge_label(edge)
            lines.append("%s%s ｜ %s ｜ 原文：%s" % (prefix, event["name"], dates[event["doc_id"]], event["evidence"]))
    lines.append("")
    lines.append("## 抽象传导链（前 15 条）")
    for chain in result["transmission_chains"][:15]:
        names = [abstract_title(node) for node in chain["nodes"]]
        lines.append("- " + _path_text(names, chain["edges"]) + "（支持数 %d，文档 %d 篇）" % (chain["support"], len(chain["docs"])))
    lines.append("")
    lines.append("## 预期验证样例")
    for link in result["expectations"][:20]:
        lines.append("- [%s] %s（%s） → %s（%s）" % (link["outcome"], by_id[link["head"]]["name"], dates[by_id[link["head"]]["doc_id"]],
                                                   by_id[link["tail"]]["name"], dates[by_id[link["tail"]]["doc_id"]]))
    lines.append("")
    return "\n".join(lines) + "\n"


def _edge_label(edge):
    if edge["type"] == "跨期传导":
        return "跨期" + edge["relation"]
    if edge["type"] == "事态延续":
        return "延续·" + edge["change"]
    if edge["type"] == "预期验证":
        return "验证·" + edge["outcome"]
    label = edge["type"] + ("·推断" if edge.get("basis") == "综合推断" else "")
    return label + ("·校正" if edge.get("polarity") else "")
