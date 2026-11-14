# -*- coding: utf-8 -*-
"""Batched, idempotent Cypher: one statement per line, MERGE on uid only.

load_cypher reads one statement per line, so every statement here stays on a
single line. Nodes are merged by uid and their properties set afterwards, so a
second run (or the other half of the pipeline) updates instead of duplicating.
"""
import json
import os

BATCH = 200


def literal(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, (list, tuple)):
        items = [item for item in value if item is not None and item != ""]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            return "[" + ", ".join(json.dumps(item) for item in items) + "]"
        return "[" + ", ".join(literal(str(item)) for item in items) + "]"
    if isinstance(value, dict):
        return literal(json.dumps(value, ensure_ascii=False))
    text = str(value)
    return "'" + (text.replace("\\", "\\\\").replace("'", "\\'").replace("\r", "")
                  .replace("\n", "\\n")) + "'"


def quote_name(name):
    return "`" + str(name).replace("`", "") + "`"


def props_map(props):
    parts = []
    for key, value in props.items():
        if value is None or value == "" or value == []:
            continue
        parts.append("%s: %s" % (quote_name(key), literal(value)))
    return "{" + ", ".join(parts) + "}"


def write_cypher(output_file, nodes, edges, header=()):
    """nodes: {uid: {"labels": [primary, ...], "props": {...}}}
    edges: [(head_uid, relation, tail_uid, props)]"""
    folder = os.path.dirname(output_file)
    if folder:
        os.makedirs(folder, exist_ok=True)
    lines = ["// " + line for line in header]
    primaries = sorted({node["labels"][0] for node in nodes.values()} |
                       {label for node in nodes.values() for label in node["labels"]})
    for label in primaries:
        lines.append("CREATE INDEX IF NOT EXISTS FOR (n:%s) ON (n.uid);" % quote_name(label))
    groups = {}
    for uid, node in nodes.items():
        groups.setdefault(tuple(node["labels"]), []).append((uid, node["props"]))
    for labels, rows in sorted(groups.items()):
        primary = quote_name(labels[0])
        extra = "".join(":" + quote_name(label) for label in labels[1:])
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            payload = ", ".join("{uid: %s, props: %s}" % (literal(uid), props_map(props)) for uid, props in chunk)
            statement = "UNWIND [%s] AS row MERGE (n:%s {uid: row.uid}) SET n += row.props" % (payload, primary)
            if extra:
                statement += " SET n%s" % extra
            lines.append(statement + ";")
    edge_groups = {}
    for head, relation, tail, props in edges:
        if head not in nodes or tail not in nodes:
            continue
        key = (nodes[head]["labels"][0], relation, nodes[tail]["labels"][0])
        edge_groups.setdefault(key, []).append((head, tail, props or {}))
    for (head_label, relation, tail_label), rows in sorted(edge_groups.items()):
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            payload = ", ".join("{a: %s, b: %s, props: %s}" % (literal(head), literal(tail), props_map(props))
                                for head, tail, props in chunk)
            lines.append(
                "UNWIND [%s] AS row MATCH (a:%s {uid: row.a}) MATCH (b:%s {uid: row.b}) "
                "MERGE (a)-[r:%s]->(b) SET r += row.props;"
                % (payload, quote_name(head_label), quote_name(tail_label), quote_name(relation)))
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(nodes), sum(len(rows) for rows in edge_groups.values())
