import hashlib
import json
import re
from pathlib import Path


def exact_span(name, text):
    chars = [(char, index) for index, char in enumerate(text) if not char.isspace()]
    probe = "".join(name.split())
    if not probe:
        return None
    compact = "".join(char for char, _ in chars)
    positions = [match.start() for match in re.finditer(re.escape(probe), compact)]
    if len(positions) != 1:
        return None
    at = positions[0]
    return chars[at][1], chars[at + len(probe) - 1][1] + 1


def load_source_index(path):
    if not path or not Path(path).is_file():
        return {}
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text_id = hashlib.sha256((row["text"] + "_text").encode("utf-8")).hexdigest()
            key = (row["document_id"], text_id)
            if key in result:
                previous = result[key]
                if previous["metadata"]["text_sha256"] != row["metadata"]["text_sha256"]:
                    raise ValueError("同一文档出现不同文本版本：" + str(key))
                known = {(unit["block_id"], unit["start"], unit["end"]) for unit in previous["units"]}
                previous["units"].extend(unit for unit in row["units"]
                                         if (unit["block_id"], unit["start"], unit["end"]) not in known)
            else:
                result[key] = row
    return result


def add_structured_evidence(cypher, source_index, passage_ids, by_key):
    supported = 0
    unsupported = []
    for (name, scope), ntype in by_key.items():
        if ntype != "event":
            continue
        evidence = []
        for text_id in dict.fromkeys(passage_ids.get((name, scope), [])):
            source = source_index.get((scope, text_id))
            if source is None:
                continue
            meta = source["metadata"]
            for unit in source["units"]:
                span = exact_span(name, unit["text"])
                if span is None:
                    continue
                block_id = unit["block_id"]
                evidence_name = "%s:%s:%s" % (block_id, unit.get("row", "text"), unit["start"])
                extra = {"来源文件": meta["source_file"], "页码": unit.get("page"),
                         "坐标说明": unit.get("coordinate_system"),
                         "章节": unit.get("section"), "区域坐标": json.dumps(unit.get("bbox"), ensure_ascii=False),
                         "文本SHA256": meta["text_sha256"], "文档起始": unit["start"], "文档结束": unit["end"],
                         "定位方式": "事件全文唯一匹配", "内容类型": unit["kind"], "表格行号": unit.get("row")}
                cypher.add_node(evidence_name, "evidence", scope, unit["text"], extra=extra)
                cypher.add_edge(name, "event", scope, "证据", evidence_name, "evidence", scope)
                cypher.add_edge(evidence_name, "evidence", scope, "摘自", text_id, "passage", text_id)
                evidence.append({"起始": unit["start"] + span[0], "结束": unit["start"] + span[1],
                                 "证据": unit["text"], **extra})
        if evidence:
            supported += 1
            if len(evidence) == 1:
                cypher.add_node(name, "event", scope, extra=evidence[0])
            else:
                cypher.add_node(name, "event", scope, extra={"定位方式": "多处证据，请查看证据关系"})
        else:
            unsupported.append({"event": name, "document_id": scope,
                                "reason": "未发现唯一的事件全文匹配，请沿提及关系核对原文"})
    return {"events_with_full_text_match": supported, "unlocated_events": unsupported}
