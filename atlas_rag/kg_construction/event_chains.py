import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import json_repair

from atlas_rag.kg_construction.commodity import numbers
from atlas_rag.kg_construction.event_ontology import event_status
from atlas_rag.kg_construction.neo4j_export import _escape, _uid


RELATIONS = {"无关系", "同一事实", "共同因素", "可能影响"}
PROMPT = """你在核对两篇研报中的事件。每对事件只根据给定的两句原文判断。
返回 JSON 数组，每个输入编号必须返回一条：
[{"pair":"P0","relation":"无关系|同一事实|共同因素|可能影响","head":"A|B","factor":"原文中两端都出现的原样词组或空串","reason":"判断依据"}]
同一事实：主体、指标、地点、时间、数值、趋势兼容；跨文章重复报道也要标出。
共同因素：两句原文明确提到同一个具体因素；factor 必须从两句各自原样截取。
可能影响：A 事件可以经明确写出的机制影响 B 事件；head 取影响的起点。该关系仅为待核对的推断，不能当作已证实的因果。
无关系：主题相近、同一品类、相邻句子、共现或只凭常识猜测均属于无关系。
未来预测与已发生事实不等同；不同日期、地区、规格、数值不等同。禁止增添原文没有的主体、数字或事实。
reason 写出具体机制或同一事实的对应内容，不能重复关系名称。不要输出其他内容。"""


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _digest(paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                digest.update(block)
    return digest.hexdigest()


def _grams(text):
    cleaned = re.sub(r"[\W\d_]+", "", text or "", flags=re.UNICODE)
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def _sentence(text, event):
    query = _grams(event)
    values = numbers(event)
    best = (0.0, "", "")
    for sentence in re.split(r"[。！？；\n]", text):
        sentence = sentence.strip()
        if not 6 <= len(sentence) <= 500:
            continue
        if not values.issubset(numbers(sentence)):
            continue
        overlap = len(query & _grams(sentence))
        score = overlap / max(1, len(query))
        if overlap >= 2 and score > best[0]:
            start = text.find(sentence)
            context = text[max(0, start - 150):start + len(sentence) + 150] if start >= 0 else sentence
            best = (score, sentence, context)
    return best if best[0] >= 0.28 else (0.0, "", "")


def _events(root, pattern):
    base = root / "triples_csv"
    node_path = base / f"triple_nodes_{pattern}_from_json_without_emb.csv"
    edge_path = base / f"triple_edges_{pattern}_from_json_without_emb.csv"
    text_path = base / f"text_nodes_{pattern}_from_json.csv"
    source_path = base / f"text_edges_{pattern}_from_json.csv"
    paths = (node_path, edge_path, text_path, source_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    texts = {row["text_id:ID"]: row["original_text"] for row in _read_csv(text_path)}
    sources = defaultdict(list)
    for row in _read_csv(source_path):
        sources[(row[":START_ID"], row.get("scope", ""))].append(row[":END_ID"])
    commodities = defaultdict(set)
    titles = {}
    documents = root / "kg_extraction" / "documents.jsonl"
    if documents.is_file():
        with documents.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                titles[row["id"]] = row.get("title") or ""
    existing = set()
    for row in _read_csv(edge_path):
        scope = row.get("scope", "")
        if row["relation"] == "涉及":
            commodities[(row[":START_ID"], scope)].add(row[":END_ID"])
        existing.add((row[":START_ID"], scope, row[":END_ID"], scope))
    result = []
    for row in _read_csv(node_path):
        if row["type"] != "event":
            continue
        name, scope = row["name:ID"], row.get("scope", "")
        if not scope:
            raise ValueError("事件缺少文档编号；请使用 V5 文档范围输出")
        candidates = [_sentence(texts.get(text_id, ""), name)
                      for text_id in sources.get((name, scope), ())]
        matched = max(candidates, default=(0.0, "", ""))
        evidence = matched[1]
        if evidence:
            result.append({"name": name, "scope": scope, "evidence": evidence,
                           "context": matched[2], "title": titles.get(scope, ""),
                           "commodities": sorted(commodities[(name, scope)]),
                           "uid": _uid("事件", scope, name)})
    return result, existing, paths


def _ranked_candidates(events, existing, intra_limit, cross_limit):
    grams = [_grams(row["name"] + " " + row["evidence"][:180]) for row in events]
    by_doc = defaultdict(list)
    postings = defaultdict(list)
    for index, row in enumerate(events):
        by_doc[row["scope"]].append(index)
        for commodity in row["commodities"]:
            for gram in grams[index]:
                postings[(commodity, gram)].append(index)
    selected = {}
    positions = {}
    for indexes in by_doc.values():
        positions.update({index: at for at, index in enumerate(indexes)})
    for index, row in enumerate(events):
        local = by_doc[row["scope"]]
        at = positions[index]
        near = set(local[max(0, at - 5):at] + local[at + 1:at + 6])
        within = []
        for other in local:
            if other == index:
                continue
            if events[other]["name"] == row["name"]:
                continue
            overlap = len(grams[index] & grams[other])
            within.append((overlap, other in near, other))
        within.sort(reverse=True)
        matches = Counter()
        for commodity in row["commodities"]:
            for gram in grams[index]:
                found = postings[(commodity, gram)]
                if len(found) > 100:
                    continue
                weight = 1 / math.log2(len(found) + 2)
                for other in found:
                    if events[other]["scope"] != row["scope"]:
                        matches[other] += weight
        cross = []
        for other, score in matches.items():
            overlap = len(grams[index] & grams[other])
            if overlap < 3 or overlap / max(1, min(len(grams[index]), len(grams[other]))) < 0.16:
                continue
            cross.append((score, other))
        cross.sort(reverse=True)
        choices = []
        close = next((other for overlap, nearby, other in within if nearby and overlap >= 2), None)
        distant = next((other for overlap, nearby, other in within if not nearby and overlap >= 3), None)
        for other in (close, distant):
            if other is not None and other not in choices:
                choices.append(other)
        for overlap, _, other in within:
            if len(choices) >= intra_limit:
                break
            if overlap >= 2 and other not in choices:
                choices.append(other)
        choices = choices[:intra_limit]
        choices += [other for _, other in cross[:cross_limit]]
        for other in choices:
            a, b = sorted((index, other))
            left, right = events[a], events[b]
            if ((left["name"], left["scope"], right["name"], right["scope"]) in existing or
                    (right["name"], right["scope"], left["name"], left["scope"]) in existing):
                continue
            key = hashlib.sha256((left["uid"] + right["uid"]).encode("ascii")).hexdigest()[:20]
            selected[key] = {"id": key, "a": left, "b": right}
    return list(selected.values())


def _ask(client, model, batch, max_tokens):
    items = [{"pair": f"P{index}", "A": {"event": row["a"]["name"],
              "document": row["a"]["scope"], "title": row["a"]["title"],
              "evidence": row["a"]["evidence"], "context": row["a"]["context"]},
              "B": {"event": row["b"]["name"], "document": row["b"]["scope"],
                     "title": row["b"]["title"], "evidence": row["b"]["evidence"],
                     "context": row["b"]["context"]}}
             for index, row in enumerate(batch)]
    answer = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
    )
    if answer.choices[0].finish_reason == "length":
        raise ValueError("事件关系输出达到 token 上限；请增加 --max-tokens")
    rows = json_repair.loads(answer.choices[0].message.content or "")
    if not isinstance(rows, list) or len(rows) != len(batch):
        raise ValueError("事件关系返回数量不匹配")
    by_id = {item["pair"]: item for item in rows}
    if set(by_id) != {f"P{index}" for index in range(len(batch))}:
        raise ValueError("事件关系编号不匹配")
    decisions = []
    for index, pair in enumerate(batch):
        item = by_id[f"P{index}"]
        relation = item["relation"]
        if relation not in RELATIONS or item.get("head") not in ("A", "B"):
            raise ValueError("事件关系类型或方向无效：" + str(item))
        factor = item.get("factor") or ""
        reason = item.get("reason") or ""
        if not isinstance(factor, str) or not isinstance(reason, str):
            raise ValueError("事件关系依据或共同因素类型无效")
        if relation != "无关系" and not reason.strip():
            raise ValueError("事件关系缺少判断依据")
        if relation == "共同因素" and (factor in {"价格", "库存", "需求", "供应", "市场", "利润"} or
                                  len(factor) < 2 or
                                  factor not in pair["a"]["evidence"] or
                                  factor not in pair["b"]["evidence"]):
            relation = "无关系"
        if relation == "同一事实" and (
            numbers(pair["a"]["name"]) != numbers(pair["b"]["name"]) or
            event_status(pair["a"]["name"]) != event_status(pair["b"]["name"])
        ):
            relation = "无关系"
        decisions.append({"id": pair["id"], "relation": relation,
                          "head": item["head"], "factor": factor, "reason": reason})
    return decisions


def _write_graph(base, output, pairs, decisions):
    with base.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            target.write(line)
        target.write("\n// 文档事件关系：原文证据和模型判定见 event_chain_decisions.jsonl\n")
        for pair in pairs:
            decision = decisions[pair["id"]]
            relation = decision["relation"]
            if relation == "无关系":
                continue
            a, b = pair["a"], pair["b"]
            if relation == "可能影响" and decision["head"] == "B":
                a, b = b, a
            properties = {"依据": decision["reason"], "证据A": pair["a"]["evidence"],
                          "证据B": pair["b"]["evidence"], "文档A": pair["a"]["scope"],
                          "文档B": pair["b"]["scope"], "判定": "待复核" if relation == "可能影响" else "模型判定"}
            if relation == "共同因素":
                properties["共同因素"] = decision["factor"]
            values = ", ".join(f"{key}: '{_escape(value)}'" for key, value in properties.items())
            target.write(f"MATCH (a {{uid: '{a['uid']}'}}), (b {{uid: '{b['uid']}'}}) "
                         f"MERGE (a)-[r:`{relation}`]->(b) SET r += {{{values}}};\n")


def _path_examples(root, pattern, events, pairs, decisions):
    known = {(row["name"], row["scope"]): row for row in events}
    by_uid = {row["uid"]: row for row in events}
    adjacency = defaultdict(list)
    edge_path = root / "triples_csv" / f"triple_edges_{pattern}_from_json_without_emb.csv"
    for row in _read_csv(edge_path):
        if row["relation"] != "导致":
            continue
        scope = row.get("scope", "")
        head = known.get((row[":START_ID"], scope))
        tail = known.get((row[":END_ID"], scope))
        if head and tail and head["uid"] != tail["uid"]:
            adjacency[head["uid"]].append((tail["uid"], "导致"))
    cross_starts = set()
    for pair in pairs:
        decision = decisions[pair["id"]]
        relation = decision["relation"]
        a, b = pair["a"], pair["b"]
        if relation == "可能影响":
            if decision["head"] == "B":
                a, b = b, a
            adjacency[a["uid"]].append((b["uid"], relation))
            if a["scope"] != b["scope"]:
                cross_starts.add(a["uid"])
        elif relation == "同一事实":
            adjacency[a["uid"]].append((b["uid"], relation))
            adjacency[b["uid"]].append((a["uid"], relation))
            if a["scope"] != b["scope"]:
                cross_starts.update((a["uid"], b["uid"]))
    found = []
    for start in sorted(cross_starts)[:200]:
        frontier = [([start], [])]
        examined = 0
        while frontier and examined < 500:
            nodes, relations = frontier.pop()
            examined += 1
            if len(relations) >= 2 and len({by_uid[node]["scope"] for node in nodes}) >= 2:
                if any(relation == "可能影响" for relation in relations):
                    found.append({"events": [{"name": by_uid[node]["name"],
                                               "document": by_uid[node]["scope"],
                                               "evidence": by_uid[node]["evidence"]} for node in nodes],
                                  "relations": relations})
            if len(relations) == 5:
                continue
            for tail, relation in adjacency[nodes[-1]][:15]:
                if tail not in nodes:
                    frontier.append((nodes + [tail], relations + [relation]))
        if len(found) >= 100:
            break
    found.sort(key=lambda row: (-len(row["relations"]), row["events"][0]["document"]))
    return found[:5]


def build_event_chains(root, pattern, client, model, intra_limit=2, cross_limit=1,
                       batch_size=6, max_tokens=3072):
    root = Path(root)
    base = root / "neo4j" / f"{pattern}.cypher"
    if not base.is_file():
        raise FileNotFoundError(base)
    if min(intra_limit, cross_limit, batch_size, max_tokens) < 1:
        raise ValueError("事件关系参数必须为正数")
    events, existing, paths = _events(root, pattern)
    pairs = _ranked_candidates(events, existing, intra_limit, cross_limit)
    settings = {"model": model, "intra_limit": intra_limit,
                "cross_limit": cross_limit, "batch_size": batch_size, "max_tokens": max_tokens}
    fingerprint = _digest((*paths, base), settings)
    folder = root / "neo4j"
    manifest = folder / "event_chain_manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        if stored != {"fingerprint": fingerprint}:
            raise ValueError("事件关系输入已改变，请指定新的 --output 目录")
    else:
        manifest.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    candidates = folder / "event_chain_candidates.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    decision_path = folder / "event_chain_decisions.jsonl"
    decisions = {}
    if decision_path.exists():
        with decision_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                decisions[row["id"]] = row
    remaining = [pair for pair in pairs if pair["id"] not in decisions]
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        answers = _ask(client, model, batch, max_tokens)
        with decision_path.open("a", encoding="utf-8") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                decisions[row["id"]] = row
            handle.flush()
        print("事件关系已判定", len(decisions), "/", len(pairs), flush=True)
    output = folder / f"{pattern}_chains.cypher"
    _write_graph(base, output, pairs, decisions)
    counts = Counter(row["relation"] for row in decisions.values())
    report = {"events_with_evidence": len(events), "candidate_pairs": len(pairs),
              "within_document": sum(pair["a"]["scope"] == pair["b"]["scope"] for pair in pairs),
              "cross_document": sum(pair["a"]["scope"] != pair["b"]["scope"] for pair in pairs),
              "decisions": dict(counts), "path_examples": _path_examples(root, pattern, events, pairs, decisions),
              "output": str(output)}
    (folder / "event_chain_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
