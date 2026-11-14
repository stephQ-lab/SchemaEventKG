# -*- coding: utf-8 -*-
"""Long chains: instance reasoning chains and abstract transmission chains.

An instance chain is a run of in-document steps (causal, composition, and a
checked cross-document cause). Indicator time series (事态延续) and forecast
checks (预期验证) are not causal steps. They are spliced into a chain only when
the legacy baseline asks for splice_continuity=True. A jump only lands on an
event that has a step of its own. Only a trend that keeps its direction or a
forecast that came true carries a chain forward.
"""
from collections import defaultdict
import hashlib

from eventkg.lexicon import CAUSAL, COMPOSITION
from eventkg.support import verify_relation

BRIDGES = ("事态延续", "预期验证")
STEPS = CAUSAL + COMPOSITION + ("跨期传导",)
MAX_EDGES = 10
MAX_JUMPS = 3
JUMP_DEPTH = 2
EXPANSIONS = 20000
PER_START = 400


CAUSAL_CLASSES = ("ASSERTED_CAUSE", "INFERRED_SUPPORTED")
NONCAUSAL_CLASSES = ("POSSIBLE_CAUSE", "POSSIBLE_INDIRECT_CAUSE", "NEXT_OBSERVATION", "CONTINUES",
                     "REVERSES", "FORECASTS", "SAME_EVENT", "NONE", "LEGACY_CANDIDATE",
                     "GENERIC_RELATION_ONLY", "INSTANCE_AMBIGUOUS")


def allows_causal_walk(edge):
    """Causal questions may follow an asserted or evidence-backed cause, not a timeline or a guess."""
    kind = edge.get("relation_class") or ""
    if kind in NONCAUSAL_CLASSES:
        return False
    if kind in CAUSAL_CLASSES:
        return True
    if edge.get("type") in ("事态延续", "预期验证") or edge.get("type") in NONCAUSAL_CLASSES:
        return False
    if edge.get("type") in COMPOSITION:
        return False
    if edge.get("type") in CAUSAL and edge.get("type") != "必要条件" and edge.get("relation") != "必要条件":
        return True
    if edge.get("type") == "跨期传导" and not kind and edge.get("relation") in CAUSAL:
        return True
    return False


def _is_causal(edge):
    return allows_causal_walk(edge) and edge.get("type") not in COMPOSITION


def ends_chain(edge, head_dir, tail_dir):
    """抑制 is a signed cause and can stay on an explanation path. Composition is not a cause."""
    del edge, head_dir, tail_dir
    return False


def carries(edge):
    if edge.get("period_relation") == "NEXT_AVAILABLE_REPORT":
        return False
    if edge["type"] == "事态延续":
        return edge.get("change") == "延续"
    if edge["type"] == "预期验证":
        return edge.get("outcome") == "兑现"
    return False


def _step_graph(relations, links, splice_continuity=False):
    steps, bridges = defaultdict(list), defaultdict(list)
    for edge in list(relations) + list(links):
        if allows_causal_walk(edge):
            steps[edge["head"]].append(edge)
        elif splice_continuity and edge["type"] in BRIDGES and carries(edge):
            bridges[edge["head"]].append(edge)
    for table in (steps, bridges):
        for edges in table.values():
            edges.sort(key=lambda edge: (not _is_causal(edge), -edge.get("confidence", 0), edge["tail"]))
    return steps, bridges


def _jumps(steps, bridges):
    """event -> [(landing event, bridge edges)] where the landing event has steps of its own."""
    table = {}
    for origin in list(bridges):
        found, frontier, seen = [], [(origin, [])], {origin}
        for _ in range(JUMP_DEPTH):
            following = []
            for node, path in frontier:
                for edge in bridges.get(node, []):
                    if edge["tail"] in seen:
                        continue
                    seen.add(edge["tail"])
                    route = path + [edge]
                    if steps.get(edge["tail"]):
                        found.append((edge["tail"], route))
                    following.append((edge["tail"], route))
            frontier = following
        if found:
            table[origin] = found
    return table


def evidence_catalog(evidence_index):
    if not evidence_index:
        return {}
    if isinstance(evidence_index, dict):
        return evidence_index
    return {item.get("evidence_id"): item for item in evidence_index if item.get("evidence_id")}


def _cited_ids(edge):
    ids = [item for item in (edge.get("evidence_ids") or []) if item]
    for key in ("counter_evidence_ids", "unrelated_evidence_ids"):
        for item in edge.get(key) or []:
            if item and item not in ids:
                ids.append(item)
    return ids


def evidence_state(edge, evidence_index=None, events=None):
    """Use the shared verification record. A changed event identity cannot reuse it."""
    from eventkg.support import instance_fingerprint
    catalog = evidence_catalog(evidence_index)
    ids = _cited_ids(edge)
    relation = edge.get("relation") or edge.get("type") or ""
    support_ids = [item for item in (edge.get("evidence_ids") or []) if item]
    if not support_ids or any(item not in catalog for item in ids):
        return "MISSING"
    table = events if isinstance(events, dict) else {
        item["id"]: item for item in events or [] if isinstance(item, dict) and item.get("id")}
    head = table.get(edge.get("head")) or {}
    tail = table.get(edge.get("tail")) or {}
    identity = instance_fingerprint(head) + ":" + instance_fingerprint(tail)
    version = _cached_version(edge, catalog)
    record = edge.get("verification")
    stale = (
        not record or record.get("relation_evidence_ids") != ids
        or record.get("head_event") != edge.get("head") or record.get("tail_event") != edge.get("tail")
        or record.get("relation") != relation or record.get("identity_fingerprint") != identity
        or record.get("evidence_version") != version
    )
    if stale:
        record = verify_relation(edge, events, catalog)
        edge["verification"] = record
        edge["verification_status"] = record["verification_status"]
    if record["verification_status"] != "VERIFIED":
        return record["verification_status"]
    if (edge.get("relation_class") or "") not in CAUSAL_CLASSES and edge.get("type") not in CAUSAL and edge.get("relation") not in CAUSAL:
        return "UNVERIFIED"
    return "VERIFIED"


def _cached_version(edge, catalog):
    from eventkg.support import _evidence_rows
    version, _sources = _evidence_rows(catalog, _cited_ids(edge))
    return version


def _evidence_points(edge, evidence_index=None, events=None):
    return 1 if evidence_state(edge, evidence_index, events) == "VERIFIED" else 0


def _instance_id(event):
    return event.get("event_instance_id") or event["id"]


def _score(edges, by_id, evidence_index=None):
    """Verified-edge completeness only. Length, document count and bare ids are not rewards."""
    nodes = [edges[0]["head"]] + [edge["tail"] for edge in edges]
    docs = {by_id[node]["doc_id"] for node in nodes}
    causal = sum(_is_causal(edge) for edge in edges)
    evidenced = sum(_evidence_points(edge, evidence_index, by_id) for edge in edges)
    return evidenced / len(edges), docs, causal


def _walk(start, steps, jumps, by_id):
    """Every state whose last edge is a step is a candidate; maximal ones are kept later."""
    found = []
    stack = [(start, [], frozenset([start]), frozenset([_instance_id(by_id[start])]), 0, False)]
    expansions = 0
    while stack and expansions < EXPANSIONS and len(found) < PER_START:
        node, edges, visited, instances, jumps_used, jumped = stack.pop()
        expansions += 1
        if edges and not jumped:
            found.append(edges)
            last = edges[-1]
            if ends_chain(last, by_id[last["head"]]["dir"], by_id[last["tail"]]["dir"]):
                continue
        if len(edges) >= MAX_EDGES:
            continue
        for edge in steps.get(node, []):
            tail = edge["tail"]
            if tail not in by_id or tail in visited or _instance_id(by_id[tail]) in instances:
                continue
            stack.append((tail, edges + [edge], visited | {tail},
                          instances | {_instance_id(by_id[tail])}, jumps_used, False))
        if jumped or jumps_used >= MAX_JUMPS or not edges:
            continue
        for landing, route in jumps.get(node, []):
            if landing in visited or len(edges) + len(route) >= MAX_EDGES:
                continue
            passed = {edge["tail"] for edge in route}
            passed_ids = frozenset(_instance_id(by_id[edge["tail"]]) for edge in route if edge["tail"] in by_id)
            if passed_ids & instances:
                continue
            stack.append((landing, edges + route, visited | passed, instances | passed_ids, jumps_used + 1, True))
    return found


def _maximal(candidates, key_of):
    kept, covered = [], set()
    for item in sorted(candidates, key=lambda item: (-len(item["nodes"]), -item["score"])):
        key = key_of(item)
        if key in covered:
            continue
        kept.append(item)
        for start in range(len(key)):
            for end in range(start + 2, len(key) + 1):
                covered.add(key[start:end])
    return kept


def _diverse(items, edge_key, overlap=0.6, per_start=4):
    """Best-scoring first; skip a chain that mostly repeats edges of an already kept one."""
    kept, kept_sets, starts = [], [], {}
    for item in sorted(items, key=lambda item: -item["score"]):
        edges = {edge_key(edge) for edge in item["edges"]}
        start = item["nodes"][0]
        if starts.get(start, 0) >= per_start:
            continue
        if any(len(edges & other) > overlap * min(len(edges), len(other)) for other in kept_sets):
            continue
        kept.append(item)
        kept_sets.append(edges)
        starts[start] = starts.get(start, 0) + 1
    return kept


def _edge_key(edge):
    return edge["head"], edge["type"], edge["tail"]


def _edge_kind(edge):
    if edge.get("type") in COMPOSITION:
        return "COMPOSITION"
    if edge.get("type") in ("事态延续", "预期验证"):
        return "TEMPORAL"
    if _is_causal(edge) or edge.get("relation") in CAUSAL or edge.get("type") in CAUSAL:
        return "PURE_CAUSAL"
    return "OTHER"


def _path_type(edges):
    kinds = {_edge_kind(edge) for edge in edges}
    if kinds == {"PURE_CAUSAL"}:
        return "PURE_CAUSAL"
    if kinds == {"TEMPORAL"}:
        return "TEMPORAL"
    if kinds == {"COMPOSITION"}:
        return "COMPOSITION"
    return "MIXED_EXPLANATION"


def _temporal_state(edge, events):
    """A time bridge needs its own ordered time basis. Unverified and disputed do not pass."""
    table = events if isinstance(events, dict) else {
        item["id"]: item for item in events or [] if isinstance(item, dict) and item.get("id")}
    relation = edge.get("period_relation") or ""
    if relation == "NEXT_AVAILABLE_REPORT" or edge.get("change") == "反转":
        return "UNVERIFIED" if relation == "NEXT_AVAILABLE_REPORT" else "CONTRADICTED"
    head, tail = table.get(edge.get("head")) or {}, table.get(edge.get("tail")) or {}
    head_when = head.get("event_time") or ""
    tail_when = tail.get("event_time") or ""
    ordered = bool(head_when and tail_when and head_when < tail_when)
    if relation in ("NEXT_PERIOD", "CONTINUES", "NEXT_OBSERVATION") and (ordered or not head_when or not tail_when):
        if head_when and tail_when and not ordered:
            return "UNVERIFIED"
        return "TEMPORAL_VERIFIED"
    if edge.get("type") in ("事态延续", "预期验证") and ordered and edge.get("change") != "反转":
        return "TEMPORAL_VERIFIED"
    return "UNVERIFIED"


def _edge_state(edge, evidence_index, events):
    if _edge_kind(edge) == "TEMPORAL":
        return _temporal_state(edge, events)
    return evidence_state(edge, evidence_index, events)


def _step_supported(kind, state):
    """Forecasts, disputes and unverified time bridges do not confirm a path."""
    if kind == "TEMPORAL":
        return state == "TEMPORAL_VERIFIED"
    return state == "VERIFIED"


def _pack(edges, by_id, evidence_index=None):
    nodes = tuple([edges[0]["head"]] + [edge["tail"] for edge in edges])
    score, docs, causal = _score(edges, by_id, evidence_index)
    states = [_edge_state(edge, evidence_index, by_id) for edge in edges]
    kinds = [_edge_kind(edge) for edge in edges]
    causal_edges = [edge for edge, kind in zip(edges, kinds) if kind == "PURE_CAUSAL"]
    causal_ok = bool(causal_edges) and all(
        state == "VERIFIED" for state, kind in zip(states, kinds) if kind == "PURE_CAUSAL")
    all_ok = all(_step_supported(kind, state) for kind, state in zip(kinds, states))
    path_type = _path_type(edges)
    if path_type == "PURE_CAUSAL":
        confirmed = causal_ok
    elif path_type == "MIXED_EXPLANATION":
        confirmed = causal_ok and all_ok
    elif path_type == "TEMPORAL":
        confirmed = all_ok
    else:
        confirmed = False
    return {"nodes": nodes, "edges": edges, "score": round(score, 3), "docs": sorted(docs),
            "causal_edges": causal, "path_type": path_type, "evidence_status": states,
            "causal_edges_verified": causal_ok, "all_steps_verified": all_ok,
            "path_confirmed": confirmed, "confirmed": confirmed}


def explain_paths(events, relations, links=None, evidence_index=None):
    """Four path lists, each ranked by verified evidence rather than length."""
    links = links or []
    by_id = {event["id"]: event for event in events}
    groups = {name: [] for name in ("PURE_CAUSAL", "TEMPORAL", "COMPOSITION", "MIXED_EXPLANATION")}
    steps, bridges = _step_graph(relations, links, splice_continuity=True)
    composition, temporal = defaultdict(list), defaultdict(list)
    for edge in list(relations) + list(links):
        if edge.get("type") in COMPOSITION:
            composition[edge["head"]].append(edge)
        elif edge.get("type") in ("事态延续", "预期验证"):
            temporal[edge["head"]].append(edge)
    jumps = _jumps(steps, bridges)
    seen = set()

    def collect(table, jumps_table):
        for start in table:
            for edges in _walk(start, table, jumps_table, by_id):
                item = _pack(edges, by_id, evidence_index)
                key = (item["path_type"], item["nodes"])
                if key in seen:
                    continue
                seen.add(key)
                groups[item["path_type"]].append(item)

    collect(steps, jumps)
    collect(composition, {})
    collect(temporal, {})
    for name in groups:
        groups[name].sort(key=lambda item: (-item["score"], len(item["nodes"]), item["nodes"]))
    return groups


def _chain_bucket(edges, by_id, evidence_index):
    states = [_edge_state(edge, evidence_index, by_id) for edge in edges]
    if any(state in ("CONTRADICTED", "DISPUTED") for state in states):
        return "rejected", states
    if states and all(state in ("VERIFIED", "TEMPORAL_VERIFIED") for state in states):
        return "confirmed", states
    return "candidate", states


def reasoning_chains(events, relations, links, top=200, per_doc=8, splice_continuity=False, evidence_index=None,
                     classified=None):
    """Confirmed chains only. A continuous path without verified steps stays a candidate."""
    by_id = {event["id"]: event for event in events}
    steps, bridges = _step_graph(relations, links, splice_continuity=splice_continuity)
    jumps = _jumps(steps, bridges)
    stepped_into = {edge["tail"] for edges in steps.values() for edge in edges}
    starts = [node for node in steps if node not in stepped_into]
    starts += [node for node in steps if node in stepped_into and by_id[node]["salience"] == "核心"]
    cross, local, seen = [], [], set()
    buckets = {"confirmed": [], "candidate": [], "rejected": []}
    for start in starts:
        for edges in _walk(start, steps, jumps, by_id):
            nodes = tuple([edges[0]["head"]] + [edge["tail"] for edge in edges])
            if nodes in seen:
                continue
            seen.add(nodes)
            score, docs, causal = _score(edges, by_id, evidence_index)
            bucket, states = _chain_bucket(edges, by_id, evidence_index)
            item = {"nodes": nodes, "edges": edges, "score": round(score, 3), "docs": sorted(docs),
                    "causal_edges": causal, "path_type": _path_type(edges), "chain_status": bucket,
                    "evidence_status": states}
            buckets[bucket].append(item)
            if bucket != "confirmed":
                continue
            if len(docs) >= 2 and causal >= 2 and len(edges) >= 3:
                cross.append(item)
            elif len(docs) == 1 and causal >= 2 and len(edges) >= 3:
                sections = {section for node in nodes for section in by_id[node]["sections"][:1]}
                if len(sections) >= 2:
                    local.append(item)
    if classified is not None:
        classified.update(buckets)
    cross = _diverse(_maximal(cross, lambda item: item["nodes"]), _edge_key)[:top]
    local_kept = []
    per = defaultdict(int)
    for item in _diverse(_maximal(local, lambda item: item["nodes"]), _edge_key):
        doc = item["docs"][0]
        if per[doc] < per_doc:
            per[doc] += 1
            local_kept.append(item)
    for index, item in enumerate(cross, 1):
        item["id"] = "chain:x%d" % index
        item["kind"] = "跨文档推理链"
    for index, item in enumerate(local_kept, 1):
        item["id"] = "chain:d%d" % index
        item["kind"] = "文档内推理链"
    return cross, local_kept


def transmission_chains(abstract_edges, top=150, max_edges=6):
    graph = defaultdict(list)
    for edge in abstract_edges:
        graph[edge["head"]].append(edge)
    for edges in graph.values():
        edges.sort(key=lambda edge: (-edge["support"], edge["tail"]))
    incoming = {edge["tail"] for edge in abstract_edges}
    starts = [node for node in graph if node not in incoming] or list(graph)
    candidates, seen = [], set()
    for start in starts:
        stack = [(start, [], {start})]
        found = 0
        while stack and found < 60:
            node, edges, visited = stack.pop()
            extended = False
            last = edges[-1] if edges else None
            stopped = last and ends_chain(last, last["head"].rsplit("|", 1)[1], last["tail"].rsplit("|", 1)[1])
            for edge in [] if stopped else graph.get(node, [])[:6]:
                variable = edge["tail"].rsplit("|", 1)[0]
                if variable in {key.rsplit("|", 1)[0] for key in visited} or len(edges) >= max_edges:
                    continue
                extended = True
                stack.append((edge["tail"], edges + [edge], visited | {edge["tail"]}))
            if not extended and len(edges) >= 2 and any(edge["type"] in CAUSAL for edge in edges):
                nodes = tuple([edges[0]["head"]] + [edge["tail"] for edge in edges])
                if nodes in seen:
                    continue
                seen.add(nodes)
                found += 1
                docs = {doc for edge in edges for doc in edge["docs"]}
                support = sum(edge["support"] for edge in edges)
                candidates.append({"nodes": nodes, "edges": edges, "docs": sorted(docs),
                                   "support": support, "score": round(support, 3),
                                   "path_type": _path_type(edges)})
    kept = _diverse(_maximal(candidates, lambda item: item["nodes"]), _edge_key)[:top]
    for index, item in enumerate(kept, 1):
        item["id"] = "chain:a%d" % index
        item["kind"] = "传导链"
    return kept
