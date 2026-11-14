# -*- coding: utf-8 -*-
"""Merge, cross-document links and the abstract (variable + direction) layer."""
import json
import re
from collections import Counter, defaultdict
from datetime import date

from atlas_rag.kg_construction.commodity import block_reason, outline_core
from eventkg.lexicon import CAUSAL, COMPOSITION, MOVES, SIGN_PRIORS, compact, numbers
from eventkg.identity import can_merge_event_instances, coreferent, same_instance, stamp_instance
from eventkg.observations import parse_table_measures
from eventkg.roles import assign_roles
from eventkg.ontology import _alias_indicator
from eventkg.validate import explicit_date, time_bucket, trend_name
from eventkg.llm import parallel_map
from eventkg.prompts import CROSS_SYSTEM

MAX_GAP_DAYS = 42
CROSS_BATCH = 12
CROSS_LIMIT = 600
OPPOSITE = {("上升", "下降"), ("下降", "上升")}


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _prior_sign(head, tail):
    head_indicator, tail_indicator = head["var"].split("|")[1], tail["var"].split("|")[1]
    tail_commodity = tail["var"].split("|")[0]
    for head_pattern, tail_pattern, excluded, sign in SIGN_PRIORS:
        if (re.search(head_pattern, head_indicator) and re.search(tail_pattern, tail_indicator)
                and tail_commodity != excluded):
            return sign
    return 0


def align_polarity(events, relations, rewrite=False):
    """Legacy relabels a factor that moves against the known sign.

    Abductive passes rewrite=False. The original relation and its evidence stay,
    and the row is only marked POLARITY_CONFLICT for review.
    """
    by_id = {event["id"]: event for event in events}
    changed = 0
    for relation in relations:
        field = "relation" if relation["type"] == "跨期传导" else "type"
        kind = relation.get(field)
        if kind not in ("导致", "促进", "抑制"):
            continue
        head, tail = by_id[relation["head"]], by_id[relation["tail"]]
        prior = _prior_sign(head, tail)
        moved = MOVES.get(head["dir"], 0) * MOVES.get(tail["dir"], 0)
        if not prior or not moved:
            continue
        claimed = -moved if kind == "抑制" else moved
        if claimed == prior:
            continue
        if not rewrite:
            relation["polarity"] = "POLARITY_CONFLICT"
            relation["polarity_prior"] = "方向先验与原文关系不一致，保留原文，交由复核"
            changed += 1
            continue
        relation["original_type"] = kind
        relation[field] = "促进" if kind == "抑制" else "抑制"
        relation["polarity"] = "方向先验校正"
        relation["confidence"] = min(relation.get("confidence", 0), 0.6)
        changed += 1
    return changed


def _merge_outline(events, finder):
    """A contents line or chapter title that restates a longer sentence is the same event."""
    by_doc = defaultdict(list)
    for event in events:
        by_doc[event["doc_id"]].append(event)
    merged = 0
    for group in by_doc.values():
        ordered = sorted(group, key=lambda event: len(compact(event.get("evidence") or "")))
        for index, short in enumerate(ordered):
            core = outline_core(short.get("evidence") or short.get("name") or "")
            if len(core) < 6:
                continue
            for longer in ordered[index + 1:]:
                if finder.find(short["id"]) == finder.find(longer["id"]):
                    continue
                host = compact(longer.get("evidence") or "")
                if core not in host or block_reason(short.get("evidence") or "", longer.get("evidence") or ""):
                    continue
                if not can_merge_event_instances(short, longer, "same_document"):
                    continue
                finder.union(short["id"], longer["id"])
                merged += 1
                break
    return merged


def _specificity(event):
    return (event["salience"] == "核心", len(numbers(event["name"])), bool(event["change"]),
            len(event["evidence"]), event["id"])


def merge(events, relations, duplicate_groups):
    """Merge mentions of one instance. A later week is not the same instance as an earlier week."""
    by_id = {event["id"]: event for event in events}
    finder = UnionFind(by_id)
    why = {}
    ordered = sorted(by_id)
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1:]:
            left, right = by_id[left_id], by_id[right_id]
            if finder.find(left_id) == finder.find(right_id) or left.get("dir") == "不明":
                continue
            if same_instance(left, right):
                finder.union(left_id, right_id)
                why[frozenset((left_id, right_id))] = "same_instance"
    outline_merges = _merge_outline(events, finder)
    llm_merges = 0
    for group in duplicate_groups:
        first = by_id.get(group[0])
        for other_id in group[1:]:
            other = by_id.get(other_id)
            if not first or not other or not same_instance(first, other):
                continue
            finder.union(first["id"], other_id)
            why[frozenset((first["id"], other_id))] = "same_instance"
            llm_merges += 1
    cross_links = 0
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1:]:
            if finder.find(left_id) == finder.find(right_id):
                continue
            if coreferent(by_id[left_id], by_id[right_id]):
                finder.union(left_id, right_id)
                why[frozenset((left_id, right_id))] = "coreference"
                cross_links += 1
    groups = defaultdict(list)
    for event in events:
        groups[finder.find(event["id"])].append(event)
    merged, alias = [], {}
    for members in groups.values():
        chosen = max(members, key=_specificity)
        item = dict(chosen)
        item["members"] = sorted(member["id"] for member in members)
        item["evidences"] = _unique(evidence for member in members for evidence in member["evidences"])
        item["sections"] = _unique(section for member in members for section in member["sections"] if section)
        item["names"] = _unique(member["name"] for member in members)
        signals, seen = [], set()
        for member in members:
            for signal in member.get("trend_signals") or []:
                key = (signal.get("comparison_basis") or "", signal.get("level_direction") or "")
                if key[1] and key not in seen:
                    seen.add(key)
                    signals.append(signal)
            basis = member.get("comparison_basis") or ""
            if member.get("dir") in ("上升", "下降", "持平"):
                for part in basis.split("；"):
                    key = (part, member["dir"])
                    if part and key not in seen:
                        seen.add(key)
                        signals.append({"comparison_basis": part, "level_direction": member["dir"],
                                        "momentum": member.get("momentum") or ""})
        if signals:
            item["trend_signals"] = signals
            bases = []
            for signal in signals:
                if signal["comparison_basis"] and signal["comparison_basis"] not in bases:
                    bases.append(signal["comparison_basis"])
            item["comparison_basis"] = "；".join(bases)
            mom = next((signal for signal in signals if signal["comparison_basis"] == "环比"), None)
            if mom and any(signal["level_direction"] != mom["level_direction"] for signal in signals):
                item["dir"] = mom["level_direction"]
        if item["dir"] == "不明":
            known = [member["dir"] for member in members if member["dir"] != "不明"]
            item["dir"] = known[0] if known else "不明"
        for field in ("value", "change", "subject", "scope"):
            if not item[field]:
                item[field] = next((member[field] for member in members if member[field]), "")
        if any(member["salience"] == "核心" for member in members):
            item["salience"] = "核心"
        item["abstract"] = item["var"] + "|" + item["dir"]
        item["name"] = trend_name(item) or item["name"]
        method = "unlinked"
        if len(members) > 1:
            labels = [why.get(frozenset((left["id"], right["id"])))
                      for left in members for right in members]
            if "coreference" in labels:
                method = "coreference"
            elif "same_instance" in labels:
                method = "same_instance"
            else:
                method = "outline"
        stamp_instance(item, members, method)
        assign_roles(item)
        for member in members:
            alias[member["id"]] = item["id"]
        merged.append(item)
    best = {}
    var_of = {item["id"]: item["var"] for item in merged}
    same_var = 0
    for relation in relations:
        head, tail = alias.get(relation["head"]), alias.get(relation["tail"])
        if not head or not tail or head == tail:
            continue
        if var_of[head] == var_of[tail]:
            same_var += 1
            continue
        kind = relation["type"]
        family = kind if kind in CAUSAL or kind in COMPOSITION else "时序"
        key = (head, tail, family)
        current = dict(relation, head=head, tail=tail)
        if key not in best or current["confidence"] > best[key]["confidence"]:
            if key in best:
                current["evidences"] = _unique(best[key].get("evidences", [best[key]["evidence"]]) + [current["evidence"]])
            best[key] = current
        else:
            best[key]["evidences"] = _unique(best[key].get("evidences", [best[key]["evidence"]]) + [current["evidence"]])
    kept = []
    for key, relation in best.items():
        reverse = (key[1], key[0], key[2])
        if key[2] in ("导致", "抑制", "因果") and reverse in best and best[reverse]["confidence"] > relation["confidence"]:
            continue
        relation.setdefault("evidences", [relation["evidence"]])
        kept.append(relation)
    by_pair = {}
    for relation in kept:
        by_pair.setdefault((relation["head"], relation["tail"]), []).append(relation)
    for group in by_pair.values():
        kinds = {item["type"] for item in group}
        if "导致" in kinds and "抑制" in kinds:
            for item in group:
                item["conflict_claims"] = [
                    {"type": other["type"], "evidences": other.get("evidences") or [other.get("evidence")]}
                    for other in group if other is not item]
    merged.sort(key=lambda item: item["id"])
    kept.sort(key=lambda item: (item["head"], item["tail"], item["type"]))
    stats = {"events_before": len(events), "events_after": len(merged), "llm_merges": llm_merges,
             "outline_merges": outline_merges, "cross_document_links": cross_links,
             "relations_after": len(kept), "same_variable_relations_dropped": same_var}
    return merged, kept, stats


def _unique(values):
    seen, result = set(), []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _parse_date(text):
    if not text:
        return None
    year, month, day = (int(part) for part in text.split("-"))
    return date(year, month, day)


def _series_key(document):
    from eventkg.graph_induction import report_series_id
    return report_series_id(document or {})


def _occurrences(events, documents, status=None):
    """Group by report series and variable. An unknown series is not a series."""
    dates = {doc["id"]: _parse_date(doc["report_date"]) for doc in documents}
    series = {doc["id"]: _series_key(doc) for doc in documents}
    table = defaultdict(lambda: defaultdict(list))
    for event in events:
        when = dates.get(event["doc_id"])
        if when is None or (status and event["status"] != status):
            continue
        party = (compact(event.get("subject") or ""), compact(event.get("scope") or ""))
        table[(series.get(event["doc_id"]) or "", event["var"], party)][(when, event["doc_id"])].append(event)
    return table


def _representative(events):
    """A same-day 环比 reading represents the series. A 同比 reading does not replace it."""
    cyclic = [event for event in events if "环比" in (event.get("comparison_basis") or "")]
    pool = cyclic or events
    moved = [event for event in pool if event["dir"] != "不明"] or pool
    return max(moved, key=_specificity)


def _transition(before, after):
    if before == after:
        return "延续"
    if (before, after) in OPPOSITE:
        return "反转"
    if before == "中断" and after in ("恢复", "上升"):
        return "恢复"
    if after == "中断":
        return "中断"
    return "转变"


def continuity(events, documents):
    """事态延续: the same variable reported by consecutive dated reports."""
    links = []
    for (series, var, _party), slots in _occurrences(events, documents, status="已发生").items():
        if not series:
            continue
        ordered = sorted(slots)
        for (left_date, left_doc), (right_date, right_doc) in zip(ordered, ordered[1:]):
            gap = (right_date - left_date).days
            if gap <= 0 or gap > MAX_GAP_DAYS:
                continue
            period_relation = "NEXT_PERIOD" if gap <= 10 else "NEXT_AVAILABLE_REPORT"
            left = _representative(slots[(left_date, left_doc)])
            right = _representative(slots[(right_date, right_doc)])
            links.append({"head": left["id"], "tail": right["id"], "type": "事态延续",
                          "change": _transition(left["dir"], right["dir"]), "gap_days": gap,
                          "period_relation": period_relation, "var": var, "confidence": 1.0, "source": "跨期"})
    return links


def expectation(events, documents):
    """预期验证: a forecast checked against the next report that measures the same variable."""
    happened = _occurrences(events, documents, status="已发生")
    dates = {doc["id"]: _parse_date(doc["report_date"]) for doc in documents}
    series = {doc["id"]: _series_key(doc) for doc in documents}
    links = []
    for event in events:
        when = dates.get(event["doc_id"])
        key = (series.get(event["doc_id"]) or "", event["var"],
               (compact(event.get("subject") or ""), compact(event.get("scope") or "")))
        if not key[0] or event["status"] != "预测中" or when is None or key not in happened:
            continue
        later = sorted(slot for slot in happened[key] if 0 < (slot[0] - when).days <= 10)
        if not later:
            continue
        target = _representative(happened[key][later[0]])
        if event["dir"] == "不明" or target["dir"] == "不明":
            outcome = "待定"
        elif event["dir"] == target["dir"] or (event["dir"], target["dir"]) in (("恢复", "上升"), ("上升", "恢复")):
            outcome = "兑现"
        elif (event["dir"], target["dir"]) in OPPOSITE:
            outcome = "落空"
        else:
            outcome = "部分兑现"
        links.append({"head": event["id"], "tail": target["id"], "type": "预期验证", "outcome": outcome,
                      "gap_days": (later[0][0] - when).days, "var": event["var"], "confidence": 1.0,
                      "source": "跨期"})
    return links


def abstract_rules(events, relations):
    by_id = {event["id"]: event for event in events}
    rules = {}
    for relation in relations:
        if relation["type"] not in CAUSAL:
            continue
        head, tail = by_id[relation["head"]], by_id[relation["tail"]]
        if head["var"] == tail["var"] or "不明" in (head["dir"], tail["dir"]):
            continue
        key = (head["abstract"], relation["type"], tail["abstract"])
        rule = rules.setdefault(key, {"head": head["abstract"], "type": relation["type"], "tail": tail["abstract"],
                                      "docs": set(), "evidence": relation["evidence"], "support": 0})
        rule["docs"].add(head["doc_id"])
        rule["support"] += 1
    return rules


def cross_candidates(events, relations, documents):
    """跨期传导 candidates: an abstract rule seen anywhere, instantiated across consecutive reports."""
    rules = abstract_rules(events, relations)
    dates = {doc["id"]: _parse_date(doc["report_date"]) for doc in documents}
    by_abstract = defaultdict(list)
    for event in events:
        if dates.get(event["doc_id"]) is not None and event["dir"] != "不明":
            by_abstract[event["abstract"]].append(event)
    caused_in_doc = defaultdict(set)
    by_id = {event["id"]: event for event in events}
    for relation in relations:
        if relation["type"] in CAUSAL:
            caused_in_doc[relation["tail"]].add(by_id[relation["head"]]["abstract"])
    candidates, seen = [], set()
    for rule in sorted(rules.values(), key=lambda item: (-item["support"], item["head"], item["tail"])):
        for head in by_abstract.get(rule["head"], []):
            for tail in by_abstract.get(rule["tail"], []):
                gap = (dates[tail["doc_id"]] - dates[head["doc_id"]]).days
                if gap <= 0 or gap > MAX_GAP_DAYS or tail["status"] != "已发生":
                    continue
                if rule["head"] in caused_in_doc[tail["id"]]:
                    continue
                key = (head["id"], tail["id"])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"head": head["id"], "tail": tail["id"], "type": rule["type"],
                                   "gap_days": gap, "rule": rule})
    candidates.sort(key=lambda item: (-item["rule"]["support"], item["gap_days"], item["head"], item["tail"]))
    best_per_tail = Counter()
    trimmed = []
    for candidate in candidates:
        if best_per_tail[(candidate["tail"], candidate["rule"]["head"])] >= 1:
            continue
        best_per_tail[(candidate["tail"], candidate["rule"]["head"])] += 1
        trimmed.append(candidate)
    return trimmed[:CROSS_LIMIT], len(rules)


def verify_cross(chat, candidates, events, documents, workers, verdicts_out=None):
    if not candidates:
        return []
    by_id = {event["id"]: event for event in events}
    dates = {doc["id"]: doc["report_date"] for doc in documents}
    batches = [candidates[start:start + CROSS_BATCH] for start in range(0, len(candidates), CROSS_BATCH)]

    def describe(event):
        return {"date": dates.get(event["doc_id"], ""), "name": event["name"], "status": event["status"],
                "evidence": event["evidence"]}

    def run(batch):
        pairs = []
        for index, candidate in enumerate(batch, 1):
            pairs.append({"id": "p%d" % index, "A": describe(by_id[candidate["head"]]),
                          "B": describe(by_id[candidate["tail"]]),
                          "rule": {"relation": candidate["type"], "evidence": candidate["rule"]["evidence"]}})
        parsed = chat.complete_json(CROSS_SYSTEM, json.dumps({"pairs": pairs}, ensure_ascii=False),
                                    max_tokens=3072, tag="cross")
        rows = parsed if isinstance(parsed, list) else (parsed or {}).get("pairs", []) if isinstance(parsed, dict) else []
        verdicts = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
        accepted = []
        for index, candidate in enumerate(batch, 1):
            row = verdicts.get("p%d" % index)
            verdict = str((row or {}).get("verdict") or "").strip()
            reason = str((row or {}).get("reason") or "").strip()
            if verdict == "成立":
                status = "candidate"
            elif verdict == "不成立":
                status = "rejected"
            else:
                status = "uncertain"
            if verdicts_out is not None:
                verdicts_out.append({"head": candidate["head"], "tail": candidate["tail"],
                                     "relation": candidate["type"], "gap_days": candidate["gap_days"],
                                     "rule_evidence": candidate["rule"]["evidence"],
                                     "rule_support": candidate["rule"]["support"],
                                     "verdict": verdict or "未回答", "reason": reason, "status": status})
            if verdict == "成立":
                accepted.append({"head": candidate["head"], "tail": candidate["tail"], "type": "跨期传导",
                                 "relation": candidate["type"], "gap_days": candidate["gap_days"],
                                 "rule_evidence": candidate["rule"]["evidence"],
                                 "rule_support": candidate["rule"]["support"],
                                 "reason": reason, "confidence": 0.6, "source": "跨期",
                                 "relation_class": "LEGACY_CANDIDATE",
                                 "verification_status": "UNVERIFIED"})
        return accepted

    accepted = []
    for part in parallel_map(run, batches, workers, "跨期传导核验"):
        accepted.extend(part)
    return accepted


def _abstract_verified(relation, kind, events, evidence_index):
    """An abstract mechanism only inherits a relation the shared verifier has accepted."""
    from eventkg.support import verify_relation
    probe = dict(relation)
    probe["relation"] = kind
    if not probe.get("evidence_ids"):
        probe["evidence_ids"] = ["ev:rel:%s:%s:0" % (relation.get("head"), relation.get("tail"))]
    record = verify_relation(probe, events, evidence_index)
    return record.get("verification_status") == "VERIFIED", record


def abstract_layer(events, relations, cross_links, variables, evidence_index=None):
    abstracts = {}
    for event in events:
        node = abstracts.setdefault(event["abstract"], {
            "key": event["abstract"], "var": event["var"], "dir": event["dir"], "instances": [],
            "docs": set(), "statuses": Counter(),
        })
        node["instances"].append(event["id"])
        node["docs"].add(event["doc_id"])
        node["statuses"][event["status"]] += 1
    by_id = {event["id"]: event for event in events}
    lifted = {}
    for relation in list(relations) + list(cross_links):
        kind = relation.get("relation") if relation["type"] == "跨期传导" else relation["type"]
        if kind == "必要条件" or relation.get("basis") == "综合推断":
            continue
        status = relation.get("verification_status")
        if status not in (None, "", "VERIFIED"):
            continue
        if kind not in ("导致", "促进", "抑制", "拖累") and kind not in COMPOSITION:
            continue
        record = None
        if evidence_index is not None:
            accepted, record = _abstract_verified(relation, kind, events, evidence_index)
            if not accepted:
                continue
        head, tail = by_id[relation["head"]], by_id[relation["tail"]]
        if head["var"] == tail["var"] or "不明" in (head["dir"], tail["dir"]):
            continue
        key = (head["abstract"], kind, tail["abstract"])
        edge = lifted.setdefault(key, {"head": head["abstract"], "tail": tail["abstract"], "type": kind,
                                       "support": 0, "docs": set(), "seen": set(), "cross_period": 0,
                                       "confidence": 0.0,
                                       "evidence": relation.get("evidence") or relation.get("rule_evidence", "")})
        evidence_key = compact(relation.get("evidence") or relation.get("rule_evidence") or relation.get("head", ""))
        if evidence_key in edge["seen"]:
            continue
        edge["seen"].add(evidence_key)
        catalog = evidence_index if isinstance(evidence_index, dict) else {
            item.get("evidence_id"): item for item in evidence_index or [] if isinstance(item, dict)}
        source_docs = set()
        for evidence_id in (record or {}).get("supporting_evidence_ids") or []:
            item = catalog.get(evidence_id) or {}
            source_docs.add(item.get("document_id") or item.get("doc_id") or relation.get("doc_id") or "")
        source_docs.discard("")
        if not source_docs and relation.get("doc_id"):
            source_docs.add(relation["doc_id"])
        edge["docs"].update(source_docs)
        edge["support"] = len(edge["docs"])
        edge["cross_period"] += relation["type"] == "跨期传导"
        edge["confidence"] = max(edge["confidence"], relation["confidence"])
    for node in abstracts.values():
        node["docs"] = sorted(node["docs"])
        node["statuses"] = dict(node["statuses"])
        variable = variables.get(node["var"], {})
        node["dim"] = variable.get("dim", "")
    edges = []
    for edge in lifted.values():
        edge["docs"] = sorted(edge["docs"])
        edge.pop("seen", None)
        edges.append(edge)
    edges.sort(key=lambda item: (-item["support"], item["head"], item["tail"]))
    return abstracts, edges


_EVENT_DATE = re.compile(r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})")


def _canonical_indicator(text, scope):
    core = compact(text or "")
    core = re.sub(r"（.*?）|\(.*?\)", "", core)
    core = re.sub(r"^\d+(?:\.\d+)?%", "", core)
    core = re.sub(r"^\d+港", "", core)
    if scope and core.startswith(compact(scope)):
        core = core[len(compact(scope)):]
    return _alias_indicator(core.strip("的 "), scope or "")


def _indicator_match(event, measure):
    left = _canonical_indicator(event.get("indicator"), event.get("scope"))
    header = measure.get("row_header") or measure.get("indicator_id") or ""
    right = _canonical_indicator(header, measure.get("scope") or measure.get("region") or "")
    return bool(left and right and left == right)


def _event_date(event):
    for field in ("event_time", "time"):
        match = _EVENT_DATE.search(str(event.get(field) or ""))
        if match:
            return "%s-%02d-%02d" % (match.group(1), int(match.group(2)), int(match.group(3)))
    return ""


def _periods_align(event, observed, report_date):
    """本期 can meet this report's observation date. A missing period is not this week."""
    if not observed or observed == "UNKNOWN":
        return False
    if _EVENT_DATE.fullmatch(observed):
        event_date = _event_date(event)
        if event_date:
            return event_date == observed
        relative = compact(event.get("time") or "")
        if relative and time_bucket(relative) == "本期" and report_date == observed:
            return True
        if not relative and not event.get("event_time") and report_date == observed:
            return True
        return False
    event_bucket = time_bucket(event.get("observation_period") or event.get("time") or "")
    observed_bucket = time_bucket(observed)
    if event_bucket == "UNKNOWN" or observed_bucket == "UNKNOWN":
        return False
    return event_bucket == observed_bucket


def _observation_supports(event, measure, report_date=""):
    """A cell supports an event only when the indicator, period, scope and direction agree."""
    if measure.get("status") != "confirmed" or not measure.get("indicator_id"):
        return False
    if not _indicator_match(event, measure):
        return False
    event_scope, obs_scope = event.get("scope") or "", measure.get("scope") or measure.get("region") or ""
    if bool(event_scope) != bool(obs_scope):
        return False
    if event_scope and obs_scope and event_scope not in obs_scope and obs_scope not in event_scope:
        return False
    kind = measure.get("measure_kind") or ""
    event_dir = event.get("level_direction") or event.get("dir") or ""
    if event_dir == "UNKNOWN":
        event_dir = event.get("dir") or ""
    signal_dir = measure.get("level_direction") or measure.get("direction") or ""
    if signal_dir == "UNKNOWN":
        signal_dir = ""
    event_basis = event.get("comparison_basis") or ""
    signal_basis = measure.get("comparison_basis") or ""
    dated_event = explicit_date(event.get("event_time") or "")
    if signal_basis in ("同比", "环比") and signal_basis not in event_basis.split("；"):
        if event_basis or not dated_event:
            measure["observation_link_status"] = "AMBIGUOUS"
            return False
    if event_basis and signal_basis and signal_basis not in event_basis.split("；"):
        return False
    event_commodity = compact(event.get("commodity") or "")
    signal_commodity = compact(measure.get("commodity_id") or measure.get("commodity") or "")
    if event_commodity and signal_commodity and event_commodity != signal_commodity:
        measure["observation_link_status"] = "AMBIGUOUS"
        return False
    event_subject = compact(event.get("subject") or "")
    signal_subject = compact(measure.get("subject_id") or "")
    if event_subject and signal_subject and event_subject != signal_subject:
        return False
    if kind == "level":
        if event_dir in ("上升", "下降"):
            return False
    elif event_dir and signal_dir and event_dir != signal_dir:
        return False
    observed = measure.get("observation_period") or ""
    if not _periods_align(event, observed, report_date):
        measure["observation_link_status"] = "AMBIGUOUS"
        return False
    return True


def attach_observations(events, documents):
    """Store observations apart from event instances and link only a confirmed cell."""
    links = []
    by_doc = defaultdict(list)
    for event in events:
        by_doc[event["doc_id"]].append(event)
    observations = []
    for doc in documents:
        for observation in doc.get("observations") or []:
            measures = observation.get("measures")
            if not measures:
                measures = parse_table_measures(
                    observation.get("text") or "", document_id=doc["id"],
                    source_table_id=observation.get("id") or "", source_page=observation.get("page"),
                    source_section=observation.get("section") or "",
                    commodity_id=doc.get("commodity") or "")
            if measures:
                for measure in measures:
                    item = dict(measure)
                    item["id"] = item["observation_id"] or observation["id"]
                    item["doc_id"] = doc["id"]
                    item["section"] = observation.get("section") or item.get("source_section") or ""
                    item["page"] = observation.get("page") if observation.get("page") is not None else item.get("source_page")
                    item["text"] = item.get("source_text") or observation.get("text") or ""
                    observations.append(item)
                    if not _observation_supports_ready(item):
                        continue
                    matched = [event["id"] for event in by_doc[doc["id"]]
                               if _observation_supports(event, item, doc.get("report_date") or "")]
                    links.extend({"head": event_id, "tail": item["id"], "type": "数据支撑",
                                  "relation": "OBSERVATION_OF",
                                  "observation_period": item.get("observation_period") or "",
                                  "report_date": doc.get("report_date") or ""} for event_id in matched)
    return observations, links


def _observation_supports_ready(item):
    return item.get("status") == "confirmed" and bool(item.get("indicator_id"))
