# -*- coding: utf-8 -*-
"""Candidate retrieval. Character Jaccard is an offline baseline, not a production channel."""
import math
import re
import time
from collections import Counter

from eventkg.lexicon import compact

RRF_K = 60
_LATIN = re.compile(r"[A-Za-z0-9%.]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_METHODS = ("bm25", "dense", "hybrid", "jaccard")


_DOMAIN_WORDS = (
    "铁矿石", "澳洲发运", "港口库存", "社会库存", "钢厂库存", "总库存", "到港", "疏港",
    "铁水", "钢厂利润", "焦炭", "螺纹钢", "热轧板卷", "精粉", "装置检修", "供应中断",
    "铁精粉", "澳粉", "日均铁水产量", "发运量", "到港量", "现货价格", "期货价格",
    "高炉开工率", "盈利率", "海运费", "247家钢厂", "限产", "检修",
)
_MARKET_NUMBER = re.compile(
    r"\d+(?:\.\d+)?(?:美元/干吨|美元/吨|元/吨|万吨|吨|美元|元)")
_TOKENIZER = "jieba"
_LABELS = {"主体", "动作", "对象", "商品", "指标", "方向", "地区", "时间", "周期", "原文"}


def _retrieval_text(text):
    """Keep grades and sample sizes. Drop price and quantity figures."""
    raw = text or ""
    kept = []
    for token in re.findall(r"\d+(?:\.\d+)?%|\d+港", raw):
        if token not in kept:
            kept.append(token)
    cleaned = _MARKET_NUMBER.sub(" ", raw)
    cleaned = re.sub(r"(?<![\d.])\d+\.\d+(?!%)", " ", cleaned)
    return (cleaned + " " + " ".join(kept)).strip()


def event_signature(event):
    """Trend identity for retrieval. Market quantities stay out of the query terms."""
    values = (
        event.get("subject"), event.get("dir") or event.get("level_direction"),
        event.get("momentum"), event.get("comparison_basis"),
        event.get("indicator") or event.get("object"),
        event.get("commodity"), event.get("scope") or event.get("region"),
        event.get("event_time") or event.get("time"), event.get("observation_period"),
        event.get("status"),
        _retrieval_text(event.get("evidence") or event.get("name") or ""),
    )
    return " ".join(str(value) for value in values if value)


def tokenize(text):
    """Word-level Chinese terms. Single characters are not the production tokenizer."""
    import jieba
    for word in _DOMAIN_WORDS:
        jieba.add_word(word)
    raw = text or ""
    tokens = [item.lower() for item in _LATIN.findall(raw)]
    for word in jieba.lcut(raw):
        piece = word.strip()
        if len(piece) < 2 or piece in _LABELS:
            continue
        tokens.append(piece)
    return tokens


def tokenizer_version():
    import jieba
    return "jieba-%s" % getattr(jieba, "__version__", "unknown")


def character_jaccard(left, right):
    """Offline baseline only. Production retrieval must not call this."""
    def grams(text):
        folded = compact(text)
        return {folded[index:index + 2] for index in range(max(len(folded) - 1, 0))}

    one, other = grams(left), grams(right)
    if not one or not other:
        return 0.0
    return len(one & other) / float(len(one | other))


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class BM25Index:
    def __init__(self, rows, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.rows = list(rows)
        self.docs = []
        frequencies = []
        lengths = []
        document_frequency = Counter()
        for row in self.rows:
            tokens = tokenize(row.get("text") or "")
            counts = Counter(tokens)
            self.docs.append(counts)
            frequencies.append(counts)
            lengths.append(sum(counts.values()) or 1)
            document_frequency.update(counts.keys())
        self.frequencies = frequencies
        self.lengths = lengths
        self.average = (sum(lengths) / float(len(lengths))) if lengths else 1.0
        self.document_frequency = document_frequency
        self.size = len(self.rows)

    def score(self, query):
        tokens = tokenize(query)
        if not tokens or not self.size:
            return [0.0] * self.size
        scores = []
        for index, counts in enumerate(self.frequencies):
            total = 0.0
            length = self.lengths[index]
            for token in tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(token, 0)
                idf = math.log(1 + (self.size - df + 0.5) / (df + 0.5))
                total += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * (1 - self.b + self.b * length / self.average))
            scores.append(total)
        return scores


class VectorCache:
    def __init__(self, encoder):
        self.encoder = encoder
        self.vectors = {}
        self.calls = 0

    def encode(self, texts):
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            self.calls += 1
            encoded = list(self.encoder.encode(missing))
            if len(encoded) != len(missing):
                raise RuntimeError("编码器返回的向量数量与文本数量不一致")
            for text, vector in zip(missing, encoded):
                self.vectors[text] = [float(value) for value in vector]
        return [self.vectors[text] for text in texts]


def _ranks(scores):
    """A zero score is not recalled. It does not receive a rank."""
    positive = [index for index, score in enumerate(scores) if score > 0]
    positive.sort(key=lambda index: (-scores[index], index))
    ranks = [0] * len(scores)
    for rank, index in enumerate(positive, 1):
        ranks[index] = rank
    return ranks


def _rrf(rank_lists):
    """Only a channel that actually recalled the row contributes."""
    fused = [0.0] * len(rank_lists[0])
    for ranks in rank_lists:
        for index, rank in enumerate(ranks):
            if rank <= 0:
                continue
            fused[index] += 1.0 / (RRF_K + rank)
    return fused


def _diversify(rows, top_k):
    cap = max(2, int(top_k) // 4)
    kept, deferred, counts = [], [], Counter()
    for row in rows:
        group = row.get("group") or ""
        if group and counts[group] >= cap:
            deferred.append(row)
            continue
        if group:
            counts[group] += 1
        kept.append(row)
        if len(kept) >= top_k:
            return kept
    for row in deferred:
        if len(kept) >= top_k:
            break
        kept.append(row)
    return kept


def _require_encoder(method, encoder):
    if method in ("dense", "hybrid") and encoder is None:
        raise RuntimeError(
            "检索方式 %s 需要向量编码器。请设置 EVENTKG_EMBEDDING，或显式使用 --retrieval-method bm25。"
            "不会回退到字符 Jaccard。" % method)


def rank_events(target, events, method="bm25", encoder=None, top_k=50):
    """Rank other events. Dense similarity recalls candidates; it does not prove a cause."""
    if method not in _METHODS:
        raise ValueError("检索方式只能是 bm25、dense、hybrid 或离线对照 jaccard")
    _require_encoder(method, encoder)
    started = time.perf_counter()
    pool = [event for event in events if event.get("id") != target.get("id")]
    query = event_signature(target)
    rows = [{"id": event.get("id"), "text": event_signature(event),
             "group": event.get("commodity") or event.get("var") or ""} for event in pool]
    if method == "jaccard":
        scores = [character_jaccard(query, row["text"]) for row in rows]
        ordered = _order(rows, scores, "字符Jaccard", "离线对照，不参与正式推断")
    elif method == "bm25":
        scores = BM25Index(rows).score(query) if rows else []
        ordered = _order(rows, scores, "BM25", "词项匹配，只用于召回")
    elif method == "dense":
        ordered = _dense_order(query, rows, encoder)
    else:
        bm25_scores = BM25Index(rows).score(query) if rows else []
        dense_scores = _dense_scores(query, rows, encoder)
        fused = _rrf([_ranks(bm25_scores), _ranks(dense_scores)]) if rows else []
        ordered = _order(rows, fused, "混合检索", "RRF 融合 BM25 与向量，只用于召回")
    kept = _diversify(ordered, top_k)
    return {"method": method, "candidates": kept, "elapsed_seconds": round(time.perf_counter() - started, 6),
            "encoder_calls": getattr(encoder, "calls", 0) if method in ("dense", "hybrid") else 0,
            "model": getattr(encoder, "model_name", "") if method in ("dense", "hybrid") and encoder is not None else ""}


def _dense_scores(query, rows, encoder):
    cache = encoder if isinstance(encoder, VectorCache) else VectorCache(encoder)
    vectors = cache.encode([query] + [row["text"] for row in rows])
    encoder.calls = cache.calls
    return [_cosine(vectors[0], vector) for vector in vectors[1:]]


def _dense_order(query, rows, encoder):
    return _order(rows, _dense_scores(query, rows, encoder), "语义检索", "向量余弦相似度，只用于召回")


def _order(rows, scores, source, reason):
    ranked = []
    for row, score in zip(rows, scores):
        if score <= 0:
            continue
        ranked.append({"id": row["id"], "score": round(float(score), 6), "source": source,
                       "reason": reason, "group": row.get("group") or ""})
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked


def search_texts(query, rows, method="bm25", encoder=None, top_k=5, cache=None):
    """Search passages. A hit is not a confirmed relation."""
    _require_encoder(method if method != "jaccard" else "bm25", encoder if method in ("dense", "hybrid") else None)
    if method == "jaccard":
        raise RuntimeError("全文检索不使用字符 Jaccard")
    key = (method, query, top_k)
    if cache is not None and key in cache:
        return cache[key]
    packed = [{"id": row.get("evidence_id") or row.get("id"), "text": row.get("text") or "",
               "group": row.get("document_id") or ""} for row in rows]
    if method == "dense":
        ordered = _dense_order(query, packed, encoder)
    elif method == "hybrid":
        bm25_scores = BM25Index(packed).score(query) if packed else []
        dense_scores = _dense_scores(query, packed, encoder)
        ordered = _order(packed, _rrf([_ranks(bm25_scores), _ranks(dense_scores)]), "混合检索", "段落检索")
    else:
        ordered = _order(packed, BM25Index(packed).score(query) if packed else [], "BM25", "段落检索")
    found = ordered[:top_k]
    if cache is not None:
        cache[key] = found
    return found


def compare_methods(events, gold_pairs, encoder=None, ks=(10, 20, 50)):
    """Gold pairs are optional. Without them every recall stays empty."""
    positives = [item for item in (gold_pairs or []) if item.get("label") == 1]
    if positives:
        note = "Recall 只描述这次传入的标注，不能据此宣称 Dense 或 Hybrid 在真实研报上更好。字符 Jaccard 只作离线对照。"
    else:
        note = "没有人工正例，不能据此宣称 Dense 或 Hybrid 更好。字符 Jaccard 只作离线对照。"
    methods = {}
    by_id = {event.get("id"): event for event in events}
    for method in _METHODS:
        if method in ("dense", "hybrid") and encoder is None:
            methods[method] = {"recall": {str(k): None for k in ks}, "error": "没有向量编码器",
                               "mean_candidates": None}
            continue
        started = time.perf_counter()
        recalls = {k: [] for k in ks}
        counts = []
        calls = 0
        for item in positives:
            target = by_id.get(item.get("tail"))
            head = item.get("head")
            if target is None:
                continue
            ranked = rank_events(target, events, method=method, encoder=encoder, top_k=max(ks))
            ids = [row["id"] for row in ranked["candidates"]]
            counts.append(len(ids))
            calls += ranked["encoder_calls"]
            for k in ks:
                recalls[k].append(1 if head in ids[:k] else 0)
        methods[method] = {
            "recall": {str(k): (round(sum(recalls[k]) / len(recalls[k]), 4) if recalls[k] else None) for k in ks},
            "mean_candidates": round(sum(counts) / len(counts), 4) if counts else None,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "encoder_calls": calls,
            "model": getattr(encoder, "model_name", "") if encoder is not None else "",
        }
    return {"model_accuracy": None, "constructed_slice_f1_used": False, "note": note,
            "positives": len(positives), "methods": methods}
