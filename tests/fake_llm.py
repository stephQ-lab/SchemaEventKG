# -*- coding: utf-8 -*-
"""A rule-based stand-in for Qwen so the whole event layer runs offline."""
import json
import re

from eventkg.corpus import sentences_of
from eventkg.lexicon import TRIGGERS, match_commodities

INDICATORS = ("港口库存", "库存", "到港量", "到港", "发运量", "发运", "日均铁水产量", "铁水产量", "铁水",
              "疏港量", "盈利率", "开工率", "产能利用率", "价格指数", "现货价格", "期货价格", "价格", "价差",
              "进口利润", "海运费", "利润", "成本", "需求", "产量", "出口", "进口")
SCOPE = re.compile(r"\d+港|\d+家钢厂|\d+家|澳洲|巴西|全球|华东|华南|非主流")
UP = re.compile(r"上涨|上升|增加|回升|累库|走扩|扩大|走强|走高|增长|反弹")
DOWN = re.compile(r"下跌|下降|减少|回落|去库|收窄|走弱|走低|下滑|下调")
FORECAST = re.compile(r"预计|或将|有望|下期|下周|展望")


class FakeChat:
    def __init__(self):
        self.stats = {"calls": 0, "cached": 0, "failed": 0}

    def complete_json(self, system, user, max_tokens=4096, tag=""):
        self.stats["calls"] += 1
        payload = json.loads(user)
        if tag.startswith("extract:"):
            return self._extract(payload)
        if tag.startswith("document:"):
            return self._document(payload)
        if tag.startswith("ontology:"):
            return [{"id": item["id"], "key": "|".join((item["commodity"], item["indicator"], item["scope"],
                                                        item["period"])), "dim": item["dim"], "parent": ""}
                    for item in payload["items"]]
        if tag == "cross":
            return [{"id": pair["id"], "verdict": "成立", "reason": "规律一致"} for pair in payload["pairs"]]
        return None

    def _extract(self, payload):
        events, relations = [], []
        categories = payload["categories"]
        for sentence in sentences_of(payload["window"]):
            if sentence.startswith("【章节】") or sentence.startswith("###"):
                continue
            clauses = [part for part in re.split(r"[，,；;]", sentence) if part.strip()]
            in_sentence = []
            for clause in clauses:
                indicator = next((word for word in INDICATORS if word in clause), "")
                direction = "上升" if UP.search(clause) else "下降" if DOWN.search(clause) else ""
                if not indicator or not direction or len(clause) < 6:
                    continue
                scope = SCOPE.search(clause)
                hits = match_commodities(clause)
                event_id = "E%d" % (len(events) + 1)
                events.append({
                    "id": event_id, "name": clause.strip(), "commodity": hits[0] if hits else categories[0],
                    "indicator": indicator, "scope": scope.group(0) if scope else "", "dim": "",
                    "dir": direction, "value": "", "change": "", "time": "", "period": "当期",
                    "status": "预测中" if FORECAST.search(clause) else "已发生", "subject": "",
                    "evidence": sentence, "salience": "核心" if len(events) < 2 else "一般"})
                in_sentence.append(event_id)
            if len(in_sentence) >= 2 and TRIGGERS["导致"].search(sentence):
                for head, tail in zip(in_sentence, in_sentence[1:]):
                    relations.append({"head": head, "tail": tail, "type": "导致", "trigger": "",
                                      "evidence": sentence})
        return {"events": events, "relations": relations}

    def _document(self, payload):
        ids = [item["id"] for item in payload["events"]]
        lines = []
        if len(ids) >= 3:
            lines.append({"title": "测试主线", "events": ids[:3]})
        return {"relations": [], "duplicates": [], "storylines": lines}
