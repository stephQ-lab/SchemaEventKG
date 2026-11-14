# -*- coding: utf-8 -*-
"""Closed vocabularies of the event layer and the commodity chain seed.

Commodity names, grade-to-category links and node sharing come from the triple
half (atlas_rag.kg_construction.commodity) so that both halves use one uid.
"""
import re

from eventkg.relations import CAUSAL_TYPES, pattern_for
from atlas_rag.kg_construction.commodity import (
    PARENT,
    document_categories,
    match_commodities,
    shared_scope,
)

DIRECTIONS = ("上升", "下降", "持平", "中断", "恢复", "不明")
STATUSES = ("已发生", "预测中")
PERIODS = ("当期", "月度", "年度")

# 维度 -> (编码, 事件类型). 编码沿用 event_link 的 DIM_CODE_MAP，没有编码的留空。
# 事件类型与三元组一半的「供应事件」「需求事件」等同名，导入后是同一批节点。
DIMENSIONS = {
    "价格": ("WD0004", "价格事件"),
    "供应": ("WD0001", "供应事件"),
    "进口": ("WD0005", "供应事件"),
    "出口": ("WD0003", "需求事件"),
    "需求": ("WD0007", "需求事件"),
    "库存": ("WD0006", "库存事件"),
    "成本": ("WD0002", "成本利润事件"),
    "利润": ("", "成本利润事件"),
    "价差": ("", "价格事件"),
    "物流": ("", "物流事件"),
    "宏观金融": ("", "国际事件"),
    "政策": ("", "政策事件"),
    "地缘": ("", "国际事件"),
    "天气": ("", "天气事件"),
    "企业经营": ("", "企业事件"),
    "市场情绪": ("", "需求事件"),
}

# Known transmission signs of the ore/steel chain: (head indicator, tail indicator, tail commodity excluded, sign).
# A research summary lists factors that push both ways before its conclusion; a factor whose
# movement runs against the sign here only works against the conclusion.
SIGN_PRIORS = (
    (r"到港", r"港口库存|港存", "", 1),
    (r"疏港", r"港口库存|港存", "", -1),
    (r"发运", r"到港|港口库存|港存", "", 1),
    (r"铁水|高炉开工|高炉产能利用", r"疏港|日耗|需求", "", 1),
    (r"铁水|高炉开工|高炉产能利用", r"港口库存|港存", "", -1),
    (r"铁水|高炉开工|高炉产能利用", r"价格|价格指数|矿价", "钢厂", 1),
    (r"港口库存|港存", r"价格|价格指数|矿价", "钢厂", -1),
    (r"到港|发运", r"价格|价格指数|矿价", "钢厂", -1),
    (r"盈利率", r"铁水|高炉开工|高炉产能利用", "", 1),
)
MOVES = {"上升": 1, "恢复": 1, "下降": -1, "中断": -1}

CAUSAL = CAUSAL_TYPES
TEMPORAL = ("之前", "之后", "同时")
COMPOSITION = ("构成",)
INTRA_RELATIONS = CAUSAL + TEMPORAL + COMPOSITION

TRIGGERS = {name: pattern_for(name) for name in CAUSAL}
TRIGGERS.update({
    "之前": re.compile(r"之前|此前|先前|先于|随后|而后|然后|继而|接着|前期|后续"),
    "之后": re.compile(r"之后|随后|而后|然后|接着|此后|后续|待"),
    "同时": re.compile(r"同时|与此同时|叠加|并且|此外|一方面|另一方面|同步|伴随"),
    "构成": re.compile(r"包括|其中|表现为|主要由|包含|分别|具体来看|分区域|分品种|分类型|主要来自|集中于|减量主要|增量主要"),
})

# Upstream -> downstream. Used only to label chain-crossing factors such as
# 「焦炭下游需求」 vs 「铁矿石下游需求」; it never merges two commodities.
CHAIN = [
    ("铁矿石", "铁水"),
    ("球团", "铁水"),
    ("块矿", "铁水"),
    ("焦炭", "铁水"),
    ("炼焦煤", "焦炭"),
    ("铁水", "螺纹钢"),
    ("铁水", "热轧板卷"),
    ("铁水", "冷轧板卷"),
    ("铁水", "钢坯"),
    ("兰炭", "电石"),
    ("电石", "电石法PVC"),
    ("乙烯", "乙烯法PVC"),
    ("PVC", "型材管材"),
]

MACRO_NAMES = ("美联储", "美元", "降息", "加息", "缩表", "汇率", "原油", "关税", "地缘", "谈判",
               "宏观", "国债", "通胀", "PMI", "GDP", "人民币", "流动性")

NOISE_LINE = re.compile(
    r"^(数据来源|资料来源|来源)[:：]|请阅读尾页免责声明|免责声明|版权声明|^目录|CONTENTS|"
    r"^\s*(图|表)\s*\d+|^单位[:：]|事业部\s*$"
)


def compact(text):
    return re.sub(r"\s+", "", text or "")


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", (text or "").replace(",", "")))


def canonical_commodity(name, categories=()):
    """The most specific lexicon name in the field, or the field itself when it is macro/other."""
    raw = compact(name)
    if not raw:
        return ""
    if raw in ("宏观", "宏观金融", "宏观经济", "全球宏观"):
        return "宏观"
    hits = match_commodities(raw)
    if hits:
        child = [hit for hit in hits if PARENT.get(hit)]
        return child[0] if child else hits[0]
    return raw


def category_of(commodity, categories=()):
    """Lexicon parent when there is one; otherwise the commodity itself."""
    if not commodity:
        return ""
    parent = PARENT.get(commodity, "")
    if parent:
        return parent
    if commodity in PARENT:
        return commodity
    return ""


def is_macro(commodity, indicator=""):
    text = (commodity or "") + (indicator or "")
    return commodity == "宏观" or any(word in text for word in MACRO_NAMES)


__all__ = [
    "CAUSAL", "CHAIN", "COMPOSITION", "DIMENSIONS", "DIRECTIONS", "INTRA_RELATIONS", "NOISE_LINE",
    "PARENT", "PERIODS", "STATUSES", "TEMPORAL", "TRIGGERS", "canonical_commodity", "category_of",
    "compact", "document_categories", "is_macro", "match_commodities", "numbers", "shared_scope",
]
