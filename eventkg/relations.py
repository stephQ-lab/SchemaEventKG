# -*- coding: utf-8 -*-
"""The one relation table used by extraction, verification, paths and Base.

A trigger belongs to one relation. 推高 is 促进. 拖累 is 拖累.
因为/由于 stay with 导致 because a stated reason is still that relation.
必要条件 is listed here and is not an ordinary causal walk.
"""
import re

RELATION_WORDS = {
    "导致": ("导致", "造成", "致使", "因为", "由于", "因此", "所以", "使得"),
    "促进": ("促进", "推动", "带动", "推高", "推升"),
    "抑制": ("抑制", "压制", "限制"),
    "拖累": ("拖累", "挤压", "削弱"),
    "必要条件": ("必须", "前提", "不可缺少", "只有", "除非", "仍需", "需要", "取决于",
                 "必要", "关键在于", "有待", "才能"),
}
CAUSAL_TYPES = ("导致", "促进", "抑制", "拖累", "必要条件")
_AFFECTED = r"受[^。；\n]{0,24}影响"


def pattern_for(name):
    words = "|".join(re.escape(word) for word in RELATION_WORDS[name])
    if name == "导致":
        words = words + "|" + _AFFECTED
    return re.compile(words)


def relation_of_trigger(word):
    token = word or ""
    for name, words in RELATION_WORDS.items():
        if token in words:
            return name
    return ""


def compat_map():
    """Each surface word maps back to its own relation's allowed words."""
    table = {}
    for name, words in RELATION_WORDS.items():
        if name == "必要条件":
            continue
        for word in words:
            table[word] = words
    return table
