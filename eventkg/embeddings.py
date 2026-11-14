# -*- coding: utf-8 -*-
"""Embedding adapters. Nothing here falls back to character overlap."""
import math
import os


def _normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class OpenAIEncoder:
    """OpenAI-compatible embeddings. The base URL and model come from the environment."""

    def __init__(self, base_url, api_key, model_name):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)
        self.model_name = model_name
        self.model_version = model_name
        self.calls = 0

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return []
        self.calls += 1
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [_normalize(list(item.embedding)) for item in ordered]


class SentenceTransformerEncoder:
    """Local sentence-transformers model. Imported only when this encoder is constructed."""

    def __init__(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("未安装 sentence-transformers，不能加载本地向量模型 %s" % model_name) from exc
        self.model_name = model_name
        self.model_version = model_name
        self.calls = 0
        self._model = SentenceTransformer(model_name)

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return []
        self.calls += 1
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(vector) for vector in vectors]


def encoder_from_env():
    """Return an encoder, or None when no embedding backend was configured."""
    kind = os.environ.get("EVENTKG_EMBEDDING", "").strip().lower()
    if not kind:
        return None
    model = os.environ.get("EVENTKG_EMBEDDING_MODEL", "").strip()
    if kind == "openai":
        base = os.environ.get("EVENTKG_EMBEDDING_BASE_URL", "").strip()
        key = os.environ.get("EVENTKG_EMBEDDING_API_KEY", "").strip()
        if not base or not key or not model:
            raise RuntimeError(
                "EVENTKG_EMBEDDING=openai 时必须设置 EVENTKG_EMBEDDING_BASE_URL、"
                "EVENTKG_EMBEDDING_API_KEY 和 EVENTKG_EMBEDDING_MODEL")
        return OpenAIEncoder(base, key, model)
    if kind == "sentence":
        if not model:
            raise RuntimeError("EVENTKG_EMBEDDING=sentence 时必须设置 EVENTKG_EMBEDDING_MODEL 为模型名或本地路径")
        return SentenceTransformerEncoder(model)
    raise RuntimeError("未知的 EVENTKG_EMBEDDING=%s。可用 openai 或 sentence。不会回退到字符 Jaccard。" % kind)


def require_encoder():
    encoder = encoder_from_env()
    if encoder is None:
        raise RuntimeError(
            "Dense 模型未配置。设置 EVENTKG_EMBEDDING=openai 或 sentence，以及模型地址。"
            "不会回退到字符 Jaccard。只做关键词检索时请显式指定 --retrieval-method bm25。")
    return encoder
