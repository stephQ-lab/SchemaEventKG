# -*- coding: utf-8 -*-
"""JSON completions from the running Qwen endpoint, with a disk cache and a thread pool.

Every answer is cached by (model, thinking, system, user, max_tokens). A rerun
after an interruption reads the cache, so no stage needs its own resume file.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import json_repair


def require_config():
    base = os.environ.get("QWEN_BASE_URL", "").strip()
    key = os.environ.get("QWEN_API_KEY", "").strip()
    model = os.environ.get("QWEN_MODEL", "qwen3.8-27b").strip() or "qwen3.8-27b"
    if not base or not key:
        raise RuntimeError("请设置 QWEN_BASE_URL 和 QWEN_API_KEY")
    return base, key, model


def thinking_enabled():
    return os.environ.get("QWEN_THINKING", "0").strip() in {"1", "true", "yes"}


class JsonlCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        self.lock = threading.Lock()
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.rows[row["key"]] = row

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row):
        row = dict(row, key=key)
        with self.lock:
            self.rows[key] = row
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json(content):
    text = (content or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        return None
    return json_repair.loads(text)


def endpoint_fingerprint(url=""):
    """A stable id of the service address. The API key is not part of it."""
    text = url if url is not None else os.environ.get("QWEN_BASE_URL", "")
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def cache_key(model, thinking, system, user, max_tokens, endpoint=""):
    if endpoint == "":
        endpoint = os.environ.get("QWEN_BASE_URL", "")
    raw = "\n\x00".join([model, endpoint_fingerprint(endpoint), str(thinking), system, user, str(max_tokens)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ReplayChat:
    """Answers only from an existing cache; a miss returns None. For re-running post-processing offline."""

    def __init__(self, cache_path):
        self.model = os.environ.get("QWEN_MODEL", "qwen3.8-27b").strip() or "qwen3.8-27b"
        self.thinking = thinking_enabled()
        self.cache = JsonlCache(cache_path)
        self.stats = {"calls": 0, "cached": 0, "failed": 0, "missed_tags": []}
        self._lock = threading.Lock()

    def complete_json(self, system, user, max_tokens=4096, tag=""):
        hit = self.cache.get(cache_key(self.model, self.thinking, system, user, max_tokens))
        with self._lock:
            if hit is None:
                self.stats["failed"] += 1
                self.stats["missed_tags"].append(tag)
            else:
                self.stats["cached"] += 1
        return parse_json(hit["content"]) if hit else None


class ChatClient:
    """OpenAI-compatible client. Thinking is off unless QWEN_THINKING=1."""

    def __init__(self, cache_path, timeout=900, workers=8):
        from openai import OpenAI

        base, key, self.model = require_config()
        self.client = OpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=0)
        self.thinking = thinking_enabled()
        self.cache = JsonlCache(cache_path)
        self.workers = max(1, int(workers))
        self.stats = {"calls": 0, "cached": 0, "failed": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "seconds": 0.0}
        self._lock = threading.Lock()

    def model_available(self):
        names = {item.id for item in self.client.models.list().data}
        if self.model not in names:
            raise RuntimeError("接口不提供模型：%s，可用：%s" % (self.model, sorted(names)))

    def _key(self, system, user, max_tokens):
        return cache_key(self.model, self.thinking, system, user, max_tokens)

    def complete_json(self, system, user, max_tokens=4096, tag=""):
        key = self._key(system, user, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            with self._lock:
                self.stats["cached"] += 1
            return parse_json(hit["content"])
        budget = max_tokens
        last_error = None
        for attempt in range(4):
            started = time.time()
            try:
                answer = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=budget,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
                )
                choice = answer.choices[0]
                content = choice.message.content or ""
                usage = getattr(answer, "usage", None)
                with self._lock:
                    self.stats["calls"] += 1
                    self.stats["seconds"] += time.time() - started
                    if usage is not None:
                        self.stats["prompt_tokens"] += usage.prompt_tokens or 0
                        self.stats["completion_tokens"] += usage.completion_tokens or 0
                if choice.finish_reason == "length":
                    budget = min(int(budget * 1.6), 16384)
                    raise RuntimeError("输出达到 token 上限，扩大到 %d 重试" % budget)
                parsed = parse_json(content)
                if parsed is None:
                    raise RuntimeError("模型没有返回 JSON")
                self.cache.put(key, {"tag": tag, "content": content, "model": self.model,
                                      "temperature": 0, "thinking": self.thinking, "max_tokens": max_tokens,
                                      "finish_reason": choice.finish_reason})
                return parsed
            except Exception as exc:  # network errors, truncated output, invalid JSON
                last_error = exc
                time.sleep(2 * (attempt + 1))
        with self._lock:
            self.stats["failed"] += 1
        print("模型请求失败", tag, last_error, file=sys.stderr, flush=True)
        return None


def parallel_map(function, items, workers, label=""):
    """Run function over items with a thread pool. Order is kept. Progress goes to stdout."""
    items = list(items)
    if not items:
        return []
    results = [None] * len(items)
    done = [0]
    lock = threading.Lock()
    step = max(1, len(items) // 20)

    def run(index):
        results[index] = function(items[index])
        with lock:
            done[0] += 1
            if label and (done[0] % step == 0 or done[0] == len(items)):
                print("%s %d/%d" % (label, done[0], len(items)), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(run, range(len(items))))
    return results
