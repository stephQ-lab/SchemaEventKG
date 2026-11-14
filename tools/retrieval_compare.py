# -*- coding: utf-8 -*-
"""Offline retrieval comparison. Jaccard never enters production inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventkg.retrieval import compare_methods


def main(argv=None):
    parser = argparse.ArgumentParser(description="比较 Jaccard、BM25、Dense、Hybrid。没有 Gold 时不宣布胜负。")
    parser.add_argument("--events", required=True, help="含 events 的快照或 graph.json")
    parser.add_argument("--gold", default="", help="人工 pairs。不提供则 Recall 为空")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    snapshot = json.loads(Path(args.events).read_text(encoding="utf-8"))
    events = snapshot.get("events") or []
    gold = []
    encoder = None
    if args.gold:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8")).get("pairs") or []
    try:
        from eventkg.embeddings import encoder_from_env
        encoder = encoder_from_env()
    except RuntimeError as exc:
        encoder = None
        print(exc, file=sys.stderr)
    report = compare_methods(events, gold, encoder)
    report["encoder_configured"] = encoder is not None
    if encoder is not None:
        report["embedding_model"] = encoder.model_name
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"model_accuracy": None, "output": str(path),
                      "positives": report["positives"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
