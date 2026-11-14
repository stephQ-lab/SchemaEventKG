# -*- coding: utf-8 -*-
"""Combine the two annotation sheets. Does not score a constructed slice."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="合并 Legacy 与 Abductive 的人工标注表")
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--abductive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    legacy = json.loads(Path(args.legacy).read_text(encoding="utf-8"))
    abductive = json.loads(Path(args.abductive).read_text(encoding="utf-8"))
    left = (legacy.get("manifest") or {}).get("snapshot_sha256")
    right = (abductive.get("manifest") or {}).get("snapshot_sha256")
    if not left or left != right:
        raise SystemExit("两份结果的事件快照 SHA256 不同，不能对照")
    report = {
        "model_accuracy": None,
        "constructed_slice_f1_used": False,
        "note": "这是待人工标注的对照表。构造切片 F1 不是真实模型准确率。",
        "snapshot_sha256": left,
        "code_version": (legacy.get("manifest") or {}).get("code_version"),
        "prompt_versions": (legacy.get("manifest") or {}).get("prompt_versions"),
        "legacy": legacy,
        "abductive": abductive,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"snapshot_sha256": left, "output": str(path),
                      "legacy_confirmed": legacy["counts"]["confirmed"],
                      "abductive_confirmed": abductive["counts"]["confirmed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
