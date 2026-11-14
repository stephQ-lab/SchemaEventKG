# -*- coding: utf-8 -*-
"""One pipeline: open triples, then the event-chain layer, then one Cypher file.

The character-overlap event-chain stage is not used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from atlas_rag.document_input.structure import digest_file, load_structured_sample, save_json
from eventkg.corpus import load_corpus
from eventkg.llm import thinking_enabled


ROOT = Path(__file__).resolve().parent


def pipeline_digest():
    digest = hashlib.sha256()
    for folder in (ROOT / "atlas_rag", ROOT / "eventkg"):
        for path in sorted(folder.rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def check_input(data_dir, chunk_size, expected_docs=10):
    source = data_dir / "unified10.json"
    status = json.loads((data_dir / "preparation_status.json").read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("input_sha256") != digest_file(source):
        raise ValueError("输入文件与解析完成记录不一致")
    report = json.loads((data_dir / "preparation_report.json").read_text(encoding="utf-8"))
    rows = json.loads(source.read_text(encoding="utf-8"))
    if report.get("complete") is not True or len(rows) != report["counts"]["documents"]:
        raise ValueError("输入文档数量与解析报告不一致")
    if expected_docs and len(rows) != expected_docs:
        raise ValueError("应为 %d 篇 PDF，当前是 %d" % (expected_docs, len(rows)))
    seen = set()
    structures = hashlib.sha256()
    count = 0
    longest = 0
    for row in rows:
        if row["id"] in seen:
            raise ValueError("输入文档编号重复：" + row["id"])
        seen.add(row["id"])
        if row["metadata"]["lang"] != "zh-CN" or row["metadata"].get("source_type") != ".pdf":
            raise ValueError("只接受中文 PDF：" + row["id"])
        loaded = load_structured_sample(row, data_dir, chunk_size)
        if loaded is None:
            raise ValueError("缺少文档结构文件：" + row["id"])
        _, chunks = loaded
        structures.update(digest_file(data_dir / row["metadata"]["structure_file"]).encode("ascii"))
        count += len(chunks)
        longest = max(longest, *(len(chunk["text"]) for chunk in chunks))
    documents = load_corpus(data_dir)
    return {
        "input_sha256": digest_file(source),
        "structure_sha256": structures.hexdigest(),
        "registry_sha256": digest_file(data_dir / "document_registry.json"),
        "pipeline_sha256": pipeline_digest(),
        "documents": len(rows),
        "chunks": count,
        "longest_chunk": longest,
        "triple_requests_at_least": count * 3,
        "narrative_sections": sum(len(doc["sections"]) for doc in documents),
        "event_windows": sum(len(doc["windows"]) for doc in documents),
        "observations": sum(len(doc["observations"]) for doc in documents),
        "chunk_size": chunk_size,
        "model": os.environ.get("QWEN_MODEL", "qwen3.8-27b"),
        "thinking": thinking_enabled(),
    }


def merge_cypher(triple_path, event_path, dest):
    parts = [
        "// 一张图。前半是开放三元组，后半是事件链。",
        "// 品类只来自文章标题。正文品种是实体，用「属于」连到标题品类。文档、品类、事件类型两边 uid 相同。",
        "// 实例事件不合并。跨文档的边在事件链这一半。",
        "",
    ]
    for path in (triple_path, event_path):
        if not path.is_file():
            raise ValueError("缺少待合并的 Cypher：" + str(path))
        parts.append(path.read_text(encoding="utf-8").rstrip())
        parts.append("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return dest


def run_graph(args, plan, data_dir, output):
    from openai import OpenAI
    from atlas_rag.kg_construction.triple_config import ProcessingConfig
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
    from atlas_rag.llm_generator import LLMGenerator
    from atlas_rag.llm_generator.generation_config import GenerationConfig

    base = os.environ.get("QWEN_BASE_URL", "")
    key = os.environ.get("QWEN_API_KEY", "")
    if os.environ.get("NEO4J_URI"):
        raise ValueError("运行时请取消 NEO4J_URI。完成后导入 unified10.cypher")
    if not base or not key:
        raise ValueError("请设置 QWEN_BASE_URL 和 QWEN_API_KEY")
    state_path = output / "run_state.json"
    state = {"plan": plan, "completed": []}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan") != plan:
            raise ValueError("输入或运行参数已经变化，请指定新的 --output 目录")
    save_json(state_path, state)
    client = OpenAI(base_url=base, api_key=key, timeout=args.request_timeout, max_retries=1)
    available = [item.id for item in client.models.list().data]
    if plan["model"] not in available:
        raise ValueError("QWEN_MODEL 不在接口模型列表中：" + str(available))
    generator = LLMGenerator(client=client, model_name=plan["model"], max_workers=args.workers,
                             default_config=GenerationConfig(
                                 max_tokens=args.max_tokens, temperature=0.0, do_sample=False,
                                 chat_template_kwargs={"enable_thinking": thinking_enabled()}))
    generator.strict_errors = True
    config = ProcessingConfig(
        model_path=plan["model"], data_directory=str(data_dir), filename_pattern="unified10",
        output_directory=str(output), batch_size_triple=args.workers, batch_size_concept=8,
        max_workers=args.workers, include_concept=True, remove_doc_spaces=False,
        preserve_document_context=True, link_commodities=True, document_scope_nodes=True,
        concept_grounded=True, explicit_document_links=True, resume_extraction=True,
        strict_stage_errors=True, load_local_json=True, chunk_size=args.chunk_size,
        max_new_tokens=args.max_tokens,
    )
    graph = KnowledgeGraphExtractor(generator, config)

    def event_layer():
        from eventkg.pipeline import run as run_events
        run_events(data_dir, output / "event_layer", workers=args.workers, expected_docs=plan["documents"],
                   relation_method=args.relation_method, retrieval_method=args.retrieval_method)

    def combine():
        event_cypher = "event_layer_abductive.cypher" if args.relation_method == "abductive" else "event_layer.cypher"
        merge_cypher(
            output / "neo4j" / "unified10.cypher",
            output / "event_layer" / event_cypher,
            output / "neo4j" / "unified10_full.cypher",
        )

    stages = [
        ("extraction", graph.run_extraction),
        ("document_link", graph.link_documents),
        ("triples_csv", graph.convert_json_to_csv),
        ("concepts", lambda: graph.generate_concept_csv_temp(language="zh-CN")),
        ("concept_csv", graph.create_concept_csv),
        ("neo4j", graph.convert_to_neo4j),
        ("event_layer", event_layer),
        ("merge", combine),
    ]
    for number, (name, action) in enumerate(stages, 1):
        if name in state["completed"]:
            print("已完成阶段", number, name, flush=True)
            continue
        print("开始阶段", number, name, flush=True)
        action()
        state["completed"].append(name)
        save_json(state_path, state)
    print("结果：", output / "neo4j" / "unified10_full.cypher", flush=True)


def main():
    parser = argparse.ArgumentParser(description="十篇 PDF：开放三元组与事件链的一条流水线")
    parser.add_argument("--data-dir", default="example/example_data/unified10")
    parser.add_argument("--output", default="example/generated/unified10")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--expected-docs", type=int, default=10, help="0 表示不检查篇数")
    parser.add_argument("--only-events", action="store_true", help="只跑事件层（不跑三元组）")
    parser.add_argument("--snapshot-only", action="store_true",
                        help="只抽取事件并写 event_snapshot.json，不跑跨期核验，也不跑溯因")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--relation-method", choices=("legacy", "abductive"), default="legacy",
                        help="跨文档关系：legacy 为历史规律逐对判断，abductive 为目标事件的解释图。与 --snapshot-only 互斥")
    parser.add_argument("--retrieval-method", choices=("bm25", "dense", "hybrid"), default="hybrid",
                        help="只在 --relation-method abductive 时用于候选召回。legacy 不会执行 Hybrid。没有向量模型时请显式写 bm25")
    args = parser.parse_args()
    if min(args.workers, args.max_tokens) < 1 or args.request_timeout <= 0 or args.chunk_size < 1:
        raise ValueError("运行参数必须为正数")
    data_dir = Path(args.data_dir)
    output = Path(args.output)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not output.is_absolute():
        output = ROOT / output
    plan = check_input(data_dir, args.chunk_size, args.expected_docs)
    retrieval_status = args.retrieval_method if args.relation_method == "abductive" else "NOT_USED"
    plan.update(max_tokens=args.max_tokens, workers=args.workers, relation_method=args.relation_method,
                retrieval_method=args.retrieval_method, retrieval_status=retrieval_status)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.snapshot_only and args.only_events:
        raise ValueError("--snapshot-only 已经只抽事件，不要再加 --only-events")
    if args.snapshot_only:
        from eventkg.pipeline import run as run_events
        run_events(data_dir, output, workers=args.workers, expected_docs=plan["documents"],
                   relation_method="snapshot", retrieval_method=args.retrieval_method)
        return
    if args.only_events:
        from eventkg.pipeline import run as run_events
        run_events(data_dir, output / "event_layer", workers=args.workers, expected_docs=plan["documents"],
                   relation_method=args.relation_method, retrieval_method=args.retrieval_method)
        return
    run_graph(args, plan, data_dir, output)


if __name__ == "__main__":
    main()
