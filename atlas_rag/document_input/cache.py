import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from atlas_rag.document_input.parsers import LayoutParser
from atlas_rag.document_input.structure import save_json
from atlas_rag.document_input.table_processing import refine_pdf_tables


_parser = None


def cache_key(source_hash, settings_hash, doc_id):
    return hashlib.sha256((source_hash + settings_hash + doc_id).encode("utf-8")).hexdigest()


def refined_path(cache_dir):
    signature = hashlib.sha256()
    for name in ("table_processing.py", "structure.py"):
        signature.update(Path(__file__).with_name(name).read_bytes())
    return Path(cache_dir) / ("refined_" + signature.hexdigest()[:20] + ".json")


def parse_job(job):
    global _parser
    target = Path(job["cache_dir"])
    completed = target / "document.json"
    if completed.is_file():
        document = json.loads(completed.read_text(encoding="utf-8"))
    else:
        if _parser is None:
            _parser = LayoutParser(job["model_dir"], job["threads"])
        document = _parser.parse(Path(job["path"]), job["id"], job["title"], job["sha256"], target)
        save_json(completed, document)
    if Path(job["path"]).suffix.lower() == ".pdf":
        document["source_file"] = job["path"]
        document["title"] = job["title"]
        save_json(refined_path(target), refine_pdf_tables(document))
    return job["title"]


def warm_cache(jobs, workers):
    missing = [job for job in jobs if not (Path(job["cache_dir"]) / "document.json").is_file()
               or (Path(job["path"]).suffix.lower() == ".pdf" and not refined_path(job["cache_dir"]).is_file())]
    if not missing:
        return
    print("CPU解析任务", len(missing), "进程", workers, flush=True)
    threads = str(jobs[0]["threads"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(parse_job, job) for job in missing]
        for index, future in enumerate(as_completed(futures), 1):
            title = future.result()
            print("CPU解析完成", index, "/", len(missing), title, flush=True)
