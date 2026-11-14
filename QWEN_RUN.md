# 真实 Qwen 对照运行

v1.1.9 没有改模型地址。Legacy 配合 `--retrieval-method hybrid` 时，检索状态是 `NOT_USED`，不会假装已经做了向量召回。Abductive 的 hybrid 或 dense 在没有 `EVENTKG_EMBEDDING` 时直接报错。Jieba 词项 BM25 是词项检索 baseline，不是语义向量检索。Hybrid（词项 BM25 + Dense + RRF，k=60）是接上向量服务后的正式方案。没有设置 `EVENTKG_EMBEDDING` 时，`--retrieval-method hybrid` 或 `dense` 会直接报错，不会退回字符 Jaccard，也不会退回字面重合。

`run_unified.py` 的默认关系方法是 `legacy`。Legacy 不做 Abductive 候选检索，所以即使检索参数显示 `hybrid`，默认运行也不会使用 Hybrid 事件召回。正式实验必须同时写上：

```bash
--relation-method abductive --retrieval-method hybrid
```

并确认 Embedding 服务已经配置。下面单独保留一条 BM25-only 对照命令。构造切片上的 F1 不是模型准确率。这次交付没有跑真实向量对照，也没有调用 Qwen，因此不能声称 Dense 或 Hybrid 的召回已经提升。

运行后 `manifest.json` 会记下代码版本、事件快照 SHA256、语料 SHA256、Prompt 版本、检索方法、Embedding 模型名、分词版本、领域词典哈希、RRF k=60、temperature、thinking、workers 和模型地址。不要把 API Key 写进 manifest。

```bash
export EVENTKG_EMBEDDING=openai
export EVENTKG_EMBEDDING_BASE_URL=http://127.0.0.1:8004/v1
export EVENTKG_EMBEDDING_MODEL=bge-m3
read -s EVENTKG_EMBEDDING_API_KEY
export EVENTKG_EMBEDDING_API_KEY
```

本地模型则用 `EVENTKG_EMBEDDING=sentence` 和 `EVENTKG_EMBEDDING_MODEL` 指向模型名或目录。上面的地址只是示例，不要假设服务器上已经有这个接口。

只使用这一台服务，不要再启动第二个 27B，不要结束别人的 vLLM，同一时刻只有一个请求进程：

- `QWEN_BASE_URL=http://172.20.20.24:8003/v1`
- `QWEN_MODEL=qwen3.8-27b`
- Python：`/apply/home/wangyixian/Users/JieyaoQin/py311/bin/python`

```bash
PY=/apply/home/wangyixian/Users/JieyaoQin/py311/bin/python
cd /path/to/AutoSchemaKG_eventkg
export QWEN_BASE_URL=http://172.20.20.24:8003/v1
export QWEN_MODEL=qwen3.8-27b
read -s QWEN_API_KEY
export QWEN_API_KEY
mkdir -p example/generated/qwen_snapshot example/generated/qwen_legacy example/generated/qwen_abductive
```

`mkdir` 必须写在任何 `>` 重定向之前。日志都放进已经建好的目录，不要把日志写到还不存在的 `example/generated/`。

## 1. 只抽事件，写成一份快照

`--snapshot-only` 在合并篇内关系后停止。它不跑 Legacy 跨期核验，也不跑溯因。

```bash
"$PY" -u run_unified.py --snapshot-only --workers 1 \
  --output example/generated/qwen_snapshot \
  > example/generated/qwen_snapshot/run.log 2>&1
```

固定输入是：

`example/generated/qwen_snapshot/event_snapshot.json`

里面的 `events`、篇内 `relations` 和 `snapshot_sha256` 就是后面两边共用的抽取结果。不要重新抽取，也不要改这个文件。

## 2. 两条关系分开目录、分开缓存

不要把旧的 `llm_cache.jsonl` 复制到新目录。两条命令都读上面同一个快照。等上一条完全结束，确认没有 python 进程还在打 `172.20.20.24:8003`，再跑下一条。

```bash
"$PY" -u tools/run_saved_events.py \
  --graph example/generated/qwen_snapshot/event_snapshot.json \
  --data-dir example/example_data/unified10 \
  --output example/generated/qwen_legacy \
  --relation-method legacy \
  --workers 1 \
  > example/generated/qwen_legacy/run.log 2>&1
```

```bash
"$PY" -u tools/run_saved_events.py \
  --graph example/generated/qwen_snapshot/event_snapshot.json \
  --data-dir example/example_data/unified10 \
  --output example/generated/qwen_abductive \
  --relation-method abductive \
  --retrieval-method bm25 \
  --workers 1 \
  > example/generated/qwen_abductive/run.log 2>&1
```

两个输出目录一开始都没有 `llm_cache.jsonl`。脚本发现目录里已经有缓存会直接退出。两份 `manifest.json` 里的 `snapshot_sha256` 必须相同。

上面这条是 BM25-only 对照。Jieba 按词切分，并带有铁矿石、港口库存、到港、疏港、铁水、钢厂利润等词。它仍然是词项匹配，不是向量检索。

向量服务配好之后，用 Hybrid。模型名、版本和路径以当时的 `EVENTKG_EMBEDDING_MODEL` 为准，不要假设某台机器上已经有 Embedding 服务：

```bash
"$PY" -u tools/run_saved_events.py \
  --graph example/generated/qwen_snapshot/event_snapshot.json \
  --data-dir example/example_data/unified10 \
  --output example/generated/qwen_abductive_hybrid \
  --relation-method abductive \
  --retrieval-method hybrid \
  --workers 1 \
  > example/generated/qwen_abductive_hybrid/run.log 2>&1
```

如果还没有设置 `EVENTKG_EMBEDDING`，这条命令会失败并退出。不要改成静默回退。

## 3. 对照表

```bash
mkdir -p example/generated/qwen_compare
"$PY" -u tools/experiment_sheet.py \
  --legacy example/generated/qwen_legacy/annotation_sheet.json \
  --abductive example/generated/qwen_abductive/annotation_sheet.json \
  --output example/generated/qwen_compare/comparison.json \
  > example/generated/qwen_compare/run.log 2>&1
```

每份 `annotation_sheet.json` 都有目标事件覆盖率、Candidate Recall@K、确认边、存疑边、拒绝边和证据。没有人工正例时 Recall 是空的，不要用构造切片的 F1 填上。

边上的 `kind` 以证据类型为准，不根据方法名称猜测。Legacy 边上如果没有 `inference_path` 或 `relation_class`，记为 `UNCLASSIFIED`，留给人工标注，不要自动写成 `implicit_without_trigger`。

- `explicit_textual_cause`：证据类型是明确文本因果
- `segmented_cause`：分段因果组合，两段各自保留，不合成一条直接边
- `implicit_without_trigger`：证据类型本身是没有显式触发词的隐式关系
- `UNCLASSIFIED`：证据类型不足，不能归入上面三类

## 4. 要留下的原始结果

- `qwen_snapshot/event_snapshot.json`：事件、篇内关系、快照 SHA256、代码版本、Prompt 版本
- `qwen_snapshot/llm_cache.jsonl`：抽取阶段的原始模型回答
- `qwen_legacy/manifest.json`：模型名、temperature、thinking、快照 SHA256、代码版本、Prompt 版本
- `qwen_legacy/relations.json`：全部候选和逐对结论
- `qwen_legacy/llm_cache.jsonl`：Legacy 核验的原始回答（含模型名和参数）
- `qwen_legacy/annotation_sheet.json`：覆盖率、Recall 空位、确认 / 存疑 / 拒绝和证据
- `qwen_abductive/manifest.json`：与 Legacy 同一套版本字段
- `qwen_abductive/abduction.json`：每个目标的完整候选、证据和评分
- `qwen_abductive/relations.json`：确认边、存疑边、拒绝边
- `qwen_abductive/llm_cache.jsonl`：溯因的原始回答
- `qwen_abductive/annotation_sheet.json`：同上，并标出三种关系
- `qwen_compare/comparison.json`：两边对照。`model_accuracy` 保持为空
