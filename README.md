# 钢联研报事理图谱

十篇研报一次跑完，得到一张图：前半是开放三元组（实体、品类、句内关系），后半是可跨文档检索的事件链。

## 节点里有什么

节点写**具体事物、时间和趋势**，不写对比数字。

- 保留：品种、指标、企业、地点、装置，以及用来区分对象的口径（47港、247家钢厂、62%澳粉）。
- 保留时间：本期、下期、上周、5月、截至5月21日。
- 价格、库存、产量、发运的数字对比写成趋势：上升、下降、持平。原文里的上涨、下跌、累库、去库都收成这三个方向。`108.1美元/吨`、`75万吨` 只留在原文依据里，不单独成节点。
- 同一篇里，同一个概念的不同说法、不同断句、从长句里截出的短词，合成一个节点。例如「Mysteel62%澳粉远期价格指数周环比上涨1.75美元至108.1美元」和「62%澳粉远期价格指数上涨」是同一个节点：`62%澳粉远期价格指数上升`。
- 不合在一起的：45港和47港、本期和下期、上升和下降、不同矿种。

事件层的查询名是「时间 + 口径 + 对象 + 趋势」，例如 `本期47港铁矿石港口库存下降`。具体万吨、美元留在「证据审计」节点和原始表格里，不作为实例节点上的数值属性。

v1.1.9 把时间、统计口径、证据来源和关系类型收到同一套身份上，并把 BM25、Dense、Hybrid 接到候选召回和 KGQA。详细变更见 `CHANGELOG.md`、`ACCEPTANCE.md` 和 `FULL_AUDIT_REPORT.md`。这些修正还没有经过真实 Qwen 和人工标注评测。正式语义召回实验需要显式指定 `--relation-method abductive --retrieval-method hybrid`，并配置 Embedding。默认的 legacy 运行不会执行 Hybrid，计划里会标成 `NOT_USED`。

品类只保留文章标题那一层。这十篇是铁矿石和 PVC。正文里的 PB粉、铁水、螺纹钢、钢材是实体，不再各占一个品类节点。词表里的下级（PB粉、铁水、电石法PVC）用「属于」连回标题品类。

## 服务器运行

解压到新目录，不要覆盖正在跑的目录。模型只使用这一台，不要再起第二个服务：

- `QWEN_BASE_URL=http://172.20.20.24:8003/v1`
- `QWEN_MODEL=qwen3.8-27b`
- Python：`/apply/home/wangyixian/Users/JieyaoQin/py311/bin/python`

```bash
PY=/apply/home/wangyixian/Users/JieyaoQin/py311/bin/python
cd /path/to/AutoSchemaKG_eventkg

"$PY" -m pip install -r requirements.txt

export QWEN_BASE_URL=http://172.20.20.24:8003/v1
export QWEN_MODEL=qwen3.8-27b
read -s QWEN_API_KEY
export QWEN_API_KEY

"$PY" -u run_unified.py --preflight
```

预检应打印文档 10、chunks 约 191、event_windows 26、thinking false。

这一版改过抽取提示词，也改了品类的写出规则。**用新的输出目录，不要复制旧的 `llm_cache.jsonl`。** 桌面上那份 `unified10_full.cypher` 的缓存对不上现在的窗口抽取。

先只抽事件，写成快照。目录要在重定向之前建好。真实对照见 `QWEN_RUN.md`。

```bash
mkdir -p example/generated/events_snapshot
nohup "$PY" -u run_unified.py --snapshot-only --workers 1 \
  --output example/generated/events_snapshot \
  > example/generated/events_snapshot/run.log 2>&1 &
tail -f example/generated/events_snapshot/run.log
```

确认后跑完整流水线（三元组 + 事件层 + 合并）。`--workers 8`。

```bash
mkdir -p example/generated
nohup "$PY" -u run_unified.py \
  --output example/generated/unified10_category \
  --workers 8 \
  > example/generated/unified10.log 2>&1 &
tail -f example/generated/unified10.log
```

结果：

- `example/generated/unified10_category/neo4j/unified10_full.cypher`
- `example/generated/unified10_category/event_layer/report.md`
- `example/generated/unified10_category/event_layer/graph.json`

导入 Neo4j（密码用 `read -s`，不要写进命令行历史以外的文件）：

```bash
read -s NEO4J_PASSWORD
export NEO4J_PASSWORD
cypher-shell -a bolt://192.168.169.201:7687 -u neo4j -p "$NEO4J_PASSWORD" \
  -f example/generated/unified10_category/neo4j/unified10_full.cypher
```

更细的阶段说明、查询和缓存规则在 `SERVER_RUN.txt`。
