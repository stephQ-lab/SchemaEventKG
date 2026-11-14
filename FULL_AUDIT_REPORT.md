# AutoSchemaKG_eventkg 全流程审计

## v1.1.9 收口

| 位置 | 问题 | 修复 |
| --- | --- | --- |
| `validate.clean_event` | 整段 evidence 的「预计下周」污染另一个事件 | 只读目标 Mention 分句 |
| `identity.can_merge_event_instances` | 空主体可并入已知主体；现货和期货可并 | 缺主体且无共指则拒绝；现货/期货分开 |
| `ontology.tidy_scope` / `tidy_mill` | 全国和 247家钢厂被清空 | 保留为明确口径 |
| `graph_induction.report_series_id` | 无机构时退成商品:频率 | 机构、商品、频率缺一则无系列 |
| `link.continuity` | 只按系列和变量 | 同时要求主体和口径 |
| `support.verify_relation` | 他篇无日期泛化句可绑定 | 无日期、无口径、无 Mention 则不绑定 |
| `link.align_polarity` | Legacy 把原文关系改成抑制 | 默认只记 `POLARITY_CONFLICT` |
| `link.verify_cross` | 模型说成立即成跨期因果 | `LEGACY_CANDIDATE` |
| `retrieval.event_signature` | 行情数字进入检索 | 去掉价格和数量，保留品位和港数 |
| `retrieve.Index.seeds` | 固定 BM25 | 使用图上的 retrieval_method |
| `llm.cache_key` | 不含服务地址 | 写入地址指纹，不写 API Key |
| `export.Builder.edge` | 后写的证据被丢掉 | 合并证据编号；未验证边导出为候选关系 |
| `export._resolve_category` | 宏观事件挂上唯一标题品类 | 宏观不接收标题品类 |

真实 Qwen 和真实 Embedding 本轮没有运行。Gold 指标保持 null。

## v1.1.8 收口

这一轮只修时间、实例证据和关系类型，不增加推理模块。

- `time_bucket()` 不再把认不出的表达收成「本期」。本周、上周、下周保持原规则。近期、上月、下月、本季度、年初以来、同期不再并进本期，因此不能和本周事件合并。
- `verify_relation()` 对两边都没有方向的机制句不确认具体升降事件。本期只与本篇 `report_date` 对齐。支持和否定按时期分组，另一时期的正例不能抵消本期反证。已有 Mention 映射时，不再把同一句绑到另一个事件。
- `eventkg/relations.py` 是导致、促进、抑制、拖累、必要条件的唯一触发词表。抽取、Base 归一、间接路径标签和证据验证都读这张表。
- 延续和预测验证复用 `report_series_id()`。系列不同或系列未知时不自动连接。
- 无时期观测不自动绑定。库存类型按目标指标判断。相同原文保留每个来源的证据编号和日期。局部显式关系在 Top-K 之前保留。

仍未完成，本轮不声称已经解决：Base Event 到 EventInstance 没有完整 Mention 映射；部分角色仍只在属性里；KGQA 仍是词项 BM25；没有调用 Qwen；没有真实 Dense/Hybrid；没有人工 Gold，因此没有 Precision、Recall、F1。

## v1.1.7 收口

v1.1.6 审计里仍会让已验证关系在下游失效或被重新解释的五项，这一轮按代码路径修了：

- 验证记录增加身份指纹、反证编号和来源位置。方向变化或反证进入后，不能继续沿用旧的 VERIFIED。预测中的端点返回 FORECASTS。
- `reasoning_chains()` 的正式结果只保留每一步为 VERIFIED 或 TEMPORAL_VERIFIED 的链。无证据路径在 `candidate_chains`，争议和反转在 `rejected_chains`。
- `abstract_layer()` 不再把缺失的 `verification_status`（`None`）提前跳过。支持数改为证据文档数。
- Base 的 `accept_causal()` / `normalize_event_relation()` 使用 EventKG 的触发词家族。抑制句不能保留成导致或促进。
- `_score_edge()` 在打分之后调用 `verify_relation()`。验证结果为 FORECASTS、DISPUTED 或非 VERIFIED 时，不能维持已确认因果。图的复杂度和跨文档时间门槛仍留在评分器里。

仍未做完、也没有在这一轮重写的部分：Base 事件和 EventKG 实例之间还没有显式 Mention 映射边；角色还没有全部导出成可遍历的实体关系；Dense / Hybrid 没有真实向量实验。默认 `legacy` 运行不会执行 Hybrid 召回。

## v1.1.6 审计原文

日期：2026-11-12。本文记录审计时看到的实际行为和本轮修改。单元测试通过不等于整条链路没有剩余问题。没有调用 Qwen，没有人工 Gold，没有真实 Embedding 服务。文中的测试计数是功能回归，不是关系 Precision、Recall 或 F1，也不是 Dense / Hybrid 的召回提升。

## 审计结论

同一套判断原先分散在身份合并、趋势改写、篇内关系、跨文档打分和路径确认里。严格检查会被另一条路径绕过。本轮把合并收成一个函数，把趋势按 Mention 读取，把因果触发词按关系类型分开，并让抽象层使用同一份验证结果。溯因打分和路径验证仍然是两段代码，触发词家族已经对齐，但 `_score_edge()` 还没有改成直接调用 `verify_relation()`。

## 身份与时间

| 位置 | 审计时的行为 | 问题 | 影响 | 本轮结果 |
| --- | --- | --- | --- | --- |
| `identity.same_instance` / `coreferent` | 按商品、指标、时间比较 | 提纲合并不走这里 | 时间未知或口径不同的事件被并进代表事件 | 所有合并入口改走 `can_merge_event_instances()` |
| `link._merge_outline` | 原文互相包含就合并 | 能绕过 `same_instance()` | 市场综述被并进 47港港口库存 | 包含关系只提出候选，合并仍要过身份函数 |
| `validate.time_bucket` | 空字符串落到本期 | 未知被当成已知 | 没有时间的事件与本期事件合并 | 空时间返回 UNKNOWN |
| `validate.trend_name` / `stamp_instance` | UNKNOWN 或「当期」写进名称和观测期 | 报告日期和观测期混用 | 查询把发布时间当成事件时间 | 名称不再加「本期」；观测期不从「当期」抄入 |
| `validate.clean_event` | 商品缺失时可能取文档第一品类 | 静默填入 | 无关事件挂到标题品类 | 只有唯一明确匹配才分配商品；未知维度是 UNKNOWN |

最小反例：4月17日与 4月24日的同一指标保持两条；一边没有口径不能并进 47港；62%澳粉与 65%铁精粉不合并。回归在 `tests/test_v116.py` 和 `tests/test_eventkg.py`。

## 趋势与模态

| 位置 | 审计时的行为 | 问题 | 本轮结果 |
| --- | --- | --- | --- |
| `trends.interpret_trend` | 按分句和指标读取方向、动量 | 「预期」曾把「成本预期」标成预测 | 预测词改为「预计 / 预计将 / 展望 / 或将 / 料将」 |
| `commodity.canonical_mention` | 句末趋势词被改成上升或下降 | 「收窄」「扩大」会改方向 | 这两个词不再当作水平方向 |
| `prompts.EXTRACT_SYSTEM` | 一段里的「预计」容易写成所有事件的状态 | 已发生原因和预测结果混在一起 | 提示改为只约束所在分句，并区分「成本预期」 |

「上涨且涨幅收窄」保持上升、动量收窄。同比与环比是同一观测的两条信号，不是冲突。百分号按位置区分品位和涨跌幅。

## 关系验证

| 位置 | 审计时的行为 | 问题 | 本轮结果 |
| --- | --- | --- | --- |
| `support.verify_relation` | 路径和问答使用 | 与溯因打分不一致；「库存」别名过宽 | 社会库存、总库存、厂库不再确认港口库存；带日期的实例必须对上日期；结果侧预计记为 FORECASTS；支持与同期否定保持 DISPUTED |
| `graph_induction._score_edge` | 另一套确认条件 | 上游确认的边可能在路径阶段被降级，或反过来 | 触发词家族与 `verify_relation` 对齐。确认本身仍在 `_score_edge()`，这是剩余缺口 |
| `event_ontology.accept_causal` | 字符片段覆盖，触发词可跨分句借用 | 「发运下降导致到港下降，铁水上升推高矿价上涨」会接受发运到矿价 | 论元必须整词落在同一分句，触发词必须连接这两个论元 |
| `lexicon` 导致触发词 | 随着、综上、叠加也能触发导致 | 连接词被当成因果 | 这些词不能单独确认直接因果 |
| `link.merge` | 促进、导致、抑制、拖累、必要条件收成一个因果族 | 不同类型的证据被合并 | 关系类型保持分开 |
| `paths.allows_causal_walk` | 必要条件可以走进普通因果路径 | 条件被当成已发生因果 | 必要条件不再作为普通因果边传播 |
| `link.abstract_layer` | 在 Catalog 之前提升全部关系，按记录条数计数 | 未验证的综合推断进入抽象机制 | 改到 Catalog 之后，只提升 `verify_relation()` 接受的关系，支持数按独立文档计 |

跨分句反例只接受「发运下降 → 到港下降」和「铁水上升 → 矿价上涨」。

## 检索

| 位置 | 审计时的行为 | 问题 | 本轮结果 |
| --- | --- | --- | --- |
| `retrieval.tokenize` | 中文按单字切 | 词项检索没有词 | 改为 Jieba，并加入大宗商品词 |
| `retrieval._ranks` / `_rrf` | 零分也得到名次 | 没有命中的候选进入 RRF | 零分不排名，RRF 只累计真正召回的通道 |
| `graph_induction.retrieve_neighborhood` | 超过 42 天直接丢弃，分数不参与最终排序 | 时间未知或更早的证据被丢掉 | 42 天只作衰减说明；排序使用检索分数；同一文档和同一指标不能占满候选 |
| `embeddings` | 已有 OpenAI 兼容和 SentenceTransformer 接口 | 缺模型时如果回退会假装有向量 | 缺模型直接失败 |
| `retrieve.seeds` | 仍以字段别名找种子 | 问答没有完全复用 Hybrid | 路径扩展不再用全局 seen 丢掉替代路径。种子检索仍主要靠字段，这是剩余缺口 |

没有真实 Embedding 模型，本轮只做了词项检索和 RRF 的功能测试。

## 来源、序列和实验记录

- 合并后的 Mention 仍保留各自的 `document_id`。Evidence Catalog 在原文只出现一次时记录字符位置；出现多次时留空，不再用第一次 `find` 猜测。
- 周刊缺中间一期时，间隔大于 10 天、不超过 42 天记为 `NEXT_AVAILABLE_REPORT`，不写成连续下一期。预测验证窗口也是 10 天。
- `tools/run_saved_events.py` 读取全部十篇原文，包括没有抽出事件的文档。`manifest.json` 增加事件快照哈希、语料哈希、代码版本、Prompt 版本、检索方法、Embedding 模型、分词版本、词典哈希和 RRF k。不记录 API Key。
- `experiment_sheet.py` 在缺少证据类型时标 `UNCLASSIFIED`，不再自动归为 `implicit_without_trigger`。

## 本轮测试

- `python -m unittest discover -s tests`：147 通过，0 失败。
- `python tests/manual/eventkg_ten_check.py`：关系解析 10/10，Mention 对齐 4/4，Relation Evidence Precision 11/11。
- FakeChat 十篇离线流程能跑完。打印的实例数、边数和链数只说明流程，不是准确率。传导链为 0，因为抽象边现在只继承已经验证的关系。

## 仍未解决

- 溯因确认仍由 `graph_induction._score_edge()` 完成，路径和问答由 `support.verify_relation()` 完成。两边的触发词和极性已经分开，但还不是同一次函数调用。因此仍可能出现溯因接受、路径不接受的边。
- 问答种子 `retrieve.seeds()` 还没有改成调用 Hybrid / BM25 排序，主要仍按字段和别名匹配。
- Base 事件和 EventKG 实例保持两个层级，没有新增强制 UID 合并，也还没有把每条 Base 事件写成一条显式的 Mention 映射边。
- 角色（企业、装置、品种、地区、物流起终点）还没有全部导出成独立图关系。scope 不会被写成 location。
- 原始结构里的 page、block、bbox 只有上游提供时才保留。无法定位的字符偏移留空。
- 没有人工 Gold，不能报告关系 Precision、Recall、F1。没有 Embedding 服务，不能报告 Dense 或 Hybrid 的真实召回变化。没有调用 Qwen，不能报告模型效果。

## 是否具备真实 Qwen 实验条件

代码和运行说明可以接到 `QWEN_BASE_URL=http://172.20.20.24:8003/v1`、模型 `qwen3.8-27b`。本轮没有发起这次调用。Hybrid 还需要单独配置 `EVENTKG_EMBEDDING`。没有这两项，就不能把本轮测试数字写成实验结论。
