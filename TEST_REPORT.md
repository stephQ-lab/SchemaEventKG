# 测试记录

日期：2026-11-14。下面的数字是规则核对和 FakeChat 功能测试，不是 Qwen 准确率，也没有调用线上模型。十篇规则核对的 10/10、4/4、11/11 只覆盖事先选定的小案例。本轮没有配置 Embedding 服务，不报告 Dense 或 Hybrid 的真实召回变化。

## v1.1.9

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：182 个，通过 182，失败 0。十篇规则核对：关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。v1.1.2 构造比例仍是 6/6、3/3、0/4、3/3、2/2。

`tests/test_v119.py` 覆盖目标分句的预测状态、口径和主体、跨文档泛化句、Mention 与角色指纹、Legacy 候选、Hybrid RRF、KGQA 检索方法、导出候选边、宏观事件和空 Gold。验收说明见 `ACCEPTANCE.md`。

离线十篇 FakeChat 打印：实例 395，变量 48，抽象事件 74，`cross_doc_edges` 243，推理链 0，局部链 0，传导链 0，Cypher 节点 1083，边 3512。实例数比 v1.1.8 增加，是因为全国和 247家钢厂不再被收成未知口径。这不是准确率。真实 Qwen 与真实 Dense/Hybrid 均为 `NOT_RUN`。

## v1.1.8

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：173 个，通过 173，失败 0。十篇规则核对：关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。v1.1.2 构造比例仍是 6/6、3/3、0/4、3/3、2/2。

`tests/test_v118.py` 覆盖本轮反例：近期与本周、上月与本周不合并；下月不是本期；没有方向的「发运导致到港」不确认具体下降；本期事件不绑定更早日期；4 月 17 日支持和 4 月 24 日支持加 4 月 24 日反证时，4 月 24 日为 DISPUTED；拖累和推高保持原关系；不同报告系列不生成延续或兑现；无时期观测不绑定本周事件；社会库存原文可以验证社会库存；局部显式关系不被弱词项结果挤掉；两篇相同原文分别保留来源和日期。

这些都是规则回归，不是模型准确率。离线十篇 FakeChat 仍打印：实例 387，变量 48，抽象事件 74，`cross_doc_edges` 238，推理链 0，局部链 0，传导链 0，Cypher 节点 1063，边 3389。这不是准确率。问答仍使用词项 BM25，本轮没有接入真实 Hybrid。

## v1.1.7

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：162 个，通过 162，失败 0。十篇规则核对：关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

`tests/test_v117.py` 检查模块之间的数据契约：方向改变后旧 VERIFIED 失效；正证加反证保持 DISPUTED；预测事件不是已发生因果；没有证据的路径只进候选链；已验证链的每一步都是 VERIFIED；抽象层在缺少验证状态时仍会验证，并按证据文档计数；Base 的导致和抑制不再互换；整篇综合保留反向关系和不同类型；宏观事件不继承报告品类；同比冲突不能合并；停产、检修、政策分开；上期不对齐本周；缺周不是下一期；重复原文位置为 AMBIGUOUS；「矿价」能召回价格事件。

这些都是规则和流程回归，不是模型准确率。

离线十篇 FakeChat 打印：实例 387，变量 48，抽象事件 74，`cross_doc_edges` 238，推理链 0，局部链 0，传导链 0，Cypher 节点 1063，边 3389。推理链为 0，是因为没有证据编号或没有通过统一验证的连续路径不再计入正式推理链，它们留在候选链里。这不是准确率。

## v1.1.6

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：147 个，通过 147，失败 0。十篇规则核对：关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。这些是规则回归，不是 Qwen 准确率，也不是人工 Gold 上的 F1。

`tests/test_v116.py` 覆盖本轮反例：提纲不能绕过身份检查、不同日期不合并、空时间是 UNKNOWN、62% 与 65%、47港与 45港、社会库存不能确认港口库存、泛化句不能确认带日期的实例、结果侧预计是 FORECASTS、支持加否定保持 DISPUTED、随着不能单独确认、跨分句不能借用触发词、必要条件不是普通因果传播、缺一周是 NEXT_AVAILABLE_REPORT、涨幅收窄仍是上升、成本预期不是预测、零分不进 RRF、Jieba 保留「港口库存」、45港不对齐 47港、超过 42 天仍可召回。

离线十篇 FakeChat 打印：实例 387、变量 48、抽象事件 74、`cross_doc_edges=238`、推理链 4、局部链 0、传导链 0、Cypher 节点 1067、边 3406。传导链为 0，是因为抽象边改为只继承已经通过统一验证的关系。`cross_doc_edges` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是已确认因果。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。Jieba BM25 是词项检索，不是语义向量检索。

## v1.1.5

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：134 个，通过 134，失败 0。十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。v1.1.2 的十篇原句回归仍是趋势 6/6、比较维度 3/3、错误合并 0/4、关系证据 3/3、来源追溯 2/2。这些都是规则回归，不是 Qwen 准确率。

`tests/test_v115.py` 复现了 v1.1.4 审查里失败的输入：

- PVC 华东港口库存不能由「铁矿石港口库存下降」确认。发运下降的图谱事件不能由「发运上升」确认。结果侧「没有下降」不确认。
- 同一时期的支持句加明确否定句得到 DISPUTED，反证编号保留。下周的否定不否定本周这条关系。
- 「发运量下降，导致到港量下降」可以确认。促进边不能由只写「导致」的原句确认。
- 2026-04-17 与 2026-04-24 的库存下降经过 clean_event 后不合并。2026-04-17 与 2026年4月17日是同一天。
- 「本周库存较上周下降」读出下降、环比，比较基期不是事件时间。65%铁精粉保留。
- 篇内「发运量下降导致到港量下降」写入 `ev:rel:A:B:0`，经统一 Catalog 进入已确认因果路径。综合推断不会标成 ASSERTED_CAUSE。

离线十篇 FakeChat 这次打印：实例 395、变量 48、抽象事件 74、`cross_doc_edges=267`、推理链 5、局部链 0、传导链 4、Cypher 节点 1089、边 3526。推理链从 v1.1.4 的 8 条变为 5 条，是因为篇内关系改走同一套实例验证，不再把未核对的边算进链。`cross_doc_edges` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是已确认因果。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。BM25 仍是中文单字匹配。

## v1.1.4

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：119 个，通过 119，失败 0。十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。v1.1.2 的十篇原句回归仍是趋势 6/6、比较维度 3/3、错误合并 0/4、关系证据 3/3、来源追溯 2/2。

`tests/test_v114.py` 复现了 v1.1.3 审查里失败的输入，仍是规则回归，不是 Qwen 准确率：

- 本周价格上涨和预计下周价格下跌分成两个事件，方向分别是上升、已发生和下降、预测中。顿号连接的「环比上涨、同比下降」保持两个比较基期。「没有明显上升，仍然下降」恢复为下降。
- `价格上涨1.65%`、`开工率下降17.93%` 去掉变化率。`65%精粉` 和 `62%澳粉` 保留。1.65% 不再写进 commodity。
- 「A 导致 B，C 推高 D」不确认 A→D。预计、可能、未导致，以及两条方向相反的证据，都不确认。促进、挤压、由于、受影响可以按对应关系确认。
- 环比上涨加同比下降的事件，可以并入只带同比下降的同一次观测；环比方向相反则不合并。本周事件可以连到报告日期相同的表格趋势，事件时间仍是本周。缺证据的事态延续不会让混合路径变成已确认。推理链在传入 Evidence Catalog 后评分不再是 0。

离线十篇 FakeChat 这次打印：实例 395、变量 48、抽象事件 74、`cross_doc_edges=267`、推理链 8、局部链 0、传导链 4、Cypher 节点 1092、边 3543。实例数相对 v1.1.3 的 406 有变化，是因为时间和比较基期不再并进同一次事件。`cross_doc_edges` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是已确认因果。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。BM25 仍是中文单字匹配。

## v1.1.3

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：111 个，通过 111，失败 0。十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

`tests/test_v112.py` 在十篇真实表格和原句上的规则回归保持：趋势抽取 6/6、比较维度 3/3、EventInstance 错误合并 0/4、关系证据 3/3、跨文档来源追溯 2/2。其中比较维度改用 6 月 12 日周报里「高炉炼铁产能利用率」这一分句，不再把同页其他指标的环比混进同一次判断。

`tests/test_v113.py` 复现了审查时失败的输入，仍是规则回归，不是 Qwen 准确率：

- 目标趋势：利润下降不会被同句价格上涨改成上升；「并未上涨」「没有明显上升，仍然下降」不会被认成上升；「展望下周」标成预测中；「本周上涨，预计下周继续上涨」仍是已发生。4 月 17 日周报原句「本周铁矿石价格持涨运行」经清洗后进入查询属性、Cypher 和 KGQA，查询节点不含行情数值。
- 实例身份：价格与现货价格、价格与 47港价格不再合并。同一港口库存的环比上涨和同比下降合并为一个实例，两条信号都保留，名称方向取环比。
- 趋势标签：行情变化量被去掉，对象、62%、47港和日期保留。没有对象的「上涨1.75美元/吨」不再变成节点。
- 关系证据：发运下降导致到港下降可以验证；方向相反、被否定、不确定，或两个 Mention 不在同一分句时不能验证。问答展示目录中的原句，不展示边上的过期摘要。

离线十篇 FakeChat 这次打印：实例 406、变量 48、抽象事件 74、`cross_doc_edges=282`、推理链 9、局部链 0、传导链 4、Cypher 节点 1116、边 3640。实例数相对 v1.1.2 的 409 有变化，是因为口径和指标不再按字符串包含合并。`cross_doc_edges` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是已确认因果。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。

## v1.1.2

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：103 个，通过 103，失败 0。十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

真实十篇表格和原句上的规则回归（`tests/test_v112.py`，不是 Qwen 准确率）：

- 趋势抽取正确率 6/6。其中 5 项直接使用研报原句或表格：持涨运行、降幅收窄、折扣收窄、裸价格 108.10、同比增加且环比方向。另 1 项是验收句「价格上涨但涨幅收窄」，十篇里没有这句原文，用的是和「降幅收窄」同一条动量规则。
- 比较维度正确率 3/3。6 月 12 日周报里「环比上周增加、同比去年减少」不是同一维度的冲突；4 月 17 日价格表的环比对上 4 月 17 日和 4 月 10 日；「较去年同期幅度」没有被确认成周对周比较。
- EventInstance 错误合并率 0/4。62% 与 65%、47港与 45港、上周与本周没有合并；价格数字不同不再单独阻止合并。
- 关系证据有效率 3/3。不存在的编号是 MISSING，无关原文是 UNVERIFIED，点名两端并带因果触发词的原文才是 VERIFIED。
- 跨文档来源追溯正确率 2/2。共指之后两份原文仍分别指向自己的 document_id。

离线十篇 FakeChat 这次打印：实例 409、变量 48、抽象事件 76、`cross_doc_edges=277`、推理链 10、局部链 0、传导链 2、Cypher 节点 1123、边 3663。`cross_doc_edges` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是 277 条已确认因果。节点数上升是因为每个实例另有一个证据审计节点。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。

## v1.1.1

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：102 个，通过 102，失败 0。十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

离线十篇 FakeChat 打印的实例数仍是 409。`cross_doc_edges=285` 仍是连续观测、预测验证和 Legacy 核验边的合计，不是 285 条已确认因果。推理链 9 条。图节点从 v1.1 的 795 变为 713，是因为观测不再按指标名整表回退链接。这些都不是准确率。

本轮没有配置向量服务，因此没有运行真实 Dense 或 Hybrid 对照。脚本编码器只验证了同一批事件文本不会被重复编码。

## v1.1

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：99 个，通过 99，失败 0。其中 8 个是本轮新增：检修角色、政策对象与生效时间、PVC 表头配对、无表头数字保持 AMBIGUOUS、独立证据的多原因共存、跨地区运输与无运输机制、抑制路径和同一指标的不同实例、构成关系不进入因果路径、问答的逐步证据以及无证据步不回答成确定因果。

十篇规则核对仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

离线十篇 FakeChat 这次打印的实例数仍是 409，跨文档边 285，推理链 10。推理链比 v1.0 多，是因为抑制可以留在路径里，同一指标的另一个实例也可以再进入。这不是准确率。

## v1.0

改造前先跑了当时的 v0.9：`python -m unittest discover -s tests` 为 84 个测试，全部通过；`python tests/manual/eventkg_ten_check.py` 仍是关系解析 10/10、Mention 对齐 4/4、Relation Evidence Precision 11/11。

改造后同一组命令：91 个测试，全部通过。十篇规则核对数字不变。

离线十篇 FakeChat 的实例数从合并过宽时的 266 变为 409。这只说明同一指标的不同表述不再被自动并成一条，不是准确率。

脚本编码器夹具上，正例「铁矿石价格上涨 / 铁矿石价格走强」的 Candidate Recall@10 在 Jaccard、BM25、Dense、Hybrid 四路都是 1.0。这是两条签名本身字面接近、向量又被测试写成接近的结果，不能外推到真实研报。

## 实验准备

日期：2026-11-10。下面的数字都是规则核对或 FakeChat 功能测试，不是 Qwen 准确率，也没有调用模型。构造切片 F1 不作为真实模型准确率。

## 实验准备

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：84 个，通过 84，失败 0。

`test_order_needs_one_sentence_and_a_cue` 已通过。之前/之后只在同一句同时盖住两端、并且这句里有时间线索时保留。因果和同时关系的检查没有改。回归测试 `test_order_cue_stays_inside_one_sentence_and_causal_checks_stay` 确认同一句里的“之后”仍保留，跨句的“之后”去掉，导致和同时的原有结果不变。

十篇原文规则核对：关系解析 10/10，Mention 对齐 4/4，Relation Evidence Precision 11/11。这是规则核对，不是 Qwen 准确率。

## 2026-11-09 分段关系与统一时间

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：79 个，通过 78，失败 1。

失败仍是原有的 `tests.test_eventkg.ExplicitLinkTest.test_order_needs_one_sentence_and_a_cue`。`normalize_event_relation` 对“发运增加之后到港增加。库存则另行统计。”返回 `之后`，测试期望 `None`。这次没有改这个函数。

本轮新增核对：

- 发运导致库存、库存抑制价格时，两条边分别保留导致和抑制，不生成发运到价格的直接边。
- 事件节点没有 event_time、对齐 Mention 有明确日期时，直接边和间接路径的时间基准都是 event_time，间隔天数相同。
- 默认搜索不展开两个中间事件；显式把上限设为 2 时才返回带各自关系类型的更长路径。

十篇原文规则核对：关系解析 10/10，Mention 对齐 4/4，Relation Evidence Precision 11/11。

## 上一轮隐式路径

命令：

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

单元测试：76 个，通过 75，失败 1。

失败仍是原有的 `tests.test_eventkg.ExplicitLinkTest.test_order_needs_one_sentence_and_a_cue`。`normalize_event_relation("发运增加", "之后", "库存下降", "发运增加之后到港增加。库存则另行统计。")` 返回 `之后`，测试期望 `None`。这次没有改这个函数。

新增核对：

- 2026-04-17 澳洲发运下降、2026-05-01 港区员工培训、2026-05-15 铁矿石价格上涨，即使同属铁矿石并带有历史抽象规律，也不确认因果，类别是 `POSSIBLE_INDIRECT_CAUSE`。
- 发运下降导致库存下降、库存下降导致价格上涨分别有独立证据时，保留这两条 `INFERRED_SUPPORTED`，不生成发运到价格的直接边。
- 只有时间先后的海运费不能确认发运到价格。
- 没有报告日期但有事件时间的发运可以进入召回，方向按事件时间。
- 报告日期与事件时间冲突时，方向使用事件时间。事件时间晚于结果时，即使原文有“导致”，也不确认。
- 两端时间都未知时候选保留，`time_basis` 为 `TEMPORAL_UNKNOWN`，不确认方向。

十篇原文规则核对没有改口径：关系解析 10/10，Mention 对齐 4/4，Relation Evidence Precision 11/11。

## 上一轮

## 单元测试

命令：

```text
python -m unittest discover -s tests
```

工作目录：`AutoSchemaKG_eventkg`。

结果：71 个测试，通过 70，失败 1。

失败用例：`tests.test_eventkg.ExplicitLinkTest.test_order_needs_one_sentence_and_a_cue`。

原因：`normalize_event_relation("发运增加", "之后", "库存下降", source)` 返回 `之后`，该测试期望 `None`。这是本次改动之前就存在的结果，这次没有改这个函数，也没有为了通过测试把顺序关系放宽。

本次新增或更新的关系范围、模态、缺期、事件时间和隐式路径测试都在通过的 70 个里面。

## 十篇原文规则核对

命令：

```text
python tests/manual/eventkg_ten_check.py
```

语料：`example/example_data/unified10` 的十篇报告。标注在 `tests/manual/eventkg_ten_check.py`。

| 指标 | 结果 |
| --- | --- |
| 关系解析正确率 | 1.000（10/10） |
| Mention 对齐准确率 | 1.000（4/4） |
| Relation Evidence Precision | 1.000（11/11） |

关系解析覆盖：价格下跌作为背景、采购作为中间事件、预期作为事件内容、否定阻断、检修导致产量、基本面与宏观共同压制、结果侧预期不升成已断言因果、焦炭挤压利润、飓风影响到港。

Mention 对齐：2026-04-17 的“下期”绑到 2026-04-24；2026-04-24 的下一份周刊是 2026-05-15，间隔超出周刊周期，保持未绑定；2026-06-12 的下一份是 2026-06-26，同样未绑定。PVC 日报的“上周”不绑到铁矿石周刊。这些 Mention 的 `event_time` 都是空的。

2026-04-17 报告里的“3月31日预计，2026年4月份……”没有被写成事件时间 `2026-04-17`。报告日期没有被复制成事件发生时间。

Relation Evidence Precision 只统计实际抽出的直接因果对。两条正确的空结果（否定句、结果侧预期）计在关系解析正确率里，不进入 precision 的分母。
