# v1.1.9 端到端验收

日期：2026-11-14。本文件记录规则回归和配置检查。FakeChat 计数、单元测试和十篇规则核对都不是 Qwen 准确率，也不是人工 Gold 上的 Precision、Recall 或 F1。

## 已核对

| 检查 | 结果 |
| --- | --- |
| 同一句里钢厂利润与下周铁矿石价格分开 | 利润保持已发生；价格是下周预测 |
| 近期与本周、4月17日与4月24日 | 不合并 |
| 247家钢厂、250家钢厂、未知口径 | 变量键不同 |
| 现货价格与期货价格 | 不合并 |
| 不同主体 | 不建立事态延续 |
| 机构未知 | 不生成 `商品:频率` 系列 |
| 另一篇没有日期、没有 Mention 的泛化句 | 不确认具体实例 |
| Mention 或角色实体变化 | 验证版本或身份指纹变化 |
| Legacy 无具体证据的成立判断 | `LEGACY_CANDIDATE`，不进入因果游走 |
| 导致、促进、抑制、拖累 | 保持原文类型；方向先验只记 `POLARITY_CONFLICT` |
| 未验证关系导出 | 标签为候选关系 |
| 宏观品类 | 不接收报告标题品类 |
| graph.json | 含 `trend_signals` 与 `mention_links` |
| BM25 | Jieba 词项，不输出单字 |
| Hybrid | RRF 分数进入排序；零分不进入结果 |
| KGQA | 跟随图上的 retrieval_method；未配置编码器时 hybrid 报错 |
| Qwen endpoint | 进入缓存键；API Key 不进入 |
| Legacy + Hybrid | `retrieval_status=NOT_USED` |

## 未实测

- 真实 Qwen：`NOT_RUN`
- 真实 Dense / Hybrid：`NOT_RUN`。代码路径用假想编码器测过维度和 RRF，没有可用的 Embedding 服务。
- 人工 Gold 指标全部为 null，见 `eventkg/gold.py`。

## 命令

```text
python -m unittest discover -s tests
python tests/manual/eventkg_ten_check.py
```

解压后的干净目录需已安装 `requirements.txt` 中的 jieba 与 json_repair。本环境使用的是已安装的真实包，不是兼容替身。
