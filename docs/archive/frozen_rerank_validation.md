# 冻结候选池 × 重排查询：四臂真实开发验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。四臂完整 41 轮均成功：164 条逻辑结果、90 次新重排、4500 对分类，0 token-cap hit。短查询重排显著恢复了原作者长 rewrite 在这一候选池上的排名损失，但**尚未明确优于低成本的原 user_topic**。最好的新臂 R@5 为 .55000，原 user_topic 为 .54872；新臂 nDCG@5 反而从 .51493 降至 .51044。本轮保留配置化的召回/重排分离能力；下一轮 B6 共同 A 先使用低成本 user_topic，不据此新增全 41 轮生成 pilot，也不改变默认模式。

## 真实执行与独立绑定

原始数据：`runs/remote-h200-frozen-rerank/runs/frozen-rerank-v1/`。冻结输入在 `runs/frozen-rerank-prep/frozen-v1/`；独立脚本与完整逐轮结果在 `runs/frozen-rerank-review/analyze.py`、`analysis.json`。本轮没有重新调用 ConvSearch、selector、BM25 或 dense encoder，只加载相同语料正文映射与一份 BGE reranker。

- 164 行的 task、召回池、评分查询与冻结 case 逐字段一致；90 份实际分类分数逐项重建全部 50 段排序，结果完全一致。74 个逻辑重复行明确共享原执行，没有独立时延测量。
- 部署源 `955ff92124f15e095741e3a6d3e51eb0c925ef3c` 的 304 项 Git blob 映射与导出部署记录一致；这是 tracked 文件绑定，不把远端 checkout HEAD 当作已断言事实。导回 helper 与本地 SHA `2917542bcf2fb80ba4920f507dbc87f541562f206dcfd7b50d699c9130436f23` 相同。
- 与前次 `20fef31` 相比，`DenseIndex` 和 `LocalBGE` 的实现 AST 相同；corpus、固定 reranker revision、BF16/GPU0/batch16/1024 cap 与依赖版本均一致。原本传入长 rewrite 的位置，现在传入预先固定的 current 或 user_topic；没有扩大候选数量或输出上下文。
- 远端没有重复导出较大的 baselines/prior-reranks 副本。执行 manifest 所绑定的三个 JSONL SHA 与本地冻结文件一致，导回 cases 也逐字相等；本地三份已保存 baseline 用于独立复算，没有重新生成它们或改写旧分数。
- 所有新 attempt 已落盘、模型已释放之后才读取固定 Cloud dev qrels。独立 Recall/nDCG 公式与原 summary 的逐 task 值全部一致。每臂执行 41 轮，计分为相同 39 轮；Carthage 会话 `c01c8cf11437e6bb3bc93efac26528c2` 的 t6/t7 仍缺标签。未读 test，未生成答案。

候选池有两种：原 ConvSearch 单路 RRF 前 50，或 current+ConvSearch 的四份原始 BM25/dense 排名等权 RRF 后取总 50。每种池分别使用原 current 与确定性 user_topic 评分，共四臂。全部定义在新推理之前冻结，不按 task 或标签选择评分查询。旧 current、旧 user_topic 单路、原 ConvSearch 长查询重排三个完整 41 轮对照继续保留。

## 对强简单基线的比较

下表均为 39 个带标注问题的逐轮均值。“原”行来自前次已保存的相同语料/模型排名；本轮不把它们的运行时间与新计算比较。

| 召回池 / 重排查询 | 候选 R@50 | R@5 | nDCG@5 |
|---|---:|---:|---:|
| 原 current / current | .76068 | .52308 | .48327 |
| 原 user_topic / user_topic | .74786 | .54872 | .51493 |
| 原 ConvSearch / 长 rewrite | .81026 | .27009 | .26741 |
| ConvSearch 单路 / current | .81026 | .51581 | .47345 |
| ConvSearch 单路 / user_topic | .81026 | .55000 | .51044 |
| current+ConvSearch 融合 / current | .82265 | .51453 | .47052 |
| current+ConvSearch 融合 / user_topic | .82265 | .53590 | .50211 |

单路/user_topic 相对原 user_topic 的 R@5 仅高 **0.128 个百分点**，nDCG@5 低 **0.449 个百分点**。它的 39 轮召回为 3 升、32 同、4 降；nDCG 为 6 升、29 同、4 降。R@5 小幅正差不等于大多数问题改善，也不能掩盖退化。其 top5 ID 及顺序仅 18/41 与原 user_topic 完全相同，其他材料变化可能影响生成，却没有本轮答案数据证明影响方向。

| 新臂相对原 user_topic | R@5 升 / 同 / 降（39） | nDCG 升 / 同 / 降（39） | top5 完全相同（41） |
|---|---|---|---:|
| 单路 / current | 3 / 30 / 6 | 6 / 27 / 6 | 17 |
| 单路 / user_topic | 3 / 32 / 4 | 6 / 29 / 4 | 18 |
| 融合 / current | 2 / 32 / 5 | 2 / 32 / 5 | 26 |
| 融合 / user_topic | 1 / 35 / 3 | 1 / 35 / 3 | 27 |

会话级 R@5 如下，列名为 ID 前缀。标注轮数依次为 8、8、5、9、9；这是会话内逐轮平均，没有把 39 轮当作 39 个独立会话。

| 方法 | 1c041 | b4614 | c01c8 | c6c3b | d5b1e |
|---|---:|---:|---:|---:|---:|
| 原 user_topic | .6125 | .5417 | .8667 | .4259 | .4444 |
| 单路 / current | .5563 | .5417 | .8667 | .4259 | .3519 |
| 单路 / user_topic | .5563 | .5417 | .8667 | .4259 | .5000 |
| 融合 / current | .5500 | .5417 | .8667 | .4259 | .3519 |
| 融合 / user_topic | .5500 | .4792 | .8667 | .4259 | .5000 |

单路/user_topic 的平均改善主要来自 Cloudant 会话，另一会话下降；融合池虽有更高 R@50，也没有自动带来更高 top5。

## 具体进退与来源边界

1. **有价值的补召回：`d5…t3`，图片/PDF 与文档。** 原 user_topic 的候选 50 没有标注段 `ibmcld_00510-7123-9213`；新四臂都将它排到 rank4，R@5 从 0 到 1。其正文明确讨论 Cloudant 文档附件及应何时使用 Object Storage，这是可定位的材料改善。新 top5 仍有其他产品/接口材料，不能据此宣称答案必然更好。
2. **候选损失无法由短查询补救：`1c041…t2`。** 原 user_topic 把 `ibmcld_07086-7-2167` 排在 rank4；新单路及融合候选 50 均没有它，R@5 从 .5 到 0。加入 current 路并不保证保留原路的全部前 50，等权融合后仍可能被挤出。
3. **相同对象也有 top5 边界竞争：`b4614…t2`，WebSocket。** 原 user_topic 的两篇标注段为 rank3/6；单路短查询只保留一篇至 rank8，融合池包含两篇却为 rank6/9，R@5 均为 0。这里即使池召回补齐，也没有恢复前五。
4. **customization ID 的局部机会与退化：`b4614…t4`。** 原 user_topic 没有召回 `ibmcld_13386-1713-3820`；单路/current 与单路/user_topic 分别排到 rank4/5，R@5 从 0 到 .5。融合/user_topic 中它变为 rank6，收益消失。不能把“候选更多”当成排序更好。
5. **Cloudant 首轮的损失：`d5…t1`。** 原 user_topic 两篇标注材料都在 top5；新四臂候选 50 均漏掉 `ibmcld_00474-7885-8455`，只把另一篇 `ibmcld_00513-7-2197` 排到 rank4，R@5 从 1 到 .5。不是只有后续长历史才会出现退化。
6. **SDK 必须保留实体视图：`d5…t9`。** 同一 ConvSearch 池里，current 评分把两篇 Python SDK 标注段排到 rank11/18，R@5 为 0；user_topic 评分为 rank2/1，恢复至 1，与原简单 user_topic 持平。短 current 查询本身也会缺少对象信息，不应一律替代解析查询。

此前关注的 Cloudant 大小限制仍不能只看 qrels：`d5…t4` 的标注段在单路短查询下从原作者重排 rank22 提至 rank8，仍未进 top5；top5 混有存储卷、Discovery 上传，以及分别指向不同对象的 1MB/64MB 文本，尚不足以断言附件条件已清晰。`d5…t5` 的单路/user_topic 确实把直接列附件限制的 `ibmcld_00558-23465-25360`、`ibmcld_12904-23555-25450` 放入 rank3/4；标注段为 rank6，所以该轮 R@5 仍为 0，且 rank5 还有 Discovery PDF 材料。局部来源更直接与整体答案可靠是两个待分别检验的事实。

所有新评分输入均未触及 1024 cap，原作者重排的 325 个 ConvSearch cap hit 因更换查询不再出现。但查询的语义内容和长度同时改变，不能把这批改善全部归因于去掉截断。新四臂保留相同已生成 rewrite 的候选池，也没有消除旧助手错误进入召回查询的风险。

## 本阶段结论与保留的能力

这次真实验收确认，已有的候选池可通过独立评分查询恢复有用排名，工程接口 `rerank_query_mode=resolved/current/user_topic` 可继续用于后续明确场景；默认 `resolved` 没有改变。原简单 user_topic 在两项总体指标上与最佳新臂接近，且省去 3B 完整历史 CQR 的额外调用。因此下一 B6 对照共同使用低成本 user_topic 是当前证据下合理的资源取舍，所有被比较的控制策略应共享相同 A 设置。

本轮不启动额外全 41 轮答案 pilot，不把局部 Cloudant 改善选成新的成功案例子集。仍保留后续研究空间：在明确的省略实体场景中比较精简的、来源支持的对象信息与原问题联合评分，或者检验候选配额如何减少融合时挤出；这些都需事先固定完整 cohort/对照，再评估材料和答案。未进行的新实验不计为实现收益。

这里只有五个开发会话、一次冻结历史的排名实验，旧输出与新评分之间复用了真实计算；没有新 P3 历史传播、用户答案、人工分数或在线时延对照。不报告质量准入、显著性、独立 test 优势或净系统性能收益。
