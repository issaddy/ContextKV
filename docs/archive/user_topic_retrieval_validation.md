# 用户主题补全：五视图真实检索开发对照

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24，固定五个 Cloud dev 会话、41 个查询，五种 query 视图 × 四个检索阶段共 **820 条结果全部完成**。独立重算与原 summary 完全一致。`user_topic/hybrid_rerank` 的 Recall@5 为 **54.87%**、nDCG@5 为 **51.49%**，高于 current 的 52.31% / 48.33%；主要变化是 Cloudant SDK 的两个实际支持段落进入前两名。但 WebSocket/customization ID 追问的已标注候选从融合第 9 位退至第 53 位，超出重排预算。结果支持继续完整 P3 对照，不证明回答质量改善或非劣。

原始数据在 `runs/remote-h200-user-topic/runs/cloud-dev-user-topic-retrieval-v1/`，四个导入文件经过运行方 SHA 核验。只读分析 `runs/user-topic-retrieval-review/analyze.py` 生成 `summary.json`、逐视图配对 `pairs.jsonl` 和保留原文的 `focus-source-review.jsonl`。仅使用原已冻结 dev.tsv，未读取 test 标签/答案。本文对三个补主题问题的来源适用性判断由 assistant 核查原文，不是人工评分或模型 judge。

```sh
.venv/bin/python runs/user-topic-retrieval-review/analyze.py
```

## 输入、模型和预算确实相同的部分

41 个任务与运行前 `runs/user-topic-dev-review/queries.jsonl` 逐项对应，820 条实际 query 均复现计划文本；164 条 user_topic 阶段 trace 均与计划一致。其 12 条选中 span 记录是三个问题在四个阶段重复导出，不是 12 次独立补全。37 轮使用 current，三轮附加用户原命名短语，一轮复用纠正 last3；两个含 CocoaPods 名字的纠正轮相对 correction_aware 回到 current。没有 assistant 历史进入检索视图，manifest 的 `history_audit` / `generated_history` 均为空。

语料为同一 72,442 篇原始 passage，sparse/dense corpus SHA 都是 `04fb31f7d4301aa2fd3d12c161cf06265a9bbf9f41e56116d90637f9d053afad`。BGE-M3 固定 revision `5617a9f61b028005a4858fdac845db406aefb181`，BGE-reranker-v2-m3 为 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，CUDA:0/BF16、batch16，encoder/reranker 上限分别 8192/1024。查询 encoder 与保存 dense 索引的模型、revision、pooling、长度设定相符。稀疏和 dense 各召回 50、RRF k=60、融合前 50 重排、最终前五。各视图重新召回自己的候选，因此是**同预算与同语料**，不是强行相同候选池。

205 次 query 编码、10,250 对 rerank 输入的 at-cap 计数均为 0，即本轮未触及配置 token 上限。只保存了模型累计计数，不能编造每个 view/pair 的原始 token 长度。205 个融合→重排输入顺序和候选集合全部通过；返回的 4,100 次 top5 passage 用例中，同 ID 的正文/title/source 始终一致。该正文核查针对返回结果间的一致性，原始索引身份来自固定 corpus SHA，没有声称在本地重新构建全部语料。

所有排序完成后才读 qrels。39 轮有标注；CocoaPods 会话第 6、7 轮缺失，明确排除相关性均值，但保留其 query、排名、原文和耗时，不填零。本地重算20组 top5/pool 指标均与原 summary 精确一致；同文本 query 的配对没有发现 top5 顺序差异。

## 二十组结果全部保留

数值为有标注查询宏平均百分数，R@50 是该阶段前 50 的召回；RRF 原始 union 可能大于 50。阶段耗时含一次 query 视图构造，但不含模型/索引加载和生成。

| 视图 | 阶段 | R@5 | nDCG@5 | R@50 | 耗时均值 ms |
|---|---|---:|---:|---:|---:|
| current | BM25 | 31.20 | 25.75 | 60.98 | 1.32 |
| current | dense | 38.85 | 35.45 | 71.37 | 62.91 |
| current | RRF | 38.97 | 36.22 | 76.07 | 64.62 |
| current | RRF+rerank | 52.31 | 48.33 | 76.07 | 290.59 |
| last3 | BM25 | 20.94 | 16.71 | 55.90 | 1.42 |
| last3 | dense | 29.62 | 23.88 | 58.29 | 55.41 |
| last3 | RRF | 24.66 | 23.88 | 64.32 | 57.16 |
| last3 | RRF+rerank | 32.44 | 27.56 | 64.32 | 247.14 |
| all_user | BM25 | 19.23 | 14.58 | 47.26 | 1.82 |
| all_user | dense | 24.06 | 20.30 | 51.45 | 57.85 |
| all_user | RRF | 23.38 | 20.12 | 60.90 | 59.99 |
| all_user | RRF+rerank | 30.94 | 26.63 | 60.90 | 283.75 |
| correction_aware | BM25 | 28.63 | 24.61 | 60.98 | 1.16 |
| correction_aware | dense | 40.56 | 37.25 | 73.08 | 30.24 |
| correction_aware | RRF | 37.26 | 35.56 | 76.07 | 31.72 |
| correction_aware | RRF+rerank | 52.31 | 48.03 | 76.07 | 153.40 |
| user_topic | BM25 | 29.91 | 25.43 | 60.98 | 1.22 |
| user_topic | dense | 39.70 | 36.97 | 72.65 | 32.01 |
| user_topic | RRF | 39.83 | 37.96 | 74.79 | 33.50 |
| user_topic | RRF+rerank | 54.87 | 51.49 | 74.79 | 158.45 |

相对 current，user_topic 的 top5 Recall 为 1 题改善、38 持平、0 退化；nDCG 为 2/37/0。相对 correction_aware，Recall 同为1/38/0，nDCG为4/35/0，其中另外两题是恢复 current 的 CocoaPods 排序，不能都算成新主题补全收益。**R@50 反而从76.07%降到74.79%**，这个候选召回退化没有被 top5 均值掩盖。

all_user/last3 并非没有价值。相对 last3，user_topic 的 Recall 改善/持平/退化为17/18/4；相对 all_user 为19/18/2。附件纠正第5轮的唯一标注段，all_user 将其排入第3位，current、last3、correction_aware、user_topic 均未进前五；然而现有纠正视图已经返回真实限额表，因此标注 Recall=0 也不能断言材料不能回答上限问题。正式选择必须同时观察原文可用性与完整生成。

最终重排的会话 R@5 如下（会话 ID 前缀仅用于展示）：

| 会话 | 标注轮数 | current | last3 | all_user | correction | user_topic |
|---|---:|---:|---:|---:|---:|---:|
| 1c04 | 8 | 61.25 | 20.62 | 30.00 | 61.25 | 61.25 |
| b461 | 8 | 54.17 | 29.17 | 4.17 | 54.17 | 54.17 |
| c01c | 5 | 86.67 | 93.33 | 86.67 | 86.67 | 86.67 |
| c6c3 | 9 | 42.59 | 11.11 | 18.52 | 42.59 | 42.59 |
| d5b1 | 9 | 33.33 | 33.33 | 37.04 | 33.33 | 44.44 |

## 三个补主题问题的原文适用性

**d5 第9轮：真实 SDK 来源进入前两名。** 用户历史第6轮原文中的 `IBM Cloudant` 被带入当前 Python SDK 请求。current/correction 的前五为通用 SDK handbook、Discovery、Continuous Delivery；其中 `ibmcld_07128-4280-5113` 明确是 Discovery，`00690-1043-2339` 明确是 toolchains/Tekton。last3 的前五偏向 Speech to Text/Lite 套餐，all_user 偏向 Cloudant 套餐、附件/文档；仅拼全部历史虽保留产品名，仍被旧问题牵引。

user_topic 前两段 `ibmcld_00462-7-1647` 和 `00462-1307-2903` 明确写 IBM Cloudant SDK for Python，提供固定快照中的 `ibmcloudant>=0.0.27` 安装示例，后者还给出 `https://github.com/IBM/cloudant-python-sdk`。这是核对原文产品和任务内容后的改善，不只因为它们恰好带有 qrels。其余三段仍有 Cloudant HTTP/文档基础介绍及一个 Node.js 应用教程（同时链接其他语言 SDK），不能把五段都当成 Python 操作示例或声称具体 push/fetch 代码已验证。

current 原池其实已有两个正例：RRF第25/29，重排后第23/18；user_topic 将它们移至 RRF第4/1、重排第1/2，Recall/nDCG均0→1。这里同时改变了召回顺序与重排 query，不是“只换reranker”的效果；生成阶段是否会正确选择 SDK、版本与代码还要通过真实 P3。

**d5 第7轮：产品一致性改善，剩余标注仍未全覆盖。** current/correction 排第1的 `ibmcld_01183-7-2154` 是 **Event Streams** 套餐，曾诱发三可用区适用性误用。user_topic 用 `ibmcld_00612-7-2163` 的 Cloudant migration/throughput 说明替换它，并把 Cloudant Lite 限制段排至第1；新前五均明确讨论 Cloudant。R@5仍为2/3，nDCG由0.41618升至0.65082；第三个标注段 `ibmcld_00558-2925-4840` 在融合第3但仍被重排到第13。`00612` 虽未标注，却直接给出 Lite/Standard 容量说明，不能因未标注就算错误。last3/all_user 也返回相关 Cloudant 资料，只是更易受旧附件/文档问题影响，标注顺序与覆盖不同。

**b461 第4轮：接口名不等于已消解的产品/对象。** 带入的是第3轮 `WebSocket API`，没有额外服务实体。current/correction 的前五混合 Text to Speech 定制模型 GUID 及 Speech to Text acoustic model 信息；last3/all_user 带来电话集成、语音识别/合成配置资料。user_topic 前五更集中于 Speech to Text 的 acoustic/language customization 参数、WebSocket 调用和语法设置；这些是相关使用材料，其中 `ibmcld_13446-7-2012` 确有 acoustic customization ID 解释，不能把零标注召回解释成全部语义无关。但它仍没有稳定区分问题究竟问哪类 customization ID，也未建立完整答案支持。

明确的负面是 `ibmcld_13386-1713-3820`：current/correction 融合第9、重排第6，user_topic 融合第53，被50预算裁掉；另一个正例也未入池。user_topic的 pool Recall 从0.5降到0，所有视图的该题top5 Recall仍为0。这显示词法接口词补全可能把材料推向“如何使用接口”，而丢掉更直接的定义候选。保留原始池和这个失败，不按已知正例反向写在线专名例外。

## 成本与解释边界

query 视图构造的 mean wall time 分别为 current/last3/all_user 约0.004ms、correction 0.013ms、user_topic 0.053ms。user_topic 的纯重排均值124.95ms、全链均值158.45ms，correction分别121.68/153.40ms；这是一次固定顺序测量，不是额外5ms的可靠因果估计。

每题顺序固定为 current→last3→all_user→correction→user_topic，模型与索引只加载一次；首个 current 全链1217.75ms（dense阶段636.36ms），未单独随机化或重复视图顺序。故不能从 current 全链均值290.59ms与user_topic158.45ms推算“主题补全加速45%”。阶段耗时包括共享的中间工作，不能将BM25、dense、RRF、重排四行直接相加。分析脚本已在配对阶段差值中扣除重复的 query-view 项。

运行方记录 Qwen 大池服务同 GPU 共驻但 idle、Selene已停止；这不是 c4 生成压力下的检索时延，也没有完整 P3 生成/控制器/会话时间。当前模式选择仍使用已观察 dev，五个会话不足以把这个增量当作独立泛化结果；继续开发不受此限制，但最终 test 保持冻结。

## 下一步

在相同可选 focus 样式、同一生成模型/预算和资源下比较 correction_aware 与 user_topic 的完整 P3，保留全41轮，而非只重跑 SDK 成功例。优先核 SDK 实际产品/版本/代码来源、Lite/Standard 条件与 b461 customization ID 的解释完整性；同时检查两个 CocoaPods 变更轮及未触发轮。若生成仍不正确，继续改进意图/来源适用性和事实绑定，不能以检索指标抵销回答错误。保留 all_user/last3 的局部优势作为进一步选择性历史或双路径预算分配的依据，不将当前词法方案视为终点。

另已完成不改在线代码的[双路冻结池诊断与近期方法核查](../research/query_fusion_research.md)：当前query与topic的四种预设融合/配额策略都在总50篇池内找回b461t4定义正例。该结果只涉及候选覆盖，因新query-document对缺少真实logits，未构造新top5分数；后续可以用已有BGE接口补测。
