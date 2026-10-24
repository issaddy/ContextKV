# Cloud dev：12 组检索结果与剩余误差

2026-10-24。本次固定查询开发实验中，`current/hybrid_rerank` 的平均 Recall@5 为 **52.31%**、nDCG@5 为 **48.33%**，均为这 12 组最高。相同 current 查询下，BM25、dense、RRF 的 Recall@5 分别为 31.20%、38.85%、38.97%。这说明增强检索值得接入真实 P3 路径；它没有生成答案，不证明答案正确率、引用支持度或质量非劣效。

证据来自 `runs/remote-h200-retrieval/runs/cloud-dev-hybrid-rrf50-rerank1024/{manifest.json,retrieval.jsonl,summary.json}`。本地复核脚本逐方法用原始排名和 **Cloud dev.tsv** 重算 Recall/nDCG，与原 summary 完全一致；没有读取 test qrels。记录包含 5 个会话、41 个查询、492 个方法结果。39 个查询有 qrels；`c01c8cf11437e6bb3bc93efac26528c2<::>6` 和 `...<::>7` 无标签，明确排除评分，不能填零。耗时仍包括全部 41 个查询。

## 固定实验范围

三个查询视角是当前用户问题 `current`、最近三个用户问题逐行拼接 `last3`、截至当前轮的全部用户问题 `all_user`。没有传入参考答案或生成的 assistant 历史；原始 manifest 中 `history_audit` 和 `generated_history` 均为空。检索执行完后才加载 qrels，标签只进入分析。

稀疏和 dense 使用完全相同的 72,442 篇原始 passage，corpus SHA 均为 `04fb31f7d4301aa2fd3d12c161cf06265a9bbf9f41e56116d90637f9d053afad`。稀疏为 BM25S/Lucene，dense 为 BGE-M3 的 CLS、L2 归一化向量与精确内积搜索；这里没有使用 BGE-M3 的稀疏或多向量模式。两路各取 50，按等权 RRF（k=60）融合，取融合前 50 送 BGE reranker，最终取前五。encoder/reranker revision、软件包和路径均在 manifest 中。encoder max length=8192，reranker=1024，batch=16，模型运行于 CUDA/bfloat16；123 次 query 编码和 6150 对 rerank 输入的 token-limit hit 均为 0。该计数表明本次没有触及配置 token 上限，不是普遍的无截断保证。

## 十二组总体结果

相关性指标均为有标签查询的宏平均百分数；耗时为全部查询阶段 wall time，单位 ms。RRF 行包含两路召回与融合，rerank 行还包含重排。

| 查询视角 | 方法 | Recall@5 | nDCG@5 | Recall@50 | 耗时均值 | 耗时 P95 |
|---|---|---:|---:|---:|---:|---:|
| current | BM25 | 31.20 | 25.75 | 60.98 | 1.30 | 3.04 |
| current | dense | 38.85 | 35.45 | 71.37 | 58.49 | 106.78 |
| current | RRF | 38.97 | 36.22 | 76.07 | 60.19 | 109.16 |
| current | RRF + rerank | 52.31 | 48.33 | 76.07 | 251.01 | 350.47 |
| last3 | BM25 | 20.94 | 16.71 | 55.90 | 1.31 | 1.77 |
| last3 | dense | 29.62 | 23.88 | 58.29 | 36.46 | 57.68 |
| last3 | RRF | 24.66 | 23.88 | 64.32 | 38.10 | 59.32 |
| last3 | RRF + rerank | 32.44 | 27.56 | 64.32 | 199.13 | 383.46 |
| all_user | BM25 | 19.23 | 14.58 | 47.26 | 1.67 | 2.54 |
| all_user | dense | 24.06 | 20.30 | 51.45 | 38.49 | 75.57 |
| all_user | RRF | 23.38 | 20.12 | 60.90 | 40.47 | 77.69 |
| all_user | RRF + rerank | 30.94 | 26.63 | 60.90 | 229.02 | 422.74 |

current RRF 相对 dense 的 Recall@5 仅提高 0.13 个百分点，但 Recall@50 提高 4.70 个百分点：融合主要扩充了这一批查询的可用候选。再加 rerank 后 Recall@5 提高 13.33 个百分点，39 个查询中 16 个改善、18 个持平、5 个退化。不能只看平均值而把重排视为每题都可靠。

相对 current rerank，last3 rerank 在 4 个查询改善、19 个持平、16 个退化；all_user 为 2/19/18。当前视角的优势不是“历史无用”的证明：视角由这同一批 dev 结果比较选出，且以下会话和难例确实需要历史锚点。

## 会话差异

下表为各会话 Recall@5 百分数。会话 ID 使用前八位；逐题映射保存在原 manifest 和分析 JSON 中。

| 会话 | 有标签轮数 | current BM25 | current dense | current RRF | current rerank | last3 rerank | all_user rerank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1c041ce4 | 8 | 18.75 | 50.83 | 25.00 | 61.25 | 20.62 | 30.00 |
| b4614daa | 8 | 40.62 | 54.17 | 63.54 | 54.17 | 29.17 | 4.17 |
| c01c8cf1 | 5 | 76.67 | 53.33 | 70.00 | 86.67 | 93.33 | 86.67 |
| c6c3b02c | 9 | 28.70 | 25.00 | 19.81 | 42.59 | 11.11 | 18.52 |
| d5b1e735 | 9 | 11.11 | 20.37 | 31.48 | 33.33 | 33.33 | 37.04 |

其中 b461 会话的 current 重排比 RRF 低 9.38 个百分点；c01 的 last3 比 current 高 6.67 个百分点；d5 的 all_user 比 current 高 3.70 个百分点。因此下一步应研究有选择的历史利用和排序失败，不能按整体均值永久移除历史，也不能根据当前题的 qrels 挑视角。

## 候选池与前五排序分别限制了多少

raw `hybrid.candidate_ids` 保存两路去重后的完整 union，可能超过 50；summary 的 candidate_pool 指 **RRF 前 50**。每题已核对其集合与 reranker 的 50 个输出完全一致，且输入顺序与 `rerank_candidate_ids` 一致。重排前后 Recall@50 相等是该流程的集合合同，不是另一条独立改善证据。

| 查询视角 | union 平均篇数 | union Recall | 前50 Recall | 重排前五 Recall | 前50无正例题数 | 有正例但重排前五全漏 |
|---|---:|---:|---:|---:|---:|---:|
| current | 88.07 | 80.43% | 76.07% | 52.31% | 3 | 6 |
| last3 | 85.76 | 67.78% | 64.32% | 32.44% | 9 | 12 |
| all_user | 83.51 | 61.92% | 60.90% | 30.94% | 10 | 11 |

39 个有标签查询各有 1–5 个正例，因此这里前五容量本身不限制理论 Recall=1。current 的剩余 47.69 个百分点，可以作离线标签归因：23.76 点是相关段落已经在前 50 内、却未排进前五；23.93 点是前 50 没召回。后者中，4.36 点在两路完整 union 内、因 RRF 截到 50 丢失，19.57 点连完整 union 也未召回。这里的“理想排序”只用于分析误差上限，绝不作为线上 feature 或按标签补候选。

current 的池内零正例恰为 `d5b1e735a040853ed361a3dfde1b8ef0` 第 2、3、4 轮；池有正例但前五全漏分别是 b461 第 4 轮、c6c 第 4/6/8 轮、d5 第 5/9 轮。单纯增加 reranker 算力不能修复前一类，单纯扩大池也不保证修复后一类。

## 可复核难例

这些判别只使用本次 dev 标注、query 和原始返回段落；未标注段落不能直接称为错误段落。

* **历史主题牵引：** `b4614daadceeecb400f7baf0aa48dcb8<::>5` 当前问题是希望在集成前保护 phone connection。last3 同时拼接 WebSocket/customization ID 的前两问。正例 `ibmcld_16288-1733-3996` 从 RRF 第 1 降到 rerank 第 8，Recall@5 从 1 降到 0，前五被 WebSocket 主题段落占据；current 视角保持第 1。历史主题牵引是对原文和排名的解释，未做因果干预验证。all_user 的前 50 已无正例，是更早的召回失败。
* **当前追问欠缺锚点：** `d5b1e735a040853ed361a3dfde1b8ef0<::>5` 的当前问题为 “Not documents, I was asking for attachments?”。唯一正例 `ibmcld_00510-7123-9213` 从 current RRF 第 4 降到 rerank 第 16，前二转为 Assistant input attachments 段落；当前 query 未显式带出此前 document database 语境。all_user 此题保留正例并排到第 2。相邻第 4 轮 “Is there a limit on the file size?” 的 current 池内无正例，而 last3/all_user 池中有，重排也进入前五。它们说明简单追问确实可能需要适量历史。
* **同主题的前五边界：** `b4614daadceeecb400f7baf0aa48dcb8<::>2` 询问 WebSocket API。两个正例在 current RRF 第 2/4，重排后第 3/6，Recall@5 从 1 降到 0.5。掉到第 6 的 `ibmcld_13429-163247-165127` 涉及 4 MB frames 等细节。这是已召回相关段落的排序/覆盖问题，不能由检索接口成功或前五大致同主题排除。

## 实际阶段成本和计时边界

以下均值来自**每题配对差值**，不是用两个 P95 相减。dense 包含 query tokenization、GPU 编码、同步回 CPU、精确 CPU 内积和 top-k；rerank 包含 50 对 tokenization、模型 batch forward、回传和排序。它们是 wall time，不是纯 GPU kernel 时长。

| 查询视角 | BM25 均值 | dense 编码+检索均值 | 仅融合均值 | 仅重排均值 / P95 | 全链均值 / P95 |
|---|---:|---:|---:|---:|---:|
| current | 1.30 | 58.49 | 0.40 | 190.82 / 292.41 | 251.01 / 350.47 |
| last3 | 1.31 | 36.46 | 0.33 | 161.04 / 306.00 | 199.13 / 383.46 |
| all_user | 1.67 | 38.49 | 0.31 | 188.54 / 336.28 | 229.02 / 422.74 |

本次固定顺序每题执行 current→last3→all_user，并复用中间阶段、已加载模型和索引；没有独立重复、随机化视角顺序或专门 warmup 排除。首个 current dense 为 820.09 ms，完整链为 1330.71 ms；current dense 的中位数为 11.28 ms、均值为 58.49 ms，明显受慢样本影响。初始化或缓存因素可能参与，但现有计时不能隔离原因，不能据此声称 last3 在一般负载上比 current 更快。索引构建/模型加载不计入这些逐请求耗时，生成时间、c4 排队及检索与 SGLang 共 GPU 的干扰也未包含。

这批结果足以定位成本主要在编码/重排、而非不足 0.6 ms 的 RRF 融合；不能用它替代完整 P3 TTFT、完成时延或吞吐测试。特别是重排模型与生成模型共 GPU 后，应重新观察原成本表的适用性和实际压力。

## 下一步可以直接执行的开发对照

1. 保持原始 passage 和同一生成预算，用固定 current hybrid rerank 与 BM25 做真实 P3 对照；保存检索时长、排队、可见材料、答案和引用，分别评价 passage relevance 与 answer/claim support。当前 dev 的视角选择记录下来，测试集保持冻结。
2. 用当前问题作为明确主体，把少量历史实体/主题放入独立字段，或保留 current 与历史两条召回路线，再在固定 50 对重排预算内融合。与无条件拼接比较，记录参与历史和新增成本；选择规则只能依赖输入及已生成历史，不能依赖该题 qrels。先覆盖上面的电话主题切换和附件追问两类相反难例。
3. 将召回池和排序器对照分开：同一前 50 池对照 reranker；再固定排序器对照 50/较大池及融合配额。当前只有 4.36 个 Recall 百分点来自 union→50 截断，而池内排序仍丢 23.76 点，应避免只靠盲目扩大 rerank_k 增加成本。任何新模型和参数的 dev 选择都保留记录，不能重用这 39 个查询作为独立确认集。

更多模型复用候选与研究来源见 `docs/archive/retrieval_research_next.md`。上述方向是可执行开发计划，不是已证实的改进。

## 本地复核

```bash
PYTHONPATH=. python runs/retrieval-dev-analysis/analyze.py
```

输出 `runs/retrieval-dev-analysis/analysis.json`，包含 12 组原始精度、会话指标、阶段配对差值、无正例/漏排 task IDs、池与排序误差分解。脚本只读取指定的 dev artifact 和其 dev.tsv；没有引擎请求，没有修改原始结果或在线策略。
