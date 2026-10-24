# P3 检索到生成主路径：开发集集成审查

2026-10-24。真实运行来源为 `runs/remote-h200-p3-hybrid/runs/p3-retrieval-integration/{bm25,hybrid}/` 的完整 benchmark/audit/manifest/events；独立汇总在 `runs/remote-h200-p3-hybrid/independent-pipeline-analysis.json`。以下质量个例为 **assistant evidence audit，不是 human review 或整体质量评分**。只使用 Cloud dev，未读取 test qrels。

## 检索一致性与计时

两组均使用 Q0、`normal_rag`、P3、相同 5 会话/41 轮、并发 4、1024 输出预算，均 41/41 成功，各自 cold reset 验证通过。**对照同时改变了 query 表达和检索器**：BM25 为最近三个 user questions，hybrid 为 current query 的 BM25+BGE-M3+RRF50+BGE reranker。不能把整体差异单独归因于 dense、融合或 reranking。

逐 task 对照固定离线检索运行 `cloud-dev-hybrid-rrf50-rerank1024/retrieval.jsonl`：BM25 与 `last3/bm25`、hybrid 与 `current/hybrid_rerank` 的 top5 **ID 及顺序各 41/41 完全相同**。没有把检索排名重排误认为原文改变。hybrid 只使用当前/已可见 user questions，不使用 golden answer 或当轮生成结果；实际生成历史仍进入 P3 回答上下文。

82 轮均满足 `pipeline t_arrive ≤ t_retrieval_done ≤ service timing.t_arrive ≤ t_first`。两份报告 `time_origin=pipeline_arrival`，用 benchmark arrival 重算首 token 均值与 summary 一致，证明端到端 TTFT 包含检索。Hybrid 运行时加载一组模型，串行检索锁的排队包含在这个区间；其 encoder/reranker 与生成服务同用 GPU0。模型初始化 14.42 s 单独记录，不在每轮计时内；本批首轮推理/队列效应没有被伪装成纯模型前向开销。

| 指标 | BM25 / last3 | hybrid_rerank / current |
|---|---:|---:|
| 检索墙钟均值 / P95 ms | 4.94 / 10.14 | 699.26 / 1955.57 |
| pipeline TTFT 均值 / P95 ms | 203.70 / 349.66 | 867.33 / 2110.35 |
| 完成均值 / 等会话完成均值 s | 2.546 / 2.581 | 3.624 / 3.654 |
| Control 均值 ms | 99.31 | 85.46 |
| 输入 tokens 总量 | 206,683 | 209,373 |
| 输出 tokens 总量 | 17,591 | 20,382 |
| receipt 实际 Prefill tokens | 164,123 | 160,221 |
| 输出截断 / 41 | 0 | 1 |
| 对应 dev qrels R@5 / nDCG@5 | .2094 / .1671 | .5231 / .4833 |

qrels 只有 39/41 轮：Carthage 会话 turn6/7 缺标注，明确排除；召回指标不是回答质量。hybrid 输出多约 15.9%，检索更慢约 694 ms/轮；首 token 与完成耗时均增加。Hybrid 的 2050 次 query-passage 分类输入均无 token-cap hit，不能把下面失误简单归因于 1024 reranker 截断。现有引用解析器仅在 BM25 4/41、hybrid 1/41 回答中识别到正式引用标记；未识别到错误 ID 不代表其余答案有可靠来源。

## 个例：检索改善与回答不足同时存在

**Cloudant Query，会话 `1c041ce4…` turn8。** BM25 返回混合 SQL/COS 目录文档，回答将 `CREATE TABLE ... PARQUET LOCATION cos://...` 等称为 Cloudant 操作。Hybrid top5 切换到直接介绍 Cloudant `_find`、索引的段落（如 `ibmcld_00539-7-1755`），方向明显更贴合当前问题，但回答膨胀到 1024 tokens 并截断。更具体地，三个 cURL `-d 'JSON'` 中有两个完整闭合的 payload 结构非法：一个 `$or` 最后对象漏 `}`，另一个 `$exists` 后写成 `)`；标准 `json.loads` 分别报 column112/39。两个错误后都有闭合代码围栏，后续还继续生成报告和总结，因此**不是最终输出截断才造成的语法错误**。仅扩大输出上限不能修复这种已生成的坏代码；可先加离线 JSON/code-block 结构核查，比较明确要求 API 示例可核验的提示，而不自动执行生成代码。

**附件限制，会话 `d5b1e735…` turn5。** 当前问题为 `Not documents, I was asking for attachments?`，先前用户顺序为 document databases、任意 JSON/字段、图片/PDF 附件、文件大小上限。BM25/last3 回答 279 tokens，引用到了明确的 10MB 附件限制表。Hybrid/current 回答 690 tokens，继续沿用自己上一轮生成的 50MB/32MB“uploaded/crawled documents”限制并称为 Cloudant 附件上限；当前五段可见正文没有这两个数值。说明用户纠正后，生成历史中的错误仍可能传播。

该轮 hybrid 当前五段 title 元数据都为空，不能编造题名。正文内容分别为：`ibmcld_03115-1274-2254` 的 Assistant input-attachment JSON 续页；`ibmcld_03115-7-1446` 的 “Processing input attachments”；`ibmcld_00477-8324-9733` 的 Cloudant 数据库端点表；`ibmcld_00556-7-1696` 的 “How to use attachments”；`ibmcld_00625-3455-5153` 的 Cloudant JSON/Query parameters 续页。

真正明确写出 “Single Attachment Size 10 MB” 的段落是 **`ibmcld_00558-23465-25360`**，不是 qrels 中只讨论附件最佳实践的邻段。它已经在 current RRF union 和 rerank 输入 50 内，扩池并非此例首要缺口：

| query 表达 | BM25 rank | dense rank（50 内） | RRF union rank | rerank rank（50 内） |
|---|---:|---:|---:|---:|
| current | 14 | 无 | 25 / 94 | 17 |
| last3 | 5 | 18 | 9 / 79 | 3 |
| all_user | 16 | 6 | 7 / 86 | 9 |

最近三轮保留了图片/PDF与大小限制意图，此例可以把直接限制表提升到 top5；all_user 并未更好。下一步应做 current/BM25 和 last3/hybrid_rerank 交叉消融，或仅用可见历史识别省略/纠正类问题并检验 query 选择。不能只依据这个 case 在全部任务上强制某种 history 长度，也不应把 qrels 邻段命中当作已取回限制表。

**电话连接安全，会话 `b4614daa…` turn5。** 两组都检索到 `ibmcld_16288-1733-3996`，其正文明确说明 SRTP、SIP digest、TLS 与控制台选项。BM25 回答 699 tokens，额外给出当前五段没有支持的 `assistant.updatePhoneIntegration(...)` SDK 示例；hybrid 回答 473 tokens，保留有正文依据的控制台步骤。这个样本显示更贴近证据的短回答是可达到的，但不是总体质量改善的证明。

## 有证据支撑的下一步

先补两组 query/retriever 交叉对照来拆分收益来源，并保留当前模式在附件纠正类问题上的具体失败；对生成的结构化示例做离线语法核查，将“当前证据优先、历史回答不是证据”与较短回答约束作为可测试的提示方案。检索工程上，当前 4 会话进入单模型锁的等待已反映到 P95，应比较并发 query 成批或独立 GPU 部署的真实端到端成本，而不能以离线单 query 延迟代替在线检索开销。

这些工作可继续在 dev 迭代。正式质量判定仍需要离线 judge/证据审核与独立人工盲审；保持 test 未触及，不以 R@5 增长或 41/41 完成代替完整场景效果。
