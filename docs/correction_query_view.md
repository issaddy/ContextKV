# 可选的纠正识别查询模式

`query_mode: correction_aware` 是一个窄的、可复用的实验选项：只读**当前用户文字**，句首独立词 `No`/`Not`、短语 `I mean`/`I meant` 或 `I was asking`/`I was referring` 触发最近三轮 user questions，否则使用 current query。匹配不区分大小写，`NoSQL`、`Notice` 等词不会因前缀而触发。它是词法规则，不是已验证的意图分类器，也不识别所有语言或隐含纠正。

纯函数在 [`adapters/query_views.py`](../adapters/query_views.py)，输入没有 task ID、qrels、助手答案、当前生成结果或未来问题。`hybrid_retrieval.query_text`、HybridRetriever、factory 与离线 retrieval CLI 已允许该可选 mode，原默认 current/legacy BM25 last3 不变。现有 hybrid 配置只需显式改这一字段，其余模型、池和预算维持一致。

## 冻结检索结果重组

本规则由 [P3 附件反问失败案例](p3_retrieval_integration.md) 启发，随后在已查看的 Cloud dev 上探索，**不是独立 heldout 验证**。用 `bench.query_selector` 从已保存的 12 组结果按当前文字选择 current 或 last3 的同 stage 结果；每个路由先确定，再读取 qrels 统计。没有 GPU 检索、生成请求或线上评分，保存的 stage 耗时也不代表新模式的端到端延迟。

```bash
python -m bench.query_selector \
  --retrieval runs/remote-h200-retrieval/runs/cloud-dev-hybrid-rrf50-rerank1024/retrieval.jsonl \
  --manifest runs/remote-h200-retrieval/runs/cloud-dev-hybrid-rrf50-rerank1024/manifest.json \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/correction-query-exploration-v1
```

产物 `summary.json` 记录全部 41 条路由、三个触发 query、各 stage 的 task/session 指标与来源；`selected-retrieval.jsonl` 保留实际被选中的排名。输出目录必须是新目录，旧检索结果不改。

全部触发 query 如下，没有按相关性指标挑选触发者：

| 会话 / turn | 当前用户文字 | 命中线索 |
|---|---|---|
| `c01c8cf1…` / 2 | I meant the issues that may arise when installing it with CocoaPods. | I meant |
| `c01c8cf1…` / 3 | No, there is a warning that comes up when installing it from CocoaPods. | No |
| `d5b1e735…` / 5 | Not documents, I was asking for attachments? | Not |

41 轮中 39 轮有 dev qrels；缺失的 Carthage turn6/7 仍明确排除。结果是 current → correction-aware：

| stage | R@5 | nDCG@5 |
|---|---:|---:|
| BM25 | .31197 → .28632 | .25749 → .24614 |
| dense | .38846 → .40556 | .35447 → .37252 |
| RRF hybrid | .38974 → .37265 | .36218 → .35558 |
| hybrid_rerank | .52308 → .52308 | .48327 → .48033 |

不能宣称该模式整体改善所有检索器。Reranker 的召回不变，两个 CocoaPods 触发轮的排序略降；dense 有改善，BM25/RRF 反而下降。未经查看的新会话仍需验证泛化。

目标附件轮的直接限额证据有具体变化：`ibmcld_00558-23465-25360` 明确包含 Cloudant “Single Attachment Size 10 MB”，从 current reranker 第 17 位进入 last3 第 3 位。它原本就在 current RRF 输入候选中，问题不在未召回。这个限额表并不是该轮唯一 qrel 标注的附件最佳实践邻段，因此官方 R@5 仍为 0，不能用单一 qrel 分数代替回答所需事实覆盖，也不能反过来以一个有用来源抵消全部负面指标。

该改进优先解决“当前反问需要前文的附件/大小意图”，没有使用上一轮助手给出的 50/32MB 错误答案来扩展 query。真实 P3 回答是否能据新证据覆盖历史错误尚待重跑；即使检索表已进入 top5，生成模型也可能继续相信错误历史。

## 验证边界

28 项相关测试通过，涵盖当前线索、词边界、只有一轮时的等价视图、不读取生成历史、改变 task ID/质量元数据不改变路由、保存 query 与 user view 一致、旧 factory/default 行为。Ruff 通过。模式默认关闭，后续应配 current/BM25 与 last3/hybrid 的交叉对照、可见证据审核及真实生成延迟；保留所有触发和误触发案例，不把这三条已分析 query 当作独立泛化证据。
