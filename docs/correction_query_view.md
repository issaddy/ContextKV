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

该改进优先解决“当前反问需要前文的附件/大小意图”，没有使用上一轮助手给出的 50/32MB 错误答案来扩展 query。下节记录真实 P3 重跑：检索表已进入 top5，但生成层仍有新的错误概括。

## 验证边界

28 项相关测试通过，涵盖当前线索、词边界、只有一轮时的等价视图、不读取生成历史、改变 task ID/质量元数据不改变路由、保存 query 与 user view 一致、旧 factory/default 行为。Ruff 通过。模式默认关闭，后续应配 current/BM25 与 last3/hybrid 的交叉对照、可见证据审核及真实生成延迟；保留所有触发和误触发案例，不把这三条已分析 query 当作独立泛化证据。

## 真实 P3 重跑：取证改善，答案未完整修复

完整产物在 `runs/remote-h200-correction/runs/p3-correction-aware/run/`，派生比较为 `runs/remote-h200-correction/independent-comparison.json`。同 Q0、normal_rag、P3、Cloud dev、c4、1024 预算，41/41 成功，top5 ID 与顺序 41/41 匹配离线 selector。与原 current/hybrid 相比，恰好只有三个触发轮的检索序列改变；模型修订/模板/控制器源码标识保持一致。

| 指标 | 原 current/hybrid | correction_aware |
|---|---:|---:|
| pipeline 完成均值 / 等会话均值 s | 3.624 / 3.654 | 3.685 / 3.704 |
| pipeline TTFT 均值 / P95 ms | 867.3 / 2110.3 | 860.8 / 2288.9 |
| 检索墙钟均值 / P95 ms | 699.3 / 1955.6 | 689.8 / 2080.2 |
| 输入 / 输出 tokens 总量 | 209,373 / 20,382 | 211,839 / 20,850 |
| receipt Prefill tokens | 160,221 | 161,279 |
| 截断 / 41 | 1 | 1 |

两次均为一次真实闭环运行，不能把均值的小幅变化当稳定性能收益。虽然仅改三轮检索，38 个未触发轮中仍有 30 个实际 input IDs、34 个 output IDs 发生变化：P3 自身生成历史继续传播，而且并发生成可能不同。因此这不是冻结历史的三条成对因果实验。时间起点仍是 `pipeline_arrival`，检索等待没有从 TTFT 中去掉。

以下逐项是 **assistant evidence audit，不是 human review，也不是总体质量评分**：

| 触发轮 | 输出长度（原→新） | 对照当前完整证据的观察 |
|---|---:|---|
| CocoaPods turn2，`I meant…` | 425 → 435 | Swift warning 本来已有相关来源。新答案把上一轮助手的句子当作“证据原文”引用，未建立质量改善。 |
| CocoaPods turn3，`No…` | 350 → 356 | 两版都识别 Swift-version warning；新版 `.xcworkspace` 和 Podfile 的 `SWIFT_VERSION='4.0'` 在可见段 `ibmcld_10817-1342-3184` 有依据，但没有证据可将此轮统计为新增修复。 |
| attachments turn5，`Not documents…` | 690 → 415 | 新版正确提取 10MB 附件限额，但添加了错误的“含附件文档总共≤1MB”，不能算完整修复。 |

**附件例的失败位置。** 当前两份限额表（`ibmcld_00558-23465-25360`、`ibmcld_12904-23555-25450`）分别列出 Individual Document 1MB、Single Attachment 10MB、Request Body 11MB。对应可见正文完整包含这些三行；`‡` 脚注说明 Frankfurt 的 EU 托管位置，与大小无关，未发现相关大小脚注被裁掉的证据。其他可见段讨论小 JSON 文档和附件存储，但没有给出“附件二进制也必须算进1MB总和”的条件。新答案却反复加入 `including attachments`，甚至先说附件可10MB，又说文档连附件合计不得超过1MB，属于无依据的限制关系推断和自相矛盾。

当前片段没有专门解释每种上传表示下大小如何计算，补充解释性证据可能有帮助；但不能把这次失败归咎为限额表丢失或已知脚注被截断。模型本可以准确保留三个限额的各自对象，不添加来源未说明的关系。独立审核时另核对了 [IBM 官方 Cloudant Gen2 复制说明](https://cloud.ibm.com/docs/cloudant-gen2?locale=en&topic=cloudant-gen2-replication-guide)：其在假设请求上限为11MB的例子中允许1MB文档加10MB附件，说明为何需要区分文档、附件和请求这三个量。这个不同文档版本只作解释性旁证，未送入原生成，也不代替固定 MTRAG 快照的具体约束；不将今天网页上可能变化的限额回填到原运行。

**历史回答被误当证据。** CocoaPods turn2 新答案给出带引号的句子 `This method might require additional setup for CocoaPods and could introduce dependency management issues.`，并绑定 `E_69ea040ae8`（`ibmcld_02772-4213-5899`）。该字符串不在五段 `audit_evidence` 中，却精确存在于此前 assistant turn1。这里不仅是引用格式问题，而是模型将自己的历史陈述提升成了外部证据。纠正路由只使用 user questions，未将这句送入检索；错误发生在后续回答利用会话上下文时。

可保留的结果是：狭窄纠正线索能将目标限额证据送到正确位置，但尚需独立检验**限额对象/条件的逐项引用**及**历史助手文本与外部证据的区分**，同时观察三个触发轮和未触发轮的回归。后续[焦点行完整P3](table_focus_p3_validation.md)和[用户历史主题视图](user_topic_query_view.md)沿这些实际缺口推进；后者保留当前模式作为对照，没有替换其原有规则。默认模式、正式质量门与最终 test 状态保持不变。
