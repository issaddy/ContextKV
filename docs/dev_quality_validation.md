# 完整开发集五维评价：固定 Selene、三臂 123 答案

这次完成了全部 615 次固定模型调用和完整输入审计，但没有建立质量非劣。609/615 个维度按预先冻结的原生格式得到分数，6 个缺少最终 `**Result:**`，保留为 null；所有可完整配对的维度，其预设非劣检验仍为 **inconclusive**。因此 `Fixed_4096`、`Workload_gate0` 的 `dev_quality_eligible` 均为 false，不能将未通过全部解释为格式问题。

同时，来源核查确认了评价器的实质漏判：在同一道附件问题中，Rebuild 和 Fixed 的答案均含错误来源绑定，Selene 却给引用满分；Gate0 在这一个问题的来源绑定更准确，反而因风格理由得到较低引用分。以下数字是固定评价器的开发测量，不是五维真值或人工评分；原分数没有修改。

## 固定对象、协议与产物

生成对象为 `controller-stage-gate-16384-v2/gpu-16384/r1` 的完整 Rebuild、Fixed_4096、Workload_gate0，各 5 个 Cloud dev 会话、41 轮；生成源码 `ee480e41b37ab5f5bf559ca8cc4db51d326eb56a`，P2 / Q0 / c4 / seed42 / 1024 输出上限，GPU16384、host279040。同一后续用户脚本下，各方法使用各自实际生成历史。没有选择对某一方法有利的子集，也没有把 r1 分数借给 r2 或新检索结果。

评价器为 `AtlaAI/Selene-1-Mini-Llama-3.1-8B`，revision `427792f1c3e2073cb7da216924fd884b1ba496e0`；原 `selene_absolute_v1`、项目原五维 rubric、`native_numeric_only`、temperature0、1536 输出上限、32768 上下文、无 reference targets。原生 1–5 分按 `(n−1)/4` 映射为 [0,1]，每个答案每维单独调用，不跨维平均。评价部署记录为项目 `955ff92124f15e095741e3a6d3e51eb0c925ef3c`、stock SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。原始 `runtime-before.json` 保留实际服务名、revision、context 与设备信息。

本地可复核产物：

- 原始评分：[quality-dev-completion-v1](../runs/remote-h200-quality-dev/runs/quality-dev-completion-v1/manifest.json)，下含三份 `scores.jsonl`、coverage、quality 与 runtime。
- 原始 [CPU preflight](../runs/remote-h200-quality-dev/runs/quality-dev-completion-prep/preflight-h200.json)；冻结准备 [plan](../runs/quality-dev-completion-prep/plan.json)、[完整协议](../runs/quality-dev-completion-prep/protocol.json) 与逐字复制的 run/manifest/audit。
- 独立 [审计脚本](../runs/quality-dev-completion-review/analyze.py)、预先固定的 [规约](../runs/quality-dev-completion-review/PROTOCOL.md)、[summary](../runs/quality-dev-completion-review/h200-v1/summary.json)、[全部 615 slots](../runs/quality-dev-completion-review/h200-v1/slots.jsonl)、[全部 123 答案及完整来源](../runs/quality-dev-completion-review/h200-v1/answer-review.jsonl)、[来源复核及缺失明细](../runs/quality-dev-completion-review/h200-v1/semantic-review.json)。这些 `runs` 文件按仓库规则不进 Git。

冻结 plan SHA256 为 `36af300f00498508e8dde9cff467348a0faa615819c14c6f80737a3b734aac94`，实际 preflight 为 `e404ea1b34f252bb3abc265d96ed910dc19c20ddba28e987d1f5275f22e9a557`，实际运行 manifest 为 `64c85c1f27f75ca09f624bbd1ab1bafbd4b890567c80156691e57cc6da4bb62e`。

## 输入、预算与缺失核验

独立审计 `issues=[]`：123 个答案的原 run/audit/score、query、实际 history、input token 身份、完整可见源和 dev split 全部绑定；615 个 `(method,task,dimension)` 与 CPU preflight 一一对应。重建原生消息保留完整历史、retrieved material、visible manifest、audit 正文与当前答案。这里是原始记录及确定性 renderer 的核查，不声称保存了全部 HTTP 请求抓包。

615 次均有 usage，且逐次满足 `preflight.input_tokens == native_result.input_tokens == usage.prompt_tokens`。最大输入为 17,798 tokens；加预留输出 1,536 后仍小于 32,768。总实际输入 6,041,277 tokens、输出 86,927 tokens；单次输出最少 92、中位数 140、最多 197，远低于输出预算。615 个 response ID 均唯一，冻结 runner 无重试，所有 slots 均保留；没有独立网络级日志，不能进一步宣称网络重试次数的外部证明。

**评审输出**全部 `stop`，无 HTTP 或 length 失败。但 `stop` 不代表格式完整：以下 6 条仅有 Reasoning，没有最终 Result，不能从正文的 “score of 4/5” 猜填。

| 方法 | task 后缀 | 维度 | 实际输出 tokens |
|---|---|---|---:|
| Rebuild | `1c041…<::>5` | answerability | 152 |
| Fixed_4096 | `c6c3…<::>8` | answerability | 150 |
| Workload_gate0 | `c6c3…<::>6` | groundedness | 127 |
| Workload_gate0 | `c6c3…<::>8` | answerability | 116 |
| Workload_gate0 | `1c041…<::>6` | groundedness | 134 |
| Workload_gate0 | `1c041…<::>6` | citations | 124 |

Rebuild、Fixed 各 204/205 维可解析，Gate0 为 201/205；完整五维答案为 40、40、38，共 118/123。缺失率 6/615≈0.98%，没有把缺失当零分，也没有通过交集删除对应轮来计算主要 CI。增大输出上限并不能直接解释或修复这次在 116–152 tokens 就正常 EOS 的缺 Result 问题。

**被评价的生成答案**另有 4 条达到原 1024 输出上限，全部保留：Fixed 的 `b461…<::>2`，Gate0 的 `b461…<::>6`、`c6c3…<::>8`、`b461…<::>8`。Rebuild 没有生成截断。它们不能因为评审自身没有 length 就被误记成生成完整；例如 Fixed 的截断答案仍获得前四维满分，分数本身不是完整性验证。

## 每维分布与配对结果

下表是 [0,1] 均值。括号是有效分数 / 41，带 † 的数值仅为可用样本描述均值，完整均值为 null，不参加主要配对结论。

| 维度 | Rebuild | Fixed_4096 | Workload_gate0 |
|---|---:|---:|---:|
| correctness | 0.9390 (41) | 0.9268 (41) | 0.9024 (41) |
| groundedness | 0.9268 (41) | 0.9329 (41) | 0.9103† (39) |
| answerability | 0.9563† (40) | 0.9625† (40) | 0.9125† (40) |
| conditions_versions | 0.9085 (41) | 0.9207 (41) | 0.8841 (41) |
| citations | 0.7012 (41) | 0.6768 (41) | 0.6750† (40) |

609 个可解析原生分数中，5 分 348 个、4 分 219 个、3 分 42 个，没有 1 或 2 分。这是明显偏向高分的分布描述，不能据此推断答案都没有严重错误；下节已有明确反例。完整逐方法/逐会话/逐维分布在 summary 中，避免用单一总均值掩盖差异。

主要统计固定为每轮配对均值差、按整个会话 bootstrap，Rebuild 为参考，5 clusters / 41 pairs，seed42、5000 次、95% CI、epsilon0.01。单位均为 [0,1] 分数差。

| 候选 | 维度 | 候选−Rebuild | 95% CI | 结论 |
|---|---|---:|---|---|
| Fixed_4096 | correctness | −0.0122 | [−0.0577, 0.0233] | inconclusive |
| Fixed_4096 | groundedness | 0.0061 | [−0.0298, 0.0321] | inconclusive |
| Fixed_4096 | answerability | — | 不完整 | incomplete |
| Fixed_4096 | conditions_versions | 0.0122 | [−0.0132, 0.0417] | inconclusive |
| Fixed_4096 | citations | −0.0244 | [−0.0962, 0.0179] | inconclusive |
| Workload_gate0 | correctness | −0.0366 | [−0.0876, 0.0116] | inconclusive |
| Workload_gate0 | groundedness | — | 不完整 | incomplete |
| Workload_gate0 | answerability | — | 不完整 | incomplete |
| Workload_gate0 | conditions_versions | −0.0244 | [−0.0549, 0.0061] | inconclusive |
| Workload_gate0 | citations | — | 不完整 | incomplete |

独立对齐后复算与原 `quality.json` 完全一致。等会话权重的辅助敏感性分析也没有把上述完整维度变成非劣；它没有替换原先按轮均值的主要 estimand。只有 5 个开发会话，既不能将 inconclusive 说成等效，也不能说已证实劣化。修复 6 个格式缺失，即使能得到分数，也不能消除其他完整维度现有的统计不确定性。

## 来源语义复核：分数不能代替逐项支持关系

以下为两名 assistant agent 独立核对完整原始来源后的分析，不是 human_blind，不生成替代分数，也不是 123 答案的全量语义准确率测量。

预先指定的 `d5b1e735a040853ed361a3dfde1b8ef0<::>5` 当前问的是 “Not documents, I was asking for attachments?”。三臂主结论“Cloudant 单附件最大 10 MB”均能由完整 visible 表格支持；不能将过去其他运行的“附件包含在 1 MB 文档内”错误标签按 task ID 搬到本轮。

本轮三臂共同的实际来源映射为：

| stable ref | 原 passage | 实际内容 |
|---|---|---|
| `E_53079fa73a` | `ibmcld_00558-23465-25360` | Cloudant 表：Individual Document 1 MB / Single Attachment 10 MB / Request Body 11 MB |
| `E_fe06da00dd` | `ibmcld_12904-23555-25450` | 同一 Cloudant 限额表 |
| `E_8312203498` | `ibmcld_00510-5537-7566` | 文档 1 MB、Avoid using attachments 章节开头；没有单附件 10 MB 句 |
| `E_6a86be8e02` | `ibmcld_12748-0-2435` | Security and Compliance Center；Attachment 50 per account，不是 Cloudant 文件大小 |

Rebuild 把 `Single Attachment Size 10 MB` 写成 `E_6a86be8e02` 的直接引文，另一个 `E_fe06da00dd` 才真正支持它；Fixed 又把同一直接引文错误归给 `E_8312203498`，三条引文中两条来源错绑。Selene 对这两份答案都给五维满分，citation 理由明确称 ID 正确、支持主张。**事实在别处有支持，不能替代每条附着引用自身正确。**这不是未知 ID 或 parser 漏识别：两个错误 ID 都存在，且源正文完整进入评审。

Gate0 将 `E_8312203498` 用于准确的 Avoid using attachments 原句，将 `E_53079fa73a` / `E_fe06da00dd` 用于 10 MB 表，当前来源绑定更准确。它仍较冗长，也举出当前段落未直接点名的 IBM Cloud Object Storage 例子，不能因此直接给它全部满分或声称整体质量更优。但 Selene 的 citation4主要理由是增加 ID 解释/组织，conditions4又要求它强调 individual attachment，虽然答案已经多次写明 single attachment；correctness4的理由还建议把 10 MB 扩展为文档和附件共同限制，与原表对象区分冲突。可见局部排序和 rationale 都需要独立来源检查。

遍历全部答案的格式诊断进一步发现：Rebuild / Fixed / Gate0 分别有 32 / 32 / 31 条没有可解析的 `E_ID` 引用，即共 95/123；其中 94 个 citation 分数有效，64 个得到原生4或5分，30个3分，另1个缺分。所有答案的 unknown ID 计数为0；这也不排除上述**存在但错绑**的 ID。无引用本身并不自动判错，适当拒答可能无需引用，因此这里没有机械改分。

一个具体边界是 Rebuild 的 `c01c8cf11437e6bb3bc93efac26528c2<::>5`：答案给出完整 Carthage 操作说明，却没有任何 Evidence ID；citation 理由声称提供了有效引用，末段说 score4，但最终 Result 为5。解析器正确保留最终 Result，并不能使理由与锚点一致。我们没有执行其代码，也没有据这些有限案例制造 judge 准确率。

## 对新 13 格评价的有限建议

本轮 6 个 null 原样保留，不能补问这6条、放宽 parser、从正文挖数字或改 epsilon 后覆盖原结果。若新13格继续使用原 `selene_absolute_v1`，应事先接受仍可能出现缺失，并保留完整 expected 分母和 incomplete 结论；模型质量统计不会因“应当有完整分数”而例外填补。

若在新13格评分开始前决定改善**格式可靠性**，最小可复用方案是保留同一 Selene 权重、全部材料、原五维 rubric、temperature0、1536预算和严格解析器，只新增明确版本的“原生格式约束解码”协议：要求 Reasoning 后必须有唯一最终 Result1–5。固定 stock 的 `ChatCompletionRequest` 已支持顶层 `regex` / `ebnf`；`response_format` 的基本类型则为 text/json_object/json_schema，不能虚构 `response_format.type=regex`。可直接复用这个已有服务能力，而不是换 judge 或再训练。具体约束及请求字段必须作为新协议记录，完整新矩阵各臂一律应用；不能只约束本轮失败条目后与旧分混合。[SGLang 固定官方源码](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/entrypoints/openai/protocol.py#L800-L881)

这仍是尚未实现/验证的候选，约束解码会改变生成分布；要先做纯工程的格式合同检查，再按预先声明方案全量执行并保留任何 length/格式失败。它只处理输出结构，不能解决已实证的来源错绑、对象范围混淆或评价器打分偏高，不应以“全绿格式”为新质量贡献。

B6 选择需要完整、同源、同预算的新开发矩阵及各方法实际答案的质量绑定；本批旧 r1 分数只描述本批答案。正式结论继续区分固定自动评价器结果和技术方案 §11.5 要求的风险层人工盲审，test仍冻结。本轮没有修改质量gate、执行重试、读取test问答或生成任何人工分数。
