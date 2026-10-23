# 独立开源 judge：首轮开发验证

本轮已经真实运行跨家族的 Selene Mini 8B 与 Prometheus 2 7B，但**当前两者均不足以直接作为可靠的五维质量准入器**。原生评分都漏检附件限额回答的关键范围差异；Selene 在 GroUSE 暴露格式与语义问题，Prometheus 在当前 GroUSE JSON 协议下全部响应失败。不能把后者的 0 分泛化为模型没有语义判断能力。这是改进 evaluator 的开发证据，不是模型评分替代人审的结论。以下解释来自 assistant 对原文的核查，没有填入任何人工评分，也没有改写模型原分数。

原始文件位于 `runs/remote-h200-judge/runs/{selene,prometheus}-development-v1/`，各包含 `probe-audit.jsonl`、`probe-scores.jsonl`、两份 config、`runtime.json` 与 `grouse-train/`。只读派生脚本为 `runs/open-judge-review/analyze.py`，结果为同目录 `{selene,prometheus}-development-v1.json`。官方源码重放分别保存为 `grouse_analysis.json` 与 `prometheus-grouse-analysis.json`。未读取 GroUSE test 144 或项目最终 test；16 个 GroUSE train 样本是同一问题的不同条件变体，不是 16 个独立主题。

## 运行与比较范围

Selene 固定 revision 为 `427792f1c3e2073cb7da216924fd884b1ba496e0`，Llama 3.1 8B 基座；Prometheus 固定为 `66ffb1fc20beebfb60a3964a957d9011723116c5`，Mistral 7B 基座；生成器为 Qwen2.5-7B。runtime 记录 stock SGLang commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`、H200、BF16、TP=1、context/total tokens 32768、max running requests=1，以及本地模型/tokenizer 路径和实际 served model/revision。身份来源仍是保存的运行记录，不是权重的远程认证。

两个模型原生评分使用相同的答案、证据、五维 rubric、temperature=0、输出上限 1536、无参考答案、`native_numeric_only`；外层分别保留作者的原生模板与 Result 格式，并非字节相同的 prompt。GroUSE 则使用相同固定官方四模板 JSON、temperature=0.01、输出上限 2048、`strict_v1`。两条轨道评估不同任务，不能把两个分数平均为一个“judge 质量”。模型选择尚未冻结：

| 开发观察 | Selene Mini 8B | Prometheus 2 7B |
|---|---:|---:|
| 原生两个 probe 的有效维度响应 | 10/10，均 stop | 10/10，均 stop |
| `off`：correctness / groundedness / answerability / conditions / citations | 1 / 1 / 1 / 1 / 0.5 | 0.75 / 0.75 / 0.75 / 0.75 / 0.75 |
| `on`：同上 | 1 / 1 / 1 / 1 / 0.5 | 0.75 / 1 / 0.75 / 0.75 / 0 |
| GroUSE train 条件通过 | 48/96 = 0.5000 | 0/96，全部来自失败状态 |
| GroUSE 已调用响应解析成功 | 39/45 = 86.67% | 0/32 |
| GroUSE 完成状态 | 45 stop | 30 length、2 stop |

上述 probe 是已发现问题后的定向检查，不能据两个回答推算整体正确率、模型相关性或统计显著性。原生格式分数也不能用“更低就是更可靠”排序：需要看是否抓到了实际错误以及是否遵守相同评分锚点。

## 原生评分：没有发现预算或输入标签泄漏

两个 probe 对应真实生成的 `old_error_history` fixture，`probe-001=off`、`probe-002=on`。当前问题均为 “Not documents, I was asking for attachments?”；自身历史、五个可见证据项和完整证据正文逐项相同，只有被评答案与生成提示身份等字段不同。它们保留了此前 assistant 将 Discovery 限额误说成 Cloudant 限额的历史，因此这里测试的不是一段孤立的干净问答。

judge 的显式白名单不包含 `source_fixture_for_offline_review`、`source_arm_for_offline_review`、`source_generation`、targets 或 enrichments；off/on 标签没有进入请求。`reference_targets_supplied=false`。保存分数的 query/input_hash 与对应 audit 绑定一致。诊断包的 `retrieved_evidence=[]`，但五个 `visible_evidence_manifest` 与五份正文均在；正文共 9,615 字符，其中两个不同 passage ID 的限额表正文相同。这是完整可见来源的 probe，不是完整检索轨迹评测。

Selene 的 10 次评分 exact input tokens 为 **7,139–7,215**，Prometheus 为 **9,091–9,189**，均逐次等于对应 API `usage.prompt_tokens`，加 1536 输出预算仍远低于 32768。实际输出分别 105–141 与 142–262 tokens，20 次全部 `finish_reason=stop`，没有长度截断或解析失败。两模型的 probe audit 完全相同；token 差异涉及各自 tokenizer/原生模板，不能当成不同证据量。不能把此次高分归因于证据被截短、token 预算错误或 NA 被填成满分。

## 原文与理由的具体错位

可见的两个重复限额 passage 是 `ibmcld_00558-23465-25360`（`E_53079fa73a`）与 `ibmcld_12904-23555-25450`（`E_fe06da00dd`）。原文将三个对象分为三行：

| 原文对象 | 原文最大值 |
|---|---:|
| Individual Document Size | 1 MB |
| Single Attachment Size | 10 MB |
| Request Body Size | 11 MB |

`off` 回答反复添加“document, **including attachments**, is 1 MB”，并将其称为严格的合计限制。原文没有给 Individual Document 行添加这个包含关系；回答把单独的对象限制扩成一个更强的总量限制。该问题不需要缺失的脚注才能发现：当前完整表格已经区分了 document、attachment 与 request body。

Selene 的 correctness/groundedness 理由只概括“attachment 10 MB、document 1 MB”，没有核查新增的 **including attachments**。conditions 理由转而表扬结构，甚至建议明确“每个附件”，而被评答案已写明 single attachment / each attachment。其理由遗漏关键限定词，却对四维均给 5/5；这是对象与范围检查不足，不是末尾数字解析错误。

`on` 回答把 1 MB 文档、10 MB 单附件、11 MB 请求体分开，因而比 `off` 更贴近这张表。但“请求体等于 document + attachments”的实现细节、性能保证等仍不能仅凭三个数字便认定全部满分。本审查没有为它补写黄金分数。

两份答案都没有引用标记。Selene 的 citation 理由确实指出缺引用，却都给 **3/5**；当前 rubric 的低分锚点明确包括中心证据主张缺少必要引用，而 3 分要求混合的引用支持。这里没有任何已有引用，理由与分数锚点不一致。`off` 的 citation 理由还建议使用 `E_8312203498` 支持 attachment sizes，但该片段只给出 1 MB 文档限制与附件章节开头，没有 10 MB 单附件值。judge 找到“相关主题”不等于核对了“这一来源支持这一限额”。这些建议没有被执行为引用修复。

多个维度的理由反复赞扬标题、组织、简洁性，而这些不是对象、条件或证据支持的替代条件。现有输出证明专用 evaluator 也可能偏向流畅的总结；不能因为模型跨家族、输出合法就视为质量已验证。

Prometheus 并未补上这个缺口。其 `off` groundedness 与 conditions 理由明确复述并认可“document, including attachments”的总量限制，仍给 4/5；不是只在摘要中漏掉限定词。它还批评答案未提 external object storage 或性能影响，而原答案已经有这两项。`on` groundedness 升到 5/5，但 correctness/conditions 与 `off` 同为 4/5，理由主要围绕是否够详细，仍未指出两者最重要的对象范围差别。

Prometheus 的引用分歧更明显：两个答案都没有引用，`off` 得 4/5、`on` 得 1/5。`off` 理由承认完全缺引用，却将其解释为“只有少量覆盖问题”；`on` 虽符合低分锚点，理由又批评未回答更早历史中的 images/PDF 存储问题，偏离当前的附件限额问题和当前 citation 维度。它既出现分数锚点不稳定，也把历史话题混入了当前评分要求。两份原分数均保留，未取较符合预期的一份替代另一份。

## GroUSE：格式失败与语义失败分开保留

六指标条件通过数依次为 **8、3、8、11、9、9 / 16**，保持官方全部样本分母。只有 `train:7` 六项全部通过。45 次模型调用为 relevancy 16、completeness 16、faithfulness 10、usefulness 3；45 次均正常 stop。usefulness 的 11 次通过中，8 次是条件流程跳过后正确保持 null，不能解释为模型实际完成了 11 次 usefulness 判断。positive acceptance 与 negative rejection 由前两项推导，也不是两个独立模型判分任务。

6 次 Failed 的响应都是语法合法 JSON，但把 JSON null 写成了字符串 `"null"`：

- `train:2` 的 relevancy；
- `train:4` 的 relevancy 与 completeness；
- `train:8` 的 relevancy，其中 answer_1 为字符串 `"null"`、answer_2 为字符串 `"1"`；
- `train:12` 的 relevancy 与 completeness。

这些不是 HTTP、上下文或输出长度失败。原样记录为 Failed；没有在看过标签后将字符串改成 null 重算通过率。四次 relevancy 失败又阻断后续条件调用，因此不能把所有失败指标当作独立模型错误，也不能在缺少后续实际响应时推算“修好解析后的总分”。

独立重放固定官方 DTO 与 MetaEvaluator 后，Selene 的 45 次调用接受/拒绝和 96 个条件结果均与保存结果一致，没有发现适配器独有的失败。后续已实现默认不启用的 `normalized_v1`：仅转换 score 字段精确 `"null"`、ASCII 整数字符串及有限整数 float，保留原始响应/转换 trace，并绑定新的冻结身份。离线探测可恢复现有 6 次 Failed，39 次原成功值不变（11 处 string-null、1 处 integer-string）；但还缺少 4 次必需后续调用，以及最多 3 次条件性的 faithfulness 调用，故**没有生成新总分**。详见 `runs/open-judge-review/normalization-probe.json`；这不是新模型验证结果。

解析成功但仍违反条件的数量依次为 **4、11、4、1、3、3**。completeness 最弱，不能靠格式修复解决：

| 开发案例 | 可见的模型行为 | 解释与边界 |
|---|---|---|
| `train:0`，完整正确回答 | completeness=4，称还需更多具体细节 | 原条件要求 5；泛化地要求更详细，偏离“覆盖给定材料”的标准 |
| `train:1`，证据不回答问题且合理拒答 | completeness=1 | 应为 null；混淆“没有可回答信息”和“漏答已有信息” |
| `train:10`，来源明确涉及森林破坏/过度放牧，但回答拒答 | completeness=null，并称没有相关来源 | 应为 1；与上一例呈相反方向误判，说明没有稳定执行证据可回答性检查 |
| `train:3`，多来源覆盖的正确回答 | 将 excrement 对 soil quality 的概括判成不忠实 | 按原条件为误判；对同义表述过严，不能等同于已发现事实冲突 |
| `train:6`，来源明确提供 2.5 m 的虚构 wingspan | 模型说 2 m 50 cm 不受来源支持 | 来源第 9 段确有该数值；同时丢失单位等价和仅按给定来源判定的要求 |
| `train:5`，另一个带虚构 wings 的正确回答 | 模型接受 wings 句，却因 environment/wildlife 措辞给 faithfulness=0 | 不能把所有失败笼统归因为“模型用常识否定虚构资料”；这一例主要是表述范围判断失准 |

反例中原数据故意提供虚构资料，用于检查 evaluator 是否忠实评估给定来源；本项目并没有把这些内容当作真实世界知识。条件和 `metadata.goal/test_type` 只用于离线核对，没有发送给 judge。

## Prometheus 的 GroUSE 0 分主要反映格式不适配

Prometheus 的 16 个样本都在 relevancy/completeness 阶段失败，因此只执行了 32 次调用；faithfulness/usefulness 没有被实际调用，其失败状态由前置错误传播。这里不存在 96 个独立的错误语义判断。

其中 **30 次生成到 2048-token 上限**。原始文本是自由比较反馈，常继续复述 `[TO EVALUATE]` 或重复前面的解释，未产生可接受的完整 JSON。其余两次正常 stop：`train:1` relevancy 在裸 JSON 前有长段自然语言，`train:10` relevancy 同样有前缀 prose，并在 JSON 中写了反引号包裹的 `null`，不能按原协议解析。输入只有 **885–2,169 tokens**，加输出预算仍远小于 32K；这是输出格式/终止行为的问题，不是超出上下文。

实际累计生成 62,389 output tokens，而 Selene 为 11,879；较多输出主要消耗在未完成的协议响应，不能作为更充分推理的证据。Prometheus 在自己的原生 rubric 格式下 10 次全部正常结束，说明它可以完成对应原生 HTTP 任务，但没有可靠服从这里的官方 JSON 双答案模板。

独立官方路径重放进一步确认：即使暂时不看 finish gate，32 次全部在原官方 `postprocess_response` 之后 JSONDecodeError，均未到 DTO；官方 MetaEvaluator 的 96 条结果与本地严格结果一致。`normalized_v1` 也恢复 0/32，因为它不修复截断文本、prose 前缀或非法 JSON。没有借格式规范化替模型补答案，原总分仍为 0。

因此，这次可比较的结论是“Selene 更能遵守当前 GroUSE JSON 协议，但其语义可靠性仍不足”；不能写成“Prometheus 对所有语义都判断错误”，也不能据此否定论文在其他 prompt/适配和样本范围下的结果。单纯增加输出上限尚无有效性证据，应先验证面向该模型的明确协议适配；若改用其原生 rubric 包装，需要单独版本，并保留 GroUSE 的 nullable 适用性与六项条件含义。

## 下一步如何改进，保持比较可审查

1. **保留已完成的双模型对照。** 本轮显示跨家族换模型本身未解决 scope 漏检和分数锚点漂移。继续分别比较原生格式和 GroUSE 格式，不用某一协议优势抵消另一协议的语义漏检。
2. **将格式适配作为独立协议变化。** Selene 可先真实重跑已实现的 `normalized_v1`；Prometheus 需要不同于字符串规范化的 schema/原生格式适配。必须重新运行 train，保留旧版作对照。格式完成率提高不代表上述 completeness/范围错误已修复，更不能补造原分支未调用的响应。
3. **把“概括后给分”改为可核查的原子判断。** 下一原生 rubric 实验要求先列关键声明的对象、量值、单位、条件，以及支持它的来源 ID/原句，再按冻结锚点给分；尤其区分 unsupported scope expansion 与措辞不同。可复用现有表格行关系与来源偏移作为检查材料，但要标记是协议扩展，不能让被评回答自行提供的结论充当来源。不要将本次 1/10/11 MB 三个数字硬编码为通用规则。
4. **显式验证拒答与引用锚点。** 用成对的“来源能回答/不能回答、回答拒答/未拒答”开发探针检查 applicability；利用已有 citation_count=0 诊断核查模型理由与锚点的一致性，保留合理拒答不需要引用的情况，避免统一扣分替代语义判断。
5. **修改后用 train 开发，再冻结元评测。** 先比较真实错误是否减少、正确答案是否被误伤、调用与 token 成本；之后才冻结并运行未访问的 GroUSE test。项目已有 123 份 P3 回答可继续获得模型评分，但在 evaluator 尚不能识别这些关键错误时，不以宽松满分宣布质量非劣。

当前结论是需要改进 evaluator 的证据检查和评分校准；无需把范围缩为只报告接口成功，也不需要等待用户先提供人工分数才能继续开发。

## Selene 显式规范化后的完整 train v2

随后使用同一固定模型、stock SGLang、原始四模板、train16 和抽样设置，在新目录 `runs/remote-h200-judge/runs/selene-normalized-development-v2/` 完整重跑，显式设置 `parser_mode=normalized_v1`。旧结果保持不变，没有回填旧流程缺失的响应。全部 **51 次调用正常 stop、解析成功**；运行中记录 14 个字符串 `"null"` 和 1 个整数串的逐字段规范化，没有将非数值内容猜成分数。

六项条件通过数依次为 **11、5、10、15、12、11 / 16**，合计 **64/96=66.67%**，严格 v1 为48/96。新增的六次实际调用是 train:2/12 的 usefulness、faithfulness，train:4 的 usefulness，以及 train:8 的 faithfulness。train:4 的 usefulness 已不适用，因此没有虚构第七个调用。

这不是纯解析变更的因果效果：两次运行共有45个调用，其中仅20个原始文本逐字相同。四个原已解析 metric 的数值也发生变化（train:6 relevancy 3→2，train:7 relevancy 4→5，train:11 faithfulness 1→0，train:14 completeness 3→4）；温度0.01及条件调用序列均应保留在解释中。冻结旧45条响应的离线规范化能单独证明类型兼容性，完整新运行则提供之前没有的实际下游判断，两者不混算。可复算脚本为 `runs/open-judge-review/compare_normalized.py`，原调用、变更与新增调用列表均保留。

完整性仍只通过5/16，说明格式问题解决后语义缺口仍然明显。没有运行 GroUSE test，也没有把新分数用于宣布项目回答质量非劣。下一步继续[原子声明与条件范围核验](atomic_judge_proposal.md)及完整开发回答对照；支持率、引用是否正确、拒答适用性和答案完整性分别检查。

独立脚本 `runs/open-judge-review/independent_normalized.py` 在应用已声明转换后，直接执行固定官方 DTO、条件流程和 MetaEvaluator，51 次调用及96项条件结果均一致。若不先转换，官方 DTO 会拒绝其中8次，所以66.67%属于明确标名的 normalized_v1，不能冒称原始官方协议成绩。两次数据、模板、模型revision声明和抽样参数一致，配置仅新增 parser_mode；运行记录仍是操作者来源声明，不等于远程权重认证。
