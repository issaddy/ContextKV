# 原子声明与条件范围核验：下一版实验提案

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

本提案针对 [首轮独立 judge 验证](open_judge_validation.md) 的具体不足：Selene 与 Prometheus 都可能只注意正确数字和流畅表述，却忽略对象范围；引用评分也会与评分锚点不一致。下一步不是修改这些旧分数，而是试验一个**先忠实抽取回答声明，再核查完整限定条件与原始来源**的独立协议。`native_v1`、GroUSE strict_v1 及已有结果均不变。

本轮只查官方论文、作者源码/模型元数据并准备本地草案，没有调用 GPU、下载权重、训练模型或读取任何 test 样本。可复跑准备脚本为 `runs/open-judge-proposal/prepare.py`；完整原始 prompt/schema 草案在同目录 `protocol.json`，13 类、26 个成对合成开发探针在 `probes.jsonl`。这些探针的预期来自 assistant 显式构造的合成世界，不是实际人工评审标签；未生成任何模型分数。

## 复用优先级与边界

| 方法与固定来源 | 可复用部分 | 本项目需要的适配 |
|---|---|---|
| RefChecker，作者代码 `1df1b25cee792ba2b171302e31ca4f768bd67703`，Apache-2.0，含 NOTICE | `RCClaim` 的声明与原句关联；triplet/subsentence 抽取；Entailment/Contradiction/Neutral 核验；显式聚合 | 必须额外保留条件、否定、版本、单位和整体/单项范围。原 checker 的“任一来源支持就 entail”不能直接覆盖有适用版本冲突的情况。复用少量 prompt/数据合同即可，避免安装整个历史依赖栈 |
| FActScore，作者代码 `f28272deffcf33efc1f1117d5479c10bb75221a9`，MIT | 原子事实逐项支持判定、记录回答覆盖情况 | 默认 Wikipedia 检索不能替换本轮实际可见证据；`gamma=10` 的短回答惩罚不适用于正确简短回答或必要拒答；事实精确率不是完整五维质量 |
| VeriScore，作者代码 `8714bca27b944b9659d6a966cdb92fb6fff8f72d`，Apache-2.0 | 选择可核验内容、带邻近上下文的声明抽取；保留适用性而非把所有句子都当事实 | 官方代码用前 3/后 1 句上下文与 spaCy，检索依赖 Serper/Google。这里复用上下文保留原则与模板结构，替换为现有可见来源，不能访问网络“补证据” |
| Claimify，ACL 2025 原论文与 Microsoft Research 官方说明 | selection→disambiguation→decomposition；抽取必须被回答原句蕴含；不能随意拆解集体范围或丢掉例外条件 | 本轮没有定位可直接复用的 Microsoft 官方实现代码，不能将第三方同名仓库冒称作者实现。采用方法原则，写项目自己的 prompt，不复制不明许可代码 |

RefChecker 的三阶段接口和代码可直接审查；仓库已于 2026-04 归档，所以固定小范围复用更合适。其 `aggregator.py` 提供 soft/strict/major 三种规则，空声明为 Abstain；其源码并不提供“所有质量维度都可靠”的保证。[作者仓库](https://github.com/amazon-science/RefChecker/tree/1df1b25cee792ba2b171302e31ca4f768bd67703)、[声明结构](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/base.py)、[核验 prompt](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/checker/checker_prompts.py)、[聚合](https://github.com/amazon-science/RefChecker/blob/1df1b25cee792ba2b171302e31ca4f768bd67703/refchecker/aggregator.py)。

FActScore 原方案主要研究长文本事实精确率；代码的回答比例和短回答惩罚应与支持比例区分。VeriScore 则指出过度原子化和把不可核验内容也计入分母会失真。这两点都适用于本项目，不能只复制一个平均公式。[FActScore 论文](https://arxiv.org/abs/2305.14251)、[固定评分代码](https://github.com/shmsw25/FActScore/blob/f28272deffcf33efc1f1117d5479c10bb75221a9/factscore/factscorer.py)、[VeriScore 论文](https://aclanthology.org/2024.findings-emnlp.552/)、[固定抽取器](https://github.com/Yixiao-Song/VeriScore/blob/8714bca27b944b9659d6a966cdb92fb6fff8f72d/veriscore/claim_extractor.py)。

Claimify 特别适合纠正此次“including”范围丢失：抽取出的命题必须由原回答支持，且需保留会改变事实判断的上下文。遇到无法确定的指代或集体关系，记录歧义而不是任选一种解释。原作者的实验优势是其数据上的结果，本项目尚未复现。[ACL 2025 论文](https://aclanthology.org/2025.acl-long.348/)、[Microsoft 官方方法说明](https://www.microsoft.com/en-us/research/blog/claimify-extracting-high-quality-claims-from-language-model-outputs/)。

## 较新的直接候选：具体取舍

**RAG-Zeval（EMNLP 2025）**与本任务最接近：一次生成包含声明、是否受支持及 grounding evidence 的结构化评审轨迹；用排序奖励强化学习训练，而不是凭现有 judge 的绝对分数作奖励。论文的 correctness 以参考答案为支持来源，faithfulness 以检索材料为来源，两者不能在本项目中混写为无参考 correctness。论文报告在 WikiEval 与跨领域配对数据上的改进，不代表已解决我们的表格范围错误。[正式论文](https://aclanthology.org/2025.emnlp-main.1267/)。

可立即复用的是“声明—支持状态—原文证据”协议，以及同时验证格式、证据摘录与配对顺序的实验设计。**本轮未定位可下载的作者 checkpoint**：GitHub 搜索到同名 `leekum2018/rag-zeval`，API 返回 size=0、commits=409（空仓库），Hugging Face 的 RAG-Zeval 搜索无模型；不能把论文“code/checkpoints available”的陈述当成实际已就绪。本提案不伪造 revision 或许可，也不启动从头 RL 训练。[当前同名仓库](https://github.com/leekum2018/rag-zeval)。

**ConsJudge（2025）**有真实作者代码与权重。官方 `OpenBMB/ConsJudge` 固定 `951d1748a94268b4eb83f1455fd2849a31d3ad35`，MIT；通过不同维度组合产生判断，再利用一致性筛选 DPO 偏好数据。优先借用的实验是：同一答案在 groundedness 单维与 groundedness+scope 组合下是否发生无法解释的翻转，以及交换候选顺序后判断是否稳定。一致性不是正确性的证明，不采用同模型多数票作为独立真值。[论文](https://arxiv.org/abs/2502.18817)、[固定作者仓库](https://github.com/OpenBMB/ConsJudge/tree/951d1748a94268b4eb83f1455fd2849a31d3ad35)。

作者 README 链接的两个模型 API/config 可访问、首个权重分片未鉴权 HEAD=200，尚未下载：

| 模型 | 固定 revision | 可用性与适配限制 |
|---|---|---|
| `sanwuge/ConsJudge-llama` | `547a344231cbfaee19984ad573ad274fffabf525` | Llama-3-8B、BF16 文件约 16.06 GB；config context=8192，非 Llama 3.1 的 128K；卡片声明 MIT，但正文为空，需保留基座身份 |
| `sanwuge/ConsJudge-qwen` | `c4d7d2fbca0b79800dde1002aadfa0d27e9de437` | Qwen2.5-14B、BF16 文件约 29.54 GB；卡片 MIT；同 Qwen 家族，不作为主要跨家族证据 |

其训练判断 prompt 主要依据 ground truth 在四个选项中选 Best/Worst，不是现成的无参考五维绝对评分器；作者训练脚本还固定旧 vLLM、设备号和 3000 长度，不能直接运行这些脚本并沿用当前服务合同。ConsJudge-llama 的 8K 上下文也不足以直接复用当前约 7.2K 输入 +1536 输出预算。若后续比较，使用固定权重的正确原生任务并记录新协议，不能悄悄截掉历史或来源。[Llama 模型](https://huggingface.co/sanwuge/ConsJudge-llama/tree/547a344231cbfaee19984ad573ad274fffabf525)、[Qwen 模型](https://huggingface.co/sanwuge/ConsJudge-qwen/tree/c4d7d2fbca0b79800dde1002aadfa0d27e9de437)、[作者判断构造代码](https://github.com/OpenBMB/ConsJudge/blob/951d1748a94268b4eb83f1455fd2849a31d3ad35/src/ConsJudge_train/hybrid_evaluation.py)。

**Auto-Judge（SIGIR 2026）**是更合适的跨任务元评测后续方向，能避免只围绕 GroUSE train 的单一主题开发。它提供任务/运行/人评及可复现的 judge 变体、排名相关性和 TIRA 流程；它不是一个可以代替 Selene 的新权重。[正式论文与作者存档](https://eprints.whiterose.ac.uk/id/eprint/243700/)。官方 starter 固定 `269778a88fe6db4f9cbfd98f69bca7ec239ea20b`，README 声明 MIT；base 固定 `41b1d016dfe4e5b9e4335552d63afb2b252c1519`，本轮 GitHub license 元数据为空，直接复制其代码前应以对应文件声明为准。论文为 CC-BY-4.0，数据有单独的底层语料/CC-BY-SA 声明，不能混同。

当前优先借用其 JSON/TSV 交换、变体命名和公开 synthetic kiddie 工作流合同。真实 release 需要访问资格，且有单独的数据查看边界；本轮未下载真实数据或打开 test 样本。不要为了扩大验证主题便把隐藏元评测数据读作开发材料。[固定 starter](https://github.com/trec-auto-judge/auto-judge-starter-kit/tree/269778a88fe6db4f9cbfd98f69bca7ec239ea20b)、[官方参与说明](https://trec-auto-judge.cs.unh.edu/)。

## 最小可实现协议

建议新名 `contextkv-atomic-evidence-draft-v1`，先作为独立离线诊断。第一版使用现有 HTTP 服务、JSON schema/终态检查和固定输入预算，不引入新大模型或整个 RefChecker 依赖栈。

**第一步：忠实抽取。** 输入当前 query、实际历史和带 unit_id/原偏移的回答片段；不把 sources 或黄金答案交给抽取器，防止它看到真值后把错误声明“纠正”为正确声明。每条输出保留原回答 quote、完整 statement、object/relation/value_raw/unit_raw/scope/conditions/negation/modality 与原 citation IDs。数值可为空，非数值事实也必须覆盖。每个输入片段必须对应声明或明确的未覆盖原因；代码/表格不能被静默当成“没有事实”。

原句定位复用 `bench.grounding.claim_spans` 的偏移合同，随后增加声明层；不把原有按句切分直接改名为原子抽取。模型返回的 quote 必须由宿主在对应原片段中绑定；有歧义或找不到原文时保留为抽取问题，不能重写原答案再判分。引用集合复用已有 parser，而非从新生成的 statement 猜测。

**第二步：来源核验。** 每条声明与完整适用来源进行核验，返回 `entailed / contradicted / insufficient / ambiguous / not_verifiable`，以及具体失败的 object/value/unit/scope/condition/version 槽位和原文 quote。缺失支持用 insufficient；只有来源给出相反命题才用 contradicted。`Failed` 是独立工程状态，不能当成“不适用”。同义改写不能因字面不同被判假，显式等价单位必须比较；含混单位不猜转换。

所有用于支持/反驳的 quote 必须定位到原始 `audit_evidence`，表格行关系可用现有 `adapters/table_relations.py` 提供的来源偏移帮助读取，但不能把派生关系本身伪装成新增来源。若多段联合才足以支持，要保留联合证据；若适用来源相互冲突，不能简单取最大的支持分数。

**第三步：引用与当前任务单独检查。** 对全部可见来源计算 visible verdict，再对原答案实际引用的子集计算 citation verdict。支持存在于别处不意味着引用正确。历史 assistant 文本只用于理解指代/对话问题，不作为事实支持。先前用户问过的问题也不自动成为当前回答必须再次完成的要求。必要拒答另判“来源能否回答当前问题”，避免无声明或拒答自动获得满分。

**第四步：报告而非伪造五维标签。** 先保存声明状态、抽取覆盖、引用状态、关键范围错误及耗时/token；支持比例只在已评声明中描述，零声明或未覆盖不设为 1。它不等于 correctness，也不能衡量遗漏问题。若继续做五维分数实验，采用同一个 native_v1 rubric 加/不加该原子核验记录的明确对照，并另存新协议；不要把平均支持率直接填满五维，不让多个正确旁支声明掩盖中央数值/条件错误。

## 具体 prompt 草案与工程复用

`protocol.json` 已给出可直接作为 system message 的完整原创 extraction/checker prompts 与返回 schema。抽取 prompt 明确禁止纠错、补事实和删条件；checker 按对象→关系→值/单位→范围→否定/条件检查，要求原文证据与实际引用子集分别判定。没有把本次 Cloudant 的 1/10/11 MB 硬编码进去，也没有复制论文中不明许可的整段 prompt。

本地准备命令：

```bash
python runs/open-judge-proposal/prepare.py
```

它只写草案与请求文件，不发送请求：

- `extract-requests.jsonl`：26 条，只有 query/history/answer units；
- `checker-component-requests.jsonl`：24 条短声明的来源核验组件请求；这使用给定短回答作为声明，不能称已验证端到端抽取；
- 两个 JSON 代码探针交给结构核验，不送原子 prose checker；
- `probes.jsonl` 中的 `offline_expected`、family、探针标签均不在模型 messages 中。

可直接复用的已有代码是 `judge_payload`/`evaluation_binding`/`rule_audit`、`NativeJudgeClient.complete` 的 HTTP/终态/usage 记录、固定 tokenizer preflight、`grounding.claim_spans`、citation parser、原始完整 audit 正文、`MiniCheckScorer` 和 `bench.code_audit`。输出绑定沿用同一答案/query/input_hash；不引入重复的每请求多层 hash。

MiniCheck 继续使用当前固定官方实现与权重，并只接去掉 citation 标记的语义声明，保留原偏移和引用关系。它可提供独立的支持信号与 disagreement 列，不能单凭低支持概率区分“矛盾”与“证据不足”；按单段评分也不能排除跨段联合支持。当前真实引用修复审查已经说明，不应让同一 MiniCheck 既挑修复、又充当独立成功证据。[现有实现](../../bench/grounding.py)、[实际反例审查](../research/grounding_v2_assistant_audit.md)。

## 通用开发探针与最小对照

13 类各一对：对象范围、per-item/aggregate、显式单位等价、版本/角色条件、否定、脚注例外、联合来源、历史污染、错误引用子集、当前问题与旧问题混淆、合理拒答、错误拒答、代码结构。数值和对象覆盖配额、保留期、端口、任务数、时间、大小等，既有支持例也有难负例。对象范围扩大但来源未明确反对时，预期为 insufficient，不偷换为 contradicted；这正是本次真实错误需要的区分。

最小对照只保留三组：现有 native_v1；新原子核验；新原子核验 + MiniCheck 独立信号。若预算允许，再做同样 rubric 的维度组合/候选顺序一致性诊断，借鉴 ConsJudge，但一致性不能替代来源判断。所有实验都保留失败、抽取遗漏、假阳性和调用成本。

先在 26 个明确定义的合成 dev 探针与两份已见真实 probe 上检查抽取忠实性、对象/条件区分、正确例误伤率、原文 quote 可定位率；随后扩到固定 Cloud dev 样本。合成探针与已见案例属于开发测试，不能当作新独立效果证据。冻结改进后的 evaluator，再运行此前未访问的元评测；Auto-Judge 的合规公开开发资源可补充主题多样性，隐藏评测继续保持未见。

## 已实现的显式可选诊断入口

`bench/atomic_judge.py` 已将上述主流程实现为独立入口，固定完整协议位于 `bench/assets/atomic_judge/protocol.json`。研究草案文件保留，实际运行以这个打包资产为准。它不是 RefChecker 的完整复现，也不是生产评分 gate。

```bash
python -m bench.atomic_judge \
  --audit PATH_TO_SAVED_AUDIT.jsonl \
  --config PATH_TO_NATIVE_TRANSPORT_CONFIG.json \
  --tokenizer-path PATH_TO_FIXED_LOCAL_JUDGE_TOKENIZER \
  --output NEW_ATOMIC_LEDGER.jsonl
```

配置复用 `NativeJudgeConfig` 的模型身份、服务地址、采样和预算字段；native_v1 的评分模板/prompt/NA 规则不用于原子请求，实际完整原子协议单独记录。若设置 `context_window`，必须提供精确 tokenizer 计数；CLI 使用 `local_files_only=True`、`return_dict=False`，不会下载文件。可以为新实验另存输出上限更高的配置以容纳结构化抽取，但需要完整预算检查，不能截掉来源来适配。

API 为 `AtomicJudgeClient(config, transport=None, token_counter=None)` 与 `await client.evaluate(audit)`，结束后 `await client.aclose()`。每份答案先调用一次抽取，再按唯一且已绑定的声明检查全部可见来源；实际引用是严格子集时再单独调用引用来源核验，集合相同则复用已有结果。每一步都保留原始响应、finish reason、usage、parsed 内容及失败原因。提取器接不到来源真值；checker 同时接到实际历史以核对消歧，但历史被明确标成语境且不能作为 supporting source。

原文 binding 只证实位置，不推导蕴含。checker 必须额外输出 `extraction_faithful`：抽取删改了范围或该项无法判断时，结果保持 unresolved。object/relation/value/unit/comparator/scope/conditions/negation/modality/citation 槽位都在模型输出入口校验，未知可显式 null，不能静默省略。源 quote 必须位于本次检查所给来源中；引用子集检查不能引用只在全可见集合里的另一段材料。

返回 `source=atomic_judge_diagnostic`、`diagnostic_only=true`、`formal_quality_eligible=false`，没有五维 `scores`；`support_fraction` 与 `quality_score` 仍为 null。覆盖率只是原回答单元是否有已定位声明/明确未覆盖说明，不证明所有事实和限定词都被抽取。代码、表格和独立段尾引用组保持未覆盖/部分状态；缺引用只记录 missing，不改变 visible verdict。完整执行状态也不意味着答案正确。

专项测试覆盖端到端 fake HTTP、输入标签隔离、完整来源/实际历史、唯一原文位置、必需槽位、无效/越界 quote、联合来源、可见与引用子集、精确重复声明、拒答、代码、段尾引用歧义、截断、预算和文件防覆盖。尚未执行真实原子模型评审，未改任何默认策略、native_v1 分数或正式质量口径。
