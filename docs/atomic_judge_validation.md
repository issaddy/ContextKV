# 原子评审开发验证

本轮已完成 v1、v2 与 guided v3 三次完整 29 条开发验证。**固定 XGrammar 的 schema 编译和生成可用，消除了 v2 的两个顶层 JSON 失败，但当前模型的语义判断未改善**：v2/v3 的 24 条 prose 都是 8 matched、5 mismatched、11 unknown，真实例仍有对象混淆和正确短答误伤。这里不改 `native_v1` 的五维协议，不回填既有分数，也不构成人工验收或 test 非劣结论。v1 源码 commit 为 `dc6b8170d0a8e2619b5f187f21d26d3a05410c92`，guided v3 为 `9db698a7df088fb3071032379a43c577efece36d`。方法来源与实现边界见 [原子评审提案](atomic_judge_proposal.md)，原生评审对照见 [开源 judge 验证](open_judge_validation.md)。

## 输入与冻结边界

执行输入为 `runs/atomic-development-inputs-v1/{audit,labels}.jsonl`，共 29 条：26 个 assistant 显式构造的合成世界探针，以及三个已完成的真实冻结答案。期望标签、方法臂名及生成来源仅在 sidecar，提取与 checker 的白名单输入不含这些标签。

合成输入标记 `measurement_kind=simulation`，没有伪造生成器 input IDs、output IDs 或 receipt。13 组正反对照覆盖对象范围、逐项/总量、单位换算、版本与角色条件、否定、脚注例外、联合来源、历史答案污染、引用子集、当前问题、正确拒答、错误拒答和代码结构。其中 24 条属于 prose 检查；两条 JSON 结构对照走已有静态语法诊断，不能计入 prose 语义正确率。所有期望是合成定义，标签来源不是人类评审。

三个真实输入来自 `runs/table-frozen-pairs-v2/cases.json` 和 `h200-order-pairs/measurements.jsonl`：`old_error_history`、repeat 0、`off / all_rows / focus_refs`。独立构造器 `runs/atomic-development-review/prepare_inputs.py` 已逐项核对实际 input IDs、既有 token hash、receipt 与输出，且三者的 query、actual history、可见来源及原文相同。执行侧使用的原始生成长度分别为 277、218、114 tokens，均未截断；这些长度只作为离线来源信息，不是评审标签。

导入时三条真实生成的 `unvalidated_hardware_run` 标记被标准化成 `hardware`，差异记录在 `source-marker-note.json`，未改变模型输入。分析保留这个来源限定：这些生成并未因此成为已验证的性能或质量测量。原 audit 和响应均未回填修改。

Selene 的模型为 `AtlaAI/Selene-1-Mini-Llama-3.1-8B`，revision `427792f1c3e2073cb7da216924fd884b1ba496e0`；本轮提前固定 temperature 0、单调用输出预算 4096、上下文窗口 32768，并以固定本地 tokenizer 完整计数。输入不会为了容纳模型而截掉来源。Atomic 使用自己的提取与检查 prompt；配置中的 native protocol 名称只来自复用的运输配置，不表示本轮使用五维 Result 评分模板。

## 结果分析口径

只读脚本：

```bash
python runs/atomic-development-review/analyze.py \
  --audit runs/remote-h200-atomic/runs/atomic-development-v1/audit.jsonl \
  --labels runs/remote-h200-atomic/runs/atomic-development-v1/labels.jsonl \
  --results runs/remote-h200-atomic/runs/atomic-development-v1/ledger.jsonl \
  --output runs/atomic-development-review/analysis-v1-rerun.json
```

脚本保留全部 29 条输入分母，缺失、截断、schema 错误、定位失败和 unresolved 都不删除；无结果预检已确认 29 条全部保留，24 条 prose 全为 unknown。原始响应、finish reason 和 usage 保留在实际结果文件，派生分析引用该文件及 SHA。完整派生结果为 `runs/atomic-development-review/analysis-v1-reviewed.json`；最初生成的 `analysis-v1.json` 也保留。两者语义标签统计相同，前者细分了绑定与 evidence/verdict 自相矛盾的错误，并带上来源 marker 说明。

分析分为四层，不能用下层成功替代上层正确：

1. 运输与结构：HTTP、完成标志、上下文预算、JSON、必需槽位是否成立。
2. 原文绑定：claim 的 answer quote 是否定位，checker 的 quote 是否属于本次 visible/cited 来源子集。定位只证明来源，不能证明蕴含。
3. 抽取覆盖：原文单元是否未处理、声明不可验证、需代码/表格另审，是否有无法绑定的引用。已有单元被覆盖，也不能证明所有限定词都保留。
4. 模型 ledger 与预期的对应：正例误伤、反例漏检、拒答的来源可答性，以及相同事实在全部可见来源和被引用子集上的差异。failed-slot 原字符串、联合来源数量作为定位诊断单列，不能因字段同义名称不同把正确语义判断强改为错误。

完整但判错与未能评估分别统计。引用缺失不会变成事实错误；相同 visible/cited 集合复用的检查只计一次 HTTP 调用。`matched` 仅表示当前 ledger 标签对应合成预期，仍需检查抽取是否丢失作用域或条件，不能称语义验证通过。开发例已见、规模小，也不提供总体准确率或置信保证。

## 三个真实答案的逐源预审（assistant audit）

五个可见段落中，`E_53079fa73a` 与 `E_fe06da00dd` 是正文相同的限额资料：表格分别列出 Individual Document Size、Single Attachment Size、Request Body Size，对应 1、10、11 MB；正文建议把二进制附件或大 JSON 放到对象存储，并说明超限时不复制。两段重复正文不能算独立的证据确认。

`E_8312203498` 讨论文档保持小尺寸及文档增长带来的性能问题，没有在 1 MB 后写“包含附件”；`E_bd58806373` 介绍附件和内联 BASE64，提醒附件大小、数量影响性能；另一个 Discovery 段落讨论文档切分，与 Cloudant 附件上限无关。旧 assistant 历史含错误概括，只能用于理解用户纠正的对象，不能提升当前 claim 的事实支持度。

| 实际答案 | 待验证的具体边界 | 引用边界 |
|---|---|---|
| `off` | 单附件 10 MB 有直接支持；重复出现的“文档包含附件总计 1 MB”额外扩大了表格对象范围，不能因为 1 MB 数字出现而判完整支持。keep-small 和对象存储建议有来源，其他性能概括仍需逐项核对。 | 没有原始 Evidence ID 引用；这与事实判定分开。 |
| `all_rows` | 三项限额分别列出更贴近表格，但“组合文档和附件总量”仍是关系解释而非表格原句；末尾“不影响整体性能”的保证超出了来源的提醒。不能把整篇自动视为完美正例。 | 没有原始 Evidence ID 引用。 |
| `focus_refs` | 核心答案只指单附件 10 MB，与表格直接相符，是识别正确短答误伤的重要对照。 | ID 出现在独立的 Evidence 行，引用组与后续句子的绑定可能仍 unresolved；不能把字符串中出现 ID 当成逐 claim 绑定完成。 |

以上是 assistant 对原始来源的检查记录，未填任何 human 分数，也没有把三个答案设置成模型可见的正负标签。

## v1 实测结果

29 条都有结果：8 complete、21 partial；42 次独立 HTTP 均为 `stop`，包括 29 次提取与 13 次 checker，复用的 cited/visible 检查不重复计数。合计输入 36,662、输出 5,643 tokens，实际调用输入为 508–5,171 tokens，未碰到 32,768 窗口与 4,096 输出预算的边界。没有把失败归因于上下文截断。

| 合成 prose 分组 | 冻结输入数 | ledger 标签对应预期 | 对应错误 | 未完成所需判定 |
|---|---:|---:|---:|---:|
| 正事实对照 | 12 | 3 | 0 | 9 |
| 负事实对照 | 10 | 3 | 1 | 6 |
| 正确/错误拒答 | 2 | 0 | 0 | 2 |
| 合计 | 24 | 6 | 1 | 17 |

这不是“6/7 准确”：17 个 unknown 仍在分母中，三个真实例全部 partial，且完整的单元记账不等于语义可靠。两条代码控制的静态 JSON 检查分别识别合法 JSON 和尾逗号错误；模型试图把代码单元作为 claim，协议拒绝，不能把静态检查的成功加到 prose 结果中。

各类别没有被筛掉：object_scope、version_role 两组正反例均得到预期 ledger verdict；per_item_aggregate 为一个 unknown、一个 mismatched；unit_equivalence、joint_evidence、history_is_not_evidence、citation_subset、current_question、两类 abstention 均未完成该组完整判定；negation、footnote_exception 各只有一条完成所需判定。此处的 unknown 可同时含正确的局部 verdict 与未完成的忠实性/覆盖检查，不应擅自补为正确分。

结构与定位问题：

- 23 个 claim 被 `value_raw` 的 string-or-null 限制拒绝，实际模型返回的是 7、10 等 JSON 数值；这属于表示不兼容，不是事实错误。另有一项缺少 `kind`。
- `focus_refs` 的两个 claim 从其他单元带入引用 ID，被 v1 整体拒收，导致本可单独检查的可见事实丢失。引用关联问题应独立记录，不能扩展 cited 来源集合，也不必丢掉语义 claim。
- `all_rows` 的五个 statement 原样复制了 schema 示例字符串，而非真实回答。数值类型修复不能自行证明这些抽取忠实。
- 17 条出现未记账单元；两个代码控制都尝试抽取结构单元。覆盖不足不只来自校验器，也来自模型漏抽、漏写 uncovered。

以下语义失败保留原始结果，没有用规则修正分数：

| 探针 | 可复核问题 | 与工程失败的区别 |
|---|---|---|
| `p004` 四 worker 总量 | 原回答说“四个 worker 合计最多 7”，抽取改成“一个 worker 最多 7”；checker 仍标忠实且 insufficient，忽略来源明确的合计 28。 | complete 也会错；范围保留与忠实性判断均失败。 |
| `p010`、`p020` | 原始断言分别保留了错误否定关系/错误单位，checker 给出正确的 contradicted，却把“与来源矛盾”当作 extraction_faithful=false。 | 抽取忠实与事实真实被混淆，协议保持 unresolved，不能自动改回 true。 |
| `p015` 历史污染 | 当前正确答案及来源均为 80；checker 理由转而引用历史 assistant 的 400，并给 insufficient/不忠实。 | 即使 prompt 说历史非证据，模型仍会混淆对象与消息角色。 |
| `p021` 正确拒答 | 被抽取成 assertion，返回 insufficient，没有来源可答性判定。 | complete 不代表拒答维度已验证。 |
| `p024` 错误拒答 | 已识别来源 14 天与拒答矛盾，但 source_answers_current_question 仍为 null。 | 必需的拒答合同未完成，不能靠局部 verdict 补分。 |
| `real-003` 正确短答 | c3 对“单附件 10 MB”返回 contradicted，同时引用“Single Attachment Size 10 MB”、关系 supports，理由也复述同一值。 | verdict 与证据关系自相矛盾；宿主保留失败，没有用 quote 命中把它改为 entailed。 |

## v2 最小修正与待验证边界

在完整保存 v1 后，新的独立协议升为 `contextkv-atomic-evidence-v2`：

- `value_raw` 接受字符串、有限 JSON 数值或 null，原样传递与保留，不强制转成字符串；boolean、非有限数与容器仍在外部边界拒绝。
- 模型声明的额外引用不再丢弃语义 claim。原声明、原单元 ID 与额外 ID 分开记录；checker 的 citation_ids 和 cited 来源集合始终由宿主从原单元解析。额外 ID 不产生支持证据，仍让整体 ledger 保持 partial。
- 提取和检查模板改成字段说明，去掉容易被复读的示例答案值；保留条件、scope、模态、精确原文定位、抽取忠实性与证据支持要求。没有依据本轮错误修改已有分数或自动修答案。

本地 atomic/native 75 项针对测试通过，覆盖数值原样保留、非法结构、借入邻单元 ID、可见来源与引用子集分离、quote 精确性及旧有泄漏边界。随后已对同一 29 条完整重跑 v2，结果如下；格式可用性改善没有解决历史污染、范围推理或正确例误伤。

## v2 完整重跑：可评估范围增加，语义错误仍明显

原始结果在 `runs/remote-h200-atomic/runs/atomic-development-v2/`，只读派生结果为 `runs/atomic-development-review/analysis-v2.json`。仍为同一固定模型、temperature 0、4096 输出预算、完整 29 输入，没有选择性重跑失败项。58 次 HTTP 全部 `stop`，输入 68,276、输出 9,920 tokens；单调用输入最多 5,341、输出最多 1,009 tokens。两个 failed 不是输出截断。

| 指标 | v1 | v2 | guided v3 |
|---|---:|---:|---:|
| 全部输入 | 29 | 29 | 29 |
| complete / partial / failed | 8 / 21 / 0 | 15 / 12 / 2 | 15 / 14 / 0 |
| 24 prose：matched / mismatched / unknown | 6 / 1 / 17 | 8 / 5 / 11 | 8 / 5 / 11 |
| 12 正事实：matched / mismatched / unknown | 3 / 0 / 9 | 5 / 4 / 3 | 5 / 4 / 3 |
| 10 负事实：matched / mismatched / unknown | 3 / 1 / 6 | 3 / 1 / 6 | 3 / 1 / 6 |
| 2 拒答：完成原协议要求的判定 | 0 | 0 | 0 |
| 实际 HTTP 调用 | 42 | 58 | 67 |

数值表示拒收消失，模型也不再复制 statement 的示例占位词。由于更多 claim 进入 checker，正确例误伤被显露出来，不能把 complete 增长当质量改善。尤其：

- `p005` 正确的 2048 KiB→2 MiB 被判 insufficient，理由要求来源逐字写出换算后单位，忽略已给出的换算关系。
- `p011` Enterprise 300 秒被判 contradicted；理由同时复述“默认 90、Enterprise 300”，却仍声称与 300 矛盾，说明条件范围推理失败。
- `p017` 的全部可见来源包含明确端口事实，仍被判 insufficient；只传正确被引用来源时又得到 entailed。无关来源的加入暴露不稳定性，不能反过来把 cited 结果覆盖 visible 结果。
- `p019` 的正确 8 秒与来源逐字一致，模型仍称来源没有 backoff 信息；历史问题干扰没有解决。
- `p004` 的抽取现在保留“四个 worker 总量”，checker 却忽略来源明确的总量 28，仍判 insufficient。该例不再只是抽取错改。
- `p015` 将历史 assistant 的 400 伪装成 E_01 原文引文，被 exact quote 检查拒绝。这说明来源绑定有实际价值，但不等于 judge 已摆脱历史污染。
- `p010/p012/p014/p016/p020` 等继续把事实与来源不符和抽取不忠实混淆。`p013` 的理由甚至承认 96 小时相当于 4 天，仍判信息不足。
- `p021/p024` 两种拒答均被抽成 assertion；错误拒答还给出“来源没有保留天数”的错误理由，忽略明确的 14 天，不能因为 source_answers_current_question 出现布尔值就计为合格拒答评估。

三个真实例没有获得可接受的整体评审：`real-001` 提取输出两个连续裸 array，`real-002` 输出一个裸 array，均因顶层协议不符停止，原文未被拼接或补壳。`real-003` 的三个 claim 均进入检查，但三个 verdict 都为 contradicted、引用关系却是 supports；最后一个理由还虚构原回答说过 50 MB。原答案只说单附件 10 MB。因此必须同时保留格式失败与正确短答误伤，不能只看前两例被格式挡住。

## 可选 v3：复用固定运行时的 JSON Schema 约束

该可选模式用于减少顶层形状、缺字段和类型造成的损失，并观察对语义行为的影响。v2 的 prompt 与自由输出默认行为保持原样；新模式不手工修 JSON，不在失败后重试自由生成，也不覆盖 v1/v2 结果。完整开发对照的结果见下一节，未证明它能修复条件、换算、来源角色或拒答判断。

固定 SGLang commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 的实际调用链已核：

1. OpenAI 兼容接口接受 `response_format.type=json_schema` 与 `json_schema.{name,schema,strict}`。`strict=true` 明确请求约束；普通 chat 路径会将 schema 写入采样参数，而 `json_object` 只给顶层 object，不能保证所需字段。[固定协议定义及转换](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/entrypoints/openai/protocol.py#L223-L236)、[采样参数转换](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/entrypoints/openai/protocol.py#L1180-L1191)。
2. XGrammar backend 先检查不支持的 schema 特性，再调用 `GrammarCompiler.compile_json_schema`；编译错误产生无效 grammar，而不是成功的自由输出。固定版本的检查器拒绝 `if/then/else/not/contains/dependent*/uniqueItems/multipleOf` 等特性。本模块只用 object、required、additionalProperties、array/items、enum、简单 minLength 与 nullable type，没有加入语义蕴含规则。[固定 backend](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/xgrammar_backend.py#L384-L408)、[支持边界检查](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/json_schema_validation.py#L167-L202)。
3. 固定服务器依赖 XGrammar 0.2.7。其官方 `GrammarCompiler` 默认 `strict_mode=true`、`any_order=false`，编译 schema 到 tokenizer 对应 grammar 并复用编译缓存；C++ schema parser 支持 type 数组，因此 string/number/null 槽位可表达。没有把接口的 `strict` 布尔值误称成语义严格验证。[固定依赖](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/pyproject.toml#L97)、[XGrammar 0.2.7 compiler](https://github.com/mlc-ai/xgrammar/blob/82505d0d987c36a4209fb3d8571cf6b0f28b5acd/python/xgrammar/compiler.py#L131-L192)、[官方 type-array parser](https://github.com/mlc-ai/xgrammar/blob/82505d0d987c36a4209fb3d8571cf6b0f28b5acd/cpp/json_schema_converter.cc#L184-L225)。

实现接口为 `NativeJudgeClient.complete(messages, response_format=None)`；默认不发送 response_format。`AtomicJudgeClient(..., output_mode="json_schema_v1")` 与 `evaluate_file` 同名参数使用固定的 `bench/assets/atomic_judge/schemas.json`；CLI 加 `--output-mode json_schema_v1`。guided 模式单独标记为 `contextkv-atomic-evidence-v3-json-schema-v1`，保存完整 schema、base v2 协议、模式及每次实际请求的 response_format。没有从探针标签动态生成 schema，也没有把预期 verdict、来源 ID 或正确值固定进 grammar。

```bash
python -m bench.atomic_judge \
  --audit runs/atomic-development-inputs-v1/audit.jsonl \
  --config runs/atomic-development-inputs-v1/config.json \
  --tokenizer-path /absolute/path/to/fixed/Selene-tokenizer \
  --output-mode json_schema_v1 \
  --output runs/atomic-development-guided-v3/ledger.jsonl
```

83 项 atomic/native 测试通过，覆盖默认无约束请求不变、固定 schema 与独立协议身份、schema 失败不降级、length/裸 array 仍保留失败，以及结构合法但引文不真实仍拒绝。最初本地阶段仅运行了固定官方不支持特性检查器，记录在 `runs/atomic-development-review/schema-source-check.json`；后来在实际服务环境完成了短编译 probe 和完整 guided 运行，不能混淆这两层验证。

每次运行记录当次 `/get_server_info` 的实际 grammar_backend 与已安装 XGrammar 版本，v2 历史 runtime 不能代替新运行证明。若 tokenizer/backend 不支持，固定运行时可能拒绝请求；保留失败，不静默退回自由生成。schema、quote、覆盖及语义错误分别报告，不把 grammar 成功当作质量验收。[固定 backend 选择](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/base_grammar_backend.py#L371-L425)、[编译失败处理](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/constrained/grammar_manager.py#L145-L193)。

短 probe 已准备为 `runs/atomic-development-review/guided_probe.py`，默认只 plan，不连接服务。它预先固定第一条 `p001`，做一次抽取和一次 checker；checker 输入取该条冻结 v2 的首个 claim，不依据新抽取结果挑选样本。显式 `--execute` 后记录本机解释器包版本、实际服务配置、两个请求与完整原始 HTTP 响应/失败内容，并用服务环境已有的 jsonschema 检查返回形状。没有下载模型、重试或降级；本机包版本不会被当成远端环境证明。运行前应使用启动服务器的相同 Python 环境。

```bash
python runs/atomic-development-review/guided_probe.py \
  --audit runs/atomic-development-v2/audit.jsonl \
  --frozen-ledger runs/atomic-development-v2/ledger.jsonl \
  --config runs/atomic-development-v2/config.json \
  --tokenizer-path /absolute/path/to/fixed/Selene-tokenizer \
  --output runs/atomic-guided-probe-v3 --execute
```

该两调用 probe 仅核实际接口和 grammar 路径，不加入 29 条质量统计，也不产生质量通过结论。完整 guided 运行中仍保留每次 `requested_response_format`、HTTP 状态与原文响应，以便区分编译拒绝、生成中断和语义失败。

## guided v3 实测：结构可用，语义未改善

已独立检查导入目录 `runs/remote-h200-atomic/runs/atomic-development-v3` 与 `atomic-guided-compile-v1`。短 probe 的 extract/check 各一次，均 completed/stop 且返回通过固定 schema 检查；实际 runtime 为 XGrammar 0.2.7、SGLang `0.0.0.dev1+g3f36fc083`、Transformers 5.19.0、jsonschema 4.26.0，服务 grammar_backend 为 xgrammar。

v2/v3 的 `audit.jsonl`、`labels.jsonl`、`config.json` 三个文件逐字节相同，SHA 分别相同；其中 audit SHA 为 `2138d69c41e502ed24e2c332236ed64e4fefee94758731104ee3eb6ac6a10e8b`。两版提取和 checker prompt 文本也相同，区别为显式 schema 请求及独立协议身份。完整对照没有把裸 array 手工补成 object，没有回填旧结果。

v3 全部 67 次 HTTP 返回 200/stop，29 条提取均返回可解析顶层 object；总输入 117,222、输出 11,288 tokens。58 个共有调用阶段中，50 个模型 `raw_text` 完全相同；29 次提取中 24 次完全相同。变化的提取是 p024、p025、p026、real-001、real-002，共有 checker 中 p011、p012、p024 的原文发生变化；这些变化没有改变 24 条 prose 的最终对应情况。这里比较模型正文，不把带 request ID 等字段的 HTTP envelope 冒称逐字相同。

工程收益具体是：real-001/002 从 failed 转为 partial，新增 9 个 checker 调用。它们没有变成质量通过：

| 真实案例 | v3 实际覆盖 | 新暴露的问题 |
|---|---|---|
| real-001 / off | 6 个提取 claim；4 个进入 checker，另 2 个因 quote 去掉原文 Markdown 格式而定位失败；17 个原文单元只绑定 4 个 | 正确的“单附件 10 MB”和“每个附件最多 10 MB”仍被判 contradicted，引用却标 supports；有问题的“文档包含附件 1 MB”也没有得到可靠的范围判定。4 个 checker 均未形成有效 assessed verdict。 |
| real-002 / all_rows | 5 个 claim 进入 checker，13 个原文单元只绑定 5 个 | 三个附件 10 MB claim 得到 entailed；正确的“单文档 1 MB”却被附件 10 MB/请求 11 MB 混淆为矛盾。组合请求 11 MB 的关系解释也给出了与 supports 引文不一致的 contradicted。 |
| real-003 / focus_refs | 与 v2 的提取及三个 checker raw_text 全部相同 | 正确短答继续被误伤，仍有“原回答是 50 MB”的虚构理由；schema 并未纠正该判断。 |

本轮三条真实答案合计有 9 次 verdict=contradicted 但 evidence relation=supports 的无效结果；来源 quote 能定位仍不能转换成正确判分。另有未覆盖单元、错误拒答的类型识别、联合来源换算及历史污染问题。schema 使接口走通，但没有给出本项目语义准确性的改进证据；现在应保留该模式作工程选项，后续专用判别模型/数值关系路线见 [开源复用取舍](quantity_judge_reuse.md)，本轮未下载或启用新模型。

### tokenizer 长度告警的含义

完整 driver log 已保留为 `runs/remote-h200-atomic/runs/atomic-development-v3-driver.log`，仅包含一次 `5478 > 4096` tokenizer 警告与完成标记。它对应首个超出 tokenizer `model_max_length=4096` 元数据的输入，即 real-001 的第一个 checker；不是输出预算耗尽，也不是实际服务只允许 4096 tokens。

当前 `apply_chat_template` 路径默认 `truncation=False`，Transformers 这条警告仅在首次超过 tokenizer 元数据上限时输出，因此 5478 不是本轮最大输入。实际 ledger 最大输入为 **5501**，67 次客户端完整 token 计数全部与服务器 `usage.prompt_tokens` 相等；服务 runtime 为 **context_length=32768、allow_auto_truncate=false**。独立的输出预算是 **max_tokens=4096**，实际单调用输出最多 **862**，全部 finish_reason=stop。最大输入加预留输出预算为 9597，仍在当前服务窗口内。这些记录支持本轮没有因该警告发生静默输入截断或输出 length 停止，不能把通用 tokenizer 警告描述成已出现模型索引错误；也不能由这一次通过推断任意超长输入均安全。

独立派生文件为 `runs/atomic-development-review/analysis-v3-independent.json`；成对输入/原始文本/预算核查为 `paired-v2-v3.json`，可用同目录 `compare_v2_v3.py --output NEW_PATH.json` 重算。原始 v1/v2/v3、短 probe 和 driver log 均保留，没有修改任何模型响应。
