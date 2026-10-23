# Granite Guardian 4.1：固定29项开发验证

2026-10-24。原生 `ungroundedness` 协议已完成真实H200调用：修复tokenizer返回类型后，29/29正常结束；固定22个事实控制的二分类风险判断全部吻合，但真实长答案的对象包含关系与性能保证仍漏检，恰当拒答被误报。Granite可以继续作为独立诊断候选，**本次没有建立整体答案质量、引用正确性或质量非劣**。三个真实回答没有人工分数；下述原文审查是assistant依据保存来源作出的诊断。

## 运行、来源与可复核入口

原始两次运行保存在 `runs/remote-h200-granite/runs/granite-development-{v1,v2}/`，导入20文件的archive SHA为 `a8796b11ed1cf4f011a82e793da072ea4521176cd64534b43fda9be2fed1d582`。v1失败ledger与v2结果均保留，没有覆盖或挑样重试。独立只读分析为 `runs/granite-development-review/analyze.py`，派生结果为同目录 `review.json`；重跑不会调用模型、修改原结果或产生五维分数：

```bash
.venv/bin/python runs/granite-development-review/analyze.py \
  --root runs/remote-h200-granite/runs \
  --tokenizer-path /path/to/fixed/granite-guardian-4.1-8b \
  --output runs/granite-development-review/review-rerun.json
```

`--tokenizer-path`是可选的独立逐token重建项。本次使用固定revision的官方tokenizer文件在本地重建，29/29与提交IDs一致；只加载tokenizer，不加载权重。未提供该路径时仍核对完整来源、消息、wire请求与回显，结果明确不包含独立tokenizer检查。

模型为 `ibm-granite/granite-guardian-4.1-8b@ab01ccca5dcfb80246369a086a4a87a29198f5af`，官方prompt/helper源固定为 `0a4ec708fa916f3e9d7fc7e5b75dff836752f448`，Apache-2.0。没有将它包装成Selene五维1–5或Atomic claim schema；复用边界见[适配说明](granite_judge.md)和[NOTICE](third_party/GRANITE_GUARDIAN_NOTICE.md)。原生标准评估最后一个assistant文本，yes表示检测到不受来源支持的风险，no仅表示未检测到。[固定模型卡](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md)。

v2部署源码为 `3b852aac41c9e5dd7431c05ec373d732f57b6261`，已完整核验262个tracked文件。导出的 `runs/remote-h200-native-budget/runs/source-deployment.json` 保留同一部署清单；本次独立逐项与该Git树核对，262项全部相符。该提交包含 `856a753` 的修复，Granite三项实际组件hash同时匹配两者：Granite为 `6c7a3049b4083d9414a6be4597ee07aa58cd8b048194cb7a4faf953d74e21264`；native client与judge公共辅助也匹配。导入driver原文与manifest的 `c0df5cdcb9f267aed222cbd5765a7bd44fc6c26ef20e92536ff4062f2c69b2df` 相符。模型revision是固定本地snapshot与服务端声明，本次只读审查没有重新遍历校验远端权重字节。

## 失败保留与完整输入合同

| 项目 | v1 | v2 |
|---|---:|---:|
| 固定输入 | 29 | 29 |
| 实际HTTP请求 | 0 | 29 |
| 有效原生score | 0 | 29 |
| failed / 输出截断 | 29 / 0 | 0 / 0 |
| 实际输入tokens | 未进入模型 | 20,503 |
| 实际输出tokens | 0 | 348 |

v1的 `apply_chat_template` 默认返回 `BatchEncoding`，取list得到 `['input_ids','attention_mask']`。已有transport合同在发送前拒绝了它，29条均是同一 `input_ids must be actual nonnegative integer token IDs` 错误。v1不能计入模型格式失败、语义准确率或推理成本。`856a753`显式指定 `return_dict=False` 后完整重跑同29项；输入audit、标签sidecar与配置字节均相同，未改变判别标准。

v2所有输入303–4,092 tokens，加64输出预算均低于部署32,768窗口；模型config上限131,072与部署窗口分开记录。运行使用BF16、TP1、max_running_requests=1、temperature 0、no-think。每条输出均12 tokens、finish `stop`，原文是空think后一个yes/no score。无silent truncation、分块、自动补JSON或预算升级。

独立核验了全部29条：

- audit的visible manifest对应完整原始body，实际 `documents` 的reference ID和正文完全相符；完整历史角色/文字、当前query、待评assistant答案保持原样。三个真实例各有五个来源、9,615个原文字符和此前四轮完整QA。
- 最后user消息只有冻结guardian协议；历史assistant留在conversation，未变成证据。标签/期望、arm名称与reference答案不进入消息或documents。
- 官方tokenizer重渲染IDs、ledger IDs、原生 `/generate` 请求IDs及服务端prompt回显逐token相同。请求没有text、messages或session_id，未发生二次chat模板；output IDs、raw response与completion计数相符。
- 三个真实例仍标 `unvalidated_hardware_run`；26个合成例仍标 `simulation`。诊断保持原query/input绑定，不把真实GPU评审执行变成生成器性能认证，也不允许该来源进入正式quality gate。

串行HTTP时间总计2.596秒，均值89.50ms、中位83.61ms、范围80.95–204.13ms，仅是本次原生诊断调用的描述。没有逐例冷reset，累计8,960 tokens命中prefix；后两个真实例各复用3,648 tokens。运行时Qwen大池已加载但idle、Granite独立服务在GPU0运行；操作方核查过进程状态，但未保存完整资源时间线。本次定位为诊断验证，不是性能benchmark，不推断共驻产生了多少干扰，也不与Atomic多调用或Lettuce直接本地分类器作因果速度排名，也不包括模型加载和端到端文件处理开销。

## 22个事实控制：结果与分母

固定sidecar定义了12个entailed和10个contradicted/insufficient控制。这里只将后两类合并为风险yes、entailed为no；不是验证三分类能力。`p001–p020、p022、p023`共22条均吻合。正反例覆盖对象包含范围、逐项/合计、明确单位换算、版本与角色、否定、脚注例外、联合来源和历史assistant污染。期望标签在此次运行前已构造并固定，但都是assistant构造的开发世界，不是独立人工meta-test。

这个结果支持保留该基座作后续诊断；它不允许将22/22扩写成“所有29条正确”或总体精度100%。尤其合成 `p002` 原文明确写excluding，真实长例只有分列的限额表，难度不同；模型识别简短显式否定，不证明能在历史污染和长上下文中判断对象范围。

| 另行核对的维度 | 原生结果 | 可得结论 |
|---|---|---|
| `p018`真实端口事实引用到timeout来源 | no | 全visible含正确端口，所以其visible事实标签仍吻合；**实际引用子集不支持**，该协议没有验证引用绑定。不能把此例从22条中删掉，也不能把no当正确引用。 |
| `p021`来源未给保留天数，答案恰当说明不足 | yes | 对证据有限的恰当拒答发生风险误报。没有将它纳入事实分母来掩盖问题。 |
| `p024`来源已给14天，答案却说未给 | yes | 与此处错误拒答相容；只验证这一例，不能抵消p021。 |
| `p025/p026`合法JSON与尾逗号非法JSON | 均no | groundedness不是语法验证。代码有效性仍由静态检查和相应接口验证单列。 |

两种拒答没有 `visible_verdict` 标签，代码没有prose事实标签，三条real没有期望score，因此这七条均保留但不进入22条事实吻合率。no-think没有解释；不能猜测某次成功或失败是模型采用了哪条推理路径。

## 三个真实回答：逐源审查

三者来自同一个 `old_error_history` 冻结fixture的off、all_rows、focus_refs答案，query都是“Not documents, I was asking for attachments?”，可见原文和此前八条消息相同。两个重复限额段落 `E_53079fa73a` 与 `E_fe06da00dd` 分列Individual Document Size 1MB、Single Attachment Size 10MB、Request Body Size 11MB；另一段 `E_bd58806373` 明确提醒附件会影响性能。还包含一段Discovery而非Cloudant的材料。judge收到这些完整原文，没有把原生成器的辅助table annotation作为新来源加入，因此不是评估三种辅助表示本身。

| 当前待评答案 | Granite | 原文诊断 |
|---|---|---|
| `real-001` / off | no | 单附件10MB有据，但把“individual document 1MB”扩成“document **including attachments** ≤1MB”，多处重复，来源表没有提供该包含关系。当前对象范围缺陷漏检。 |
| `real-002` / all_rows | no | 10/1/11三个数值与表列相符，但末句称这些限额保证附件“do not impact overall performance”；原文只有小附件建议且明确可能影响性能，不能据限额推导该保证。未检测出这一额外断言。 |
| `real-003` / focus_refs | no | 核心单附件10MB及所列E530/Efe来源直接相符。该结果与此次局部原文审查相容，不是人类整体评分或非劣证明。 |

三条**当前答案均没有50MB**。50MB来自此前assistant把Discovery上传限制错当Cloudant的历史回答；旧历史还含SQL Query函数与附件JSON示例污染。原生协议目标是最后assistant，故不能将本次no描述成“已经检查并认可历史50MB”，也不能用历史断言替当前real-001的包含关系提供支持。后续若评整个历史的真实性，需要另立目标协议。

## 与Lettuce、Atomic的可比边界

Granite v2与Lettuce v1、Atomic v2/v3的audit和labels字节完全相同：audit SHA `2138d69c41e502ed24e2c332236ed64e4fefee94758731104ee3eb6ac6a10e8b`，labels SHA `f7209c6a020a5be4b4e42449fb05f29fa5ce4d4a5860beeeb9d62f2651ce7781`。这证明评的是同一批保存答案，不证明模型实际输入和输出定义相同。

Granite使用完整有角色的历史、documents及一个整答风险标签；Lettuce原生question/context/answer模板不输入历史，返回answer token的unsupported span；Atomic分解并分别检查visible/cited来源、忠实抽取和覆盖，guided v3执行67次调用。不能把Granite的22条二分类吻合与Atomic的24条prose完整合同matched数直接排模型名次，也不能把p015上的差异归因为单一模型能力。具体诊断见[Atomic结果](atomic_judge_validation.md)与[Lettuce入口和报告](lettuce_judge.md)。

## 值得验证的BYOC扩展

官方4.1支持Bring Your Own Criteria，保持原生yes/no与固定模板即可指定独立标准；其论文/模型卡上的总体成绩不能代替本项目验证。[官方固定BYOC说明](https://huggingface.co/ibm-granite/granite-guardian-4.1-8b/blob/ab01ccca5dcfb80246369a086a4a87a29198f5af/README.md)。以下是当时提出的两项候选。后续已实现为固定profile并完成旧29与新16的真实三臂验证，结果与回归见[BYOC验证](granite_byoc_validation.md)；原始ungroundedness仍保留为默认基线：

1. 对象—量值—条件绑定风险：只评当前答案，将对象、单位、逐项/合计、包含关系、版本/角色/套餐、否定和例外作为不可随意删改的断言组成；要求documents对完整关系提供支持，分列值不能自动推出包含、相加或性能保证。历史只能解析指代，不能提供事实。关于“来源未提供”的拒答按来源可答性核查，不把缺少该数值本身当成拒答失实。
2. 实际引用绑定风险：单独判断答案标出的来源ID能否支持相应断言。全visible其他地方有支持仍不能替错误引用补证；没有显式引用时标为未覆盖，不能自动推成引用正确或不正确。此项与事实grounding分别报告，避免一个yes/no混合两种错误后失去解释性。

这两项都不得写入产品名、已知正确数字、task ID、sidecar标签或期待答案；没有概率阈值可在这29条上调优。实施前固定标准、极性和输出预算，用原生标准/新标准同29条作开发回归，完整保留改善、误伤与失败；另预先构造未用于措辞选择的对象/单位/父条件/历史干扰变体，随后才用独立固定meta-test判断泛化。可比较no-think与think，但需新协议和成本记录，不作为失败后自动重试。评审改进只是测量能力：A的来源条件表达、API证据补齐与C的写入准入仍按[三个机制优先级](ac_extension_priorities.md)独立推进。
