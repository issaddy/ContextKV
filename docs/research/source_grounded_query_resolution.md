# 多轮意图与来源主题解析

## 当前多轮入口

`query_mode: conversational` 通过 [`adapters/conversation.py`](../../adapters/conversation.py) 在已有检索 worker 内组织当前任务。主线示例 [`configs/retriever_conversational.example.yaml`](../../configs/retriever_conversational.example.yaml) 显式配置 `selector`，按需使用同一个 Qwen3 服务进行语义解析。省略它时保留字面任务承接与历史检索回退，不加载客户端。两者复用同一条 BM25/BGE-M3/RRF/BGE-reranker 链。默认 single 模式只做一次索引召回；明确具名条件补充可在同一候选池追加一次条件评分，规则如下。

解析 HTTP 等待不持有 encoder/reranker 的锁；独立会话可以先检索，模型 forward 仍串行。共享 tokenizer 计数和调用计数仅持短锁，复用 httpx 的线程安全客户端。关闭前由现有 retrieval worker 排空机制等待所有请求，避免取消后继续访问已释放模型。

第一轮直接检索；明确的字面任务补充仍可直接组合。例如上一轮尚未指定具名对象、本轮仅为`Use AtlasCloud`，保留两个原始子句。其他轮次通过同一次Qwen3调用更新任务状态。名称、代词或问句形式都不能单独证明问题独立；特别是具名方案追问也可能继承已有产品和条件。无客户端时保留旧具名问题/近期历史处理，不额外加载模型。近期已提交状态可以承接更早条件，明确回访可读取全部用户历史；原QA不被改写。

无客户端、失败或未决解析使用最近三条用户问题的既有投影。窗口内有已提交的完整解释时，从最近一次解释接上后续原始问题，避免重新引入它已替换的旧条件；明确新话题的解释同时形成边界。完整生成历史始终保留，不因查询窗口而删除。

历史名称须有原文来源，不复制 assistant 的数字、步骤或推荐。当前路径直接产生名称与历史位置：找到一个合格来源即可，不再先构造全部来源窗口、字符坐标和长 ID 再丢弃。保留的枚举主题选择接口仍按需生成完整来源诊断。

实现借鉴 [CHIQ](https://aclanthology.org/2024.emnlp-main.135/) 的先解歧再改写思路与 [HAConvDR](https://aclanthology.org/2024.findings-acl.792/) 的历史去噪动机，但采用单次无训练推理，不声称复现论文完整流程。旧 `current/last3` 等低成本模式仍可直接使用。辅助模型的调用量、输入/输出 tokens、耗时和失败均进入 retrieval manifest 与完整响应计时，不能只报告主生成耗时。

字面组合沿用会话查询扩展的基本思路：从历史补齐当前查询缺少的任务信息，而不要求每次都生成一个新句子。[QuReTeC](https://arxiv.org/abs/2005.11723) 以历史词项选择实现查询补全；这里采用明确片段的工程规则，没有引入其分类器或训练流程。是否有效以检索和最终任务结果评价，不以改写句式评分。

### 当前任务的选择性更新

自由改写可能保留主题却换掉信息需求，也可能在新问题中省掉已经明确的对象。当前以`TaskFrame(topic, target, task, conditions)`分开记录主题、具体对象、请求的操作/信息及条件。字段值是开放文本，不建立产品或行业字典；它们描述用户请求，不是事实证据。

单次解析首先输出`task.operation=keep/replace`及任务值，再给出关系、历史轮次、回访位置与上下文更新。输入把当前问题放在首位，并把历史状态明确命名为`previous_turn`。固定XGrammar按schema声明顺序约束输出，任务判断先于历史选择和状态补丁，不增加第二次生成。新信息需求可替换task并保留topic；属性纠正可以保留原task。

`updates`单独修改topic、target和有名称的条件。条件不再是一段混合字符串，而是`version`、`platform`、`region`等开放名称和值；只提交变化的键，null删除该键，`conditions=null`清空条件。宿主保持每个未出现在补丁中的键不变，因此改变联网方式不需要模型重写并记住版本与平台。该局部更新约定参考[JSON Merge Patch](https://www.rfc-editor.org/info/rfc7396/)的保留/替换/删除语义，固定业务字段清空为缺省值，不提供通用HTTP PATCH接口。

普通追问基于最近已提交状态，独立话题先清空；回访从指定历史状态继续。没有frame的旧记录在同一次解析中初始化；旧复合条件原样读为`legacy`项，由正常解析显式拆成命名条件，不用字符串规则猜测其含义。条件内部保存为排序后的不可变键值对；旧空条件/复合条件的显式请求身份保持兼容。歧义不提交一个猜测的frame。

宿主应用更新并按主题、对象、任务、条件拼接一个检索视图，取消第二次自由改写机会，继续使用一次召回和既有重排。纠正附件类型时，未修改的大小限制任务保留；新的信息需求可以更新task/target而沿用topic。生成使用这一共同查询，frame本身不在提示中重复展开；它随QueryContext通过已有attempt/提交/恢复保存，未新增数据库或后台记忆服务。无frame的旧请求身份保持兼容。

这是对[SOM-DST](https://aclanthology.org/2020.acl-main.53/)选择性覆盖状态机制的适配：保留、更新和删除操作将变化部分与持续状态分开。已检查[作者实现](https://github.com/clovaai/som-dst/blob/master/utils/data_utils.py)的操作应用方式；这里没有移植其领域ontology、训练模型或依赖。状态继承能防止被省略字段自动消失，但不能保证模型选对操作或字段含义，仍需检验最终检索与回答。

近期[ICR](https://arxiv.org/abs/2509.05100)以澄清/改写轨迹、训练和结果融合处理多重歧义；[2026年多步语料反馈方法](https://arxiv.org/abs/2610.10955)学习改写、检索与停止策略。它们支持将语义操作显式化的方向，但包含训练或额外检索流程。本阶段先复用现有模型的一次状态更新，不加入这些完整链路或把论文收益视为本系统收益。

### 保留主任务资料的条件排序

对于已确认的字面任务承接，完整查询保留原任务和当前条件，当前用户原句另作条件视图。已有BGE先按完整查询评分原候选池，再在同一池上评分条件视图；不重新搜索索引，不生成新的查询或答案。

复用[原始RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)及现有k=60。选择五段时，在“包含完整任务排名第一的段落”的约束下最大化RRF分数和：先保留这一段，再取其余融合分数最高的四段，最终保留真实融合分数与顺序。该约束防止条件视图挤掉主任务资料入口；不会把排名第一等同于语义正确或全部事实已覆盖。

普通轮次、相同查询视图、候选数量不超过返回预算时沿用原排序，没有额外评分。启用时只增加至多50个query/passage评分对，仍复用已加载的BGE。`condition_evidence`记录两份排名、保留段落、最终来源、额外对数和耗时，整体耗时进入P3检索计时。快路径last3保持原行为，不新增配置开关。

这是基于现有融合器的受约束选择扩展，没有复现论文之外的收益保证。初始证据来自保存的真实候选回放：保留了Cloudant的schema说明，同时移除了另一产品的建表资料；实际回答验证和适用边界统一见数据与评价。

### A/B/C 的共同合同

- `QueryContext` 是派生意图：原用户问题和实际回答保持原样。独立检索问题同时用于召回、主任务评分、证据解析和来源定位，生成提示保留原问题并提示当前条件优先。
- 来源定位在所有轮次保留当前检索排名，不再以是否已有解析标签或ContextPilot布局作为条件；正文仍可按缓存布局排列。低名次资料不因证据 ID 的字典顺序抢占定位表首位。
- `context.layout: contextpilot` 下，即使未调用解析模型，也保留当前检索优先级。优先级注释只渲染一次，且仅包含当前可引用来源；被显式版本请求替换的检索版本不再出现在优先级列表中。
- A/R 使用同一当前证据集合、意图和引用范围。意图标签或证据零交集本身不强制重建；实际历史前缀仍可能复用。正确版本、正文可见性和预算决定资格，B 按原生报价与实际成本比较；旧材料仅可留作历史，不获得当前引用资格。
- 明确版本请求替换同一段落的其他检索版本；`v1` 不匹配 `v1.1` 或 `v1-rc2`。显式比较仍允许多版本并存。
- 回访可以指向最近三轮之外的历史。模型选中多轮时，证据恢复对整个先行词集合做判定，不通过“最近一轮”偷偷选择不同对象；不能唯一确定的归档证据不自动恢复，仍保留当前真实检索结果。
- 解析结果随 PLANNED attempt 固定，回答成功后与原 QA、epoch/outbox 原子提交。无辅助模型、失败/未完成或歧义解析共用上述历史检索投影；已识别的歧义另以原始当前问题和`relation=ambiguous`传入生成并持久保存，不丢弃这个质量信号，也不把展开后的检索文本伪装成确定解释。未完成回答不进入下一轮历史；恢复或幂等重放不重复解析。

### 来源上下文与未决指代

多资料检索的词项相关并不说明资料适用于同一对象。在线输入、索引构建和CorpusIndex统一读取公开的source_uri/source/url及标题。HTTP、控制器CLI与原生对照共享同一元数据索引，在原候选池重排前补全地址和标题，现有BGE/Qwen3重排器以“来源、标题、完整正文”评分；不追加检索或评分调用，不重新编码索引，也不改变正文。元数据无法唯一对应的未选候选保持原样；生成使用的已选证据仍按精确正文与版本解析，显式依赖在此时补齐。

来源定位表在当前问题附近显示地址、标题和章节，供模型结合完整正文判断适用性；没有从URL推断硬性产品标签或过滤证据。当前条件优先、必要澄清、操作步骤保持适用系统一致等通用规则集中在共享系统提示中，不再在每轮意图后重复。未决轮次仍保留具体歧义提示。系统提示改变时，现有A规则重建执行版本，C在新回答提交后退役旧引用。

这一步采用[Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)保留片段文档背景的思路，先利用已有原始元数据，没有实施其LLM生成的逐片段上下文。[Late Chunking](https://arxiv.org/html/2409.04701v3)在整段token编码后按片段mean pooling，当前BGE dense使用CLS，不能直接换池化后仍视为相同模型。[ContextualRetriever](https://aclanthology.org/2025.emnlp-main.602/)还涉及专门的上下文监督；这些路线保留为有依据的后续选择，本次不引入训练。

[AsK](https://aclanthology.org/2025.acl-industry.63/)研究歧义下的澄清选择。当前先复用已有解析器输出的未决关系，提示说明缺失区分、提供有据共同信息或明确标注的备选方案；不增加歧义分类器或第二轮澄清生成。该状态进入共同候选与提交链，是语义提示而非质量证明，也不强制改变缓存动作。

元数据投影没有扩大历史窗口或添加模型调用；隐式追问解释会增加条件调用，其开销纳入完整运行。尚未产生完整解释的长指代链仍可能缺失较早对象，单次解释也可能发生偏差；应检查实际任务与证据，不能用历史窗口或格式合法性代替语义评估。

结构正确不等于指代正确。定点检查分别保存独立问题的对象/条件覆盖和历史轮次最小性；模型仍可能多选先行词，不能把合法 JSON 当作语义验收。真实多轮结果及适用边界统一记录在[数据与评价](../data_and_evaluation.md)，不另立实验纪要。

## 保留的枚举主题选择入口

下面是仍有复用价值的 `source_topic` / `assistant_topic_unverified` 接口说明。它们不在本轮主线扩大实验或训练；`conversational` 复用其来源资格和 transport，使用上面的独立问题协议。

## 已实现的配置、状态与外部调用边界

`adapters/query_resolution.py`提供`PriorContext/PriorSource`、`snapshot_context`、controller/native provider和`TopicResolver`。`adapters/query_selector.py`提供固定配置的同步`OpenAICompletionClient`；它在已有retriever worker内调用，不依赖bench，避免循环import。真实controller CLI与Native CLI均已接入`run_sessions(...retrieval_context_provider=...)`。旧mode不构造selector、不读取该provider；原query/history不被改写。

新模式为`source_topic`和`assistant_topic_unverified`。两者使用相同SYSTEM/schema、同一模型选择器、同一显式历史窗口；区别是后者允许没有来源支持的assistant实体候选，因此名称和manifest始终标注unverified。两者都要求`current_topic_rrf`，最终候选池与原文通过既有`search_current_topic`和总50重排，未另写融合器。

默认结构协议仍为`static_v1`。可显式设`selector_schema_mode: enum_v2`，把每次实际目录ID写入schema enum，空目录使用`maxItems: 0`；消息、候选提取、预算和历史窗口不改。v2协议身份与每次实际schema另存，来源资格和合法ID仍不等于语义正确。既有对照中 v2 减少了格式失败，语义选择错误仍存在，因此保持可选。

配置使用 `query_mode: source_topic`、`query_fusion: current_topic_rrf`、`selector_history_turns`，以及 `selector` 下显式的本地服务、模型、revision、context_window 和输入/输出预算。无默认生成模型；已退役的 Qwen2.5-7B 示例移除。该可选路径尚未在当前 Qwen3 交付配置启用，不把旧模型的格式结果追认为新模型效果。

`tokenizer_path`和`tokenizer_revision`可显式配置；否则只从固定revision的本地缓存读取，`local_files_only=True`。请求model字符串需匹配服务的served-name；配置revision是操作者声明，不能当远端权重验证。temperature固定0，不做自动重试。模型输入使用固定chat template、`return_dict=False`计实际IDs长度；输入超过配置上限或输入＋输出预算超过context window时不发送HTTP，不截断。HTTP响应原文、usage、finish、错误、输入计数和调用耗时保留；`length`、非法JSON、未知/错类ID不得生成新query，保留原query及失败原因。

模型只选择枚举entity/user-turn ID。宿主原样拼接当前问题及所选历史原span，不能接受模型自由生成的限额、建议或代码。候选按同一名字合并多次历史出现，完整history occurrences、EvidenceKey/版本、support/source-window字符偏移保留在trace。**模型只看到每个名字的一个明示支持摘录，不是完整历史证据**；同名出现在多源并不保证产品同一或指代正确。这些是待实测的语义能力，schema只解决外部格式/身份。

`selector_history_turns=3`是当前可配置成本设置，取最近三个完整QA供selector并形成候选；trace记录省略的更早消息数。生成器仍接原完整QA。来源资格可以追溯全部已提交历史，因此近轮出现的Cloudant名称仍能链接第一轮真正可见的来源。该窗口不是方法的理论限制；可配置更多轮次，并与ConvSearch-R1作者full-history设置分别报告成本/覆盖，不能隐去窗口差异。

### 在线与离线来源集合的具体差别

在线scope为`committed_required_keys`，来自各历史TurnRecord的required集合，**只是当时visible的保守子集**。离线`bench/source_topic_eval.py`从较早audit的`visible_evidence_manifest`重建来源，通常scope为`prior_audit_visible`。它不会声称两者对任意APPEND/optional material严格等价；真实差异的反例测试已覆盖。

若每个较早audit的visible原文/passage ID恰好等于当轮retrieved原文，则所有current retrieved必为required的已有合同能证明该形状等价，trace记`prior_audit_visible_equals_retrieved`。本次旧off的完整41轮全部满足该前提，每轮5/5；冻结driver显式核对它。缺失的较早正文不能用当前轮body补回当前query。两入口manifest/trace均记录scope，不改变TurnRecord语义、不新建状态。

离线入口支持默认plan：

```bash
python -m bench.source_topic_eval \
  --audit saved-dev-audit.jsonl \
  --split-manifest configs/mtrag_split_manifest.json \
  --retriever-config source-topic.yaml \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/source-topic-retrieval-dev
# 显式加 --execute 才加载模型并发送selector/检索请求。
```

只允许dev来源，所有attempt完成后才打开qrels；实际检索失败保留failed与空ranking的全预期分母，同时单列成功子集。selector失败使用原query仍可能成功检索，与检索失败分开。当前答案/官方rewrite/labels不进入resolver。纯`prepare_audit_turns`可供selector-only冻结准备复用。
