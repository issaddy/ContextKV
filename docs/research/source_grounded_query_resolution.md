# 多轮意图与来源主题解析

## 当前多轮入口

`query_mode: conversational`通过[`adapters/conversation.py`](../../adapters/conversation.py)在已有检索worker内组织当前任务。主线示例[`configs/retriever_conversational.example.yaml`](../../configs/retriever_conversational.example.yaml)显式配置独立32B `selector`，8B继续生成回答，部署与完整资源代价见[部署说明](../deployment.md#多轮检索服务)。省略selector时保留字面任务承接与历史检索回退。两者复用同一条BM25/BGE-M3/RRF/BGE-reranker链；默认single只做一次索引召回，明确具名条件补充可在同一候选池追加一次条件评分。

解析 HTTP 等待不持有 encoder/reranker 的锁；独立会话可以先检索，模型 forward 仍串行。共享 tokenizer 计数和调用计数仅持短锁，复用 httpx 的线程安全客户端。关闭前由现有 retrieval worker 排空机制等待所有请求，避免取消后继续访问已释放模型。

配置解释器时，第一轮建立完整任务状态，历史引用为空。明确的字面任务补充可直接组合，例如尚未指定对象时的`Use AtlasCloud`；其他轮次用一次Qwen3调用更新状态，结合实际QA判断任务承接或切换。近期状态承接更早条件，明确回访可读取全部用户历史，原QA保持原样。无客户端时采用首轮直接检索和后续历史投影。

无客户端、失败或未决解析使用最近三条用户问题的既有投影。窗口内有已提交的完整解释时，从最近一次解释接上后续原始问题，避免重新引入它已替换的旧条件；明确新话题的解释同时形成边界。完整生成历史始终保留，不因查询窗口而删除。

默认会话解释使用实际QA和已提交状态恢复指代，生成事实由本轮证据支持。下文的来源资格枚举接口按需读取原文，检查候选名称的来源。

实现借鉴 [CHIQ](https://aclanthology.org/2024.emnlp-main.135/) 的先解歧再改写思路与 [HAConvDR](https://aclanthology.org/2024.findings-acl.792/) 的历史去噪动机，采用已有模型的单次推理。`current/last3` 等低成本模式也可直接使用。辅助模型的调用量、输入/输出 tokens、耗时和失败均进入 retrieval manifest 与完整响应计时。

字面组合从历史补齐当前查询缺少的任务信息，参考[QuReTeC](https://arxiv.org/abs/2005.11723)的历史词项选择思路，采用明确片段的工程规则。效果以检索和最终任务结果评价。

### 原问题与派生解释共同检索

`query_fusion: dialogue_rrf`复用当前会话解释器，同时让最近三条原始用户问题参与召回与排序。实际QA、已提交任务状态和生成提示不改写，不增加LLM调用。原始问题视图占一份总权重，其余不同的完整解释/子需求视图共同占一份；更多子需求不会自动产生更多总票数。各视图的BM25/dense列表进入同一候选池，再分别对同一池做交叉编码评分并用同样权重融合排名，最后沿用五个不同内容组的选择及原文、依赖、A/B/C链路。

该机制借鉴[MVR](https://aclanthology.org/2020.findings-emnlp.354/)分别重排再融合的思路，使用已有BGE、原问题视图和RRF。近期[Faithful Query Resolution](https://arxiv.org/abs/2610.15034)也区分请求保真与检索扩展；本实现按视图家族归一权重。查询编码和同池评分均可批处理，普通双查询入口也复用一次编码；批次耗时记录为共享成本。

当前默认仍为`single`。固定输入检索及一段完整公开会话的对照已完成：部分资料和回答适用性改善，也存在来源丢失、错领域材料及未支持的规则。融合机制保留为可用选项；结果与成本见[评价](../data_and_evaluation.md#原问题与派生解释的检索对照)。

### 当前任务的选择性更新

自由改写可能保留主题却换掉信息需求，也可能省掉复合请求的一部分。当前`TaskFrame`分别记录topic、target、task、conditions和requirements：最后一项保存复合请求中分别需要回答的内容，条件单独保存，避免改地区或平台时丢掉仍需回答的部分。字段为描述用户请求的开放文本。

已有完整状态时，单次解析首先输出`task.operation=keep/replace`、任务值和requirements，再给出关系与字段更新。keep由宿主保留整个原任务及需求列表；replace同时替换需求列表，单一需求可用空列表而沿用task。无状态或前轮未决时，同一次调用完整初始化。首轮即保存需求和条件，后续“改成Windows”等补丁更新已有状态。XGrammar约束输出结构，需求分解在同一次调用完成。示例输出上限384 tokens容纳完整状态，记录实际用量。

`updates`单独修改topic、target和有名称的条件。条件不再是一段混合字符串，而是`version`、`platform`、`region`等开放名称和值；只提交变化的键，null删除该键，`conditions=null`清空条件。宿主保持每个未出现在补丁中的键不变，因此改变联网方式不需要模型重写并记住版本与平台。该局部更新约定参考[JSON Merge Patch](https://www.rfc-editor.org/info/rfc7396/)的保留/替换/删除语义，固定业务字段清空为缺省值，不提供通用HTTP PATCH接口。

普通追问基于最近已提交状态，独立话题先清空；回访从指定历史状态继续。没有frame的旧记录在同一次解析中初始化；旧复合条件原样读为`legacy`项，由正常解析显式拆成命名条件，不用字符串规则猜测其含义。条件内部保存为排序后的不可变键值对；旧空条件/复合条件的显式请求身份保持兼容。歧义不提交一个猜测的frame。

宿主应用更新后，按主题、对象、任务、条件拼接检索视图，交给现有召回和重排。纠正附件类型时，未修改的大小限制任务保留；新的信息需求可以更新task/target而沿用topic。生成使用同一查询，frame随QueryContext通过attempt提交和恢复；旧请求身份保持兼容。

这是对[SOM-DST](https://aclanthology.org/2020.acl-main.53/)选择性覆盖状态机制的适配：参考[作者实现](https://github.com/clovaai/som-dst/blob/master/utils/data_utils.py)，通过保留、更新和删除操作将变化部分与持续状态分开。宿主保留未修改字段，模型负责选择操作和解释字段含义，效果按最终检索与回答评价。

近期[ICR](https://arxiv.org/abs/2509.05100)以澄清/改写轨迹、训练和结果融合处理多重歧义；[2026年多步语料反馈方法](https://arxiv.org/abs/2610.10955)学习改写、检索与停止策略。它们为显式语义操作提供研究参考。当前实现采用已有模型的一次状态更新。

### 条件形成的来源与适用范围

每轮显式当前请求、只对背景做增量更新的原型已完成[开发比较](../data_and_evaluation.md#当前请求与背景更新的分离原型未采用)，没有替换v16。它修复部分旧任务沿用，却会将助手给出的规则当成用户属性，或把“能否声明某条件”的问题变成该条件已成立的记录。旧格式仅保存名称和值，字段变化位置也不能证明值来自用户断言。

条件来源原型已实现原话定位、范围标记、短回复前问及持久/生成传递，但[83次开发调用](../data_and_evaluation.md#条件来源与范围原型未采用)仍出现语义误判与逐字校验误拒，未并入主线。来源匹配只能检查文本依据，不能证明条件成立。假设条件仍须服务于当前场景问答，例如“如果换平台”，不能因未成为用户事实而全部丢弃。后续先检查现有非思考辅助模式的语义能力和成本，不继续在同一队列上叠加格式约束。

研究参考包括[AgentMemGate](https://arxiv.org/abs/2610.07707)在写入时区分推测与已发生事件，以及[Gated Memory](https://arxiv.org/abs/2610.11270)的当前表达/参考上下文、来源和适用范围。它们研究长期记忆且包含额外判定阶段；ContextKV需适配动态任务条件并计入完整调用成本，不直接照搬固定字段、延迟读出或多阶段调用。[Reality Monitoring](https://arxiv.org/abs/2607.23927)也提示需要检验对话内容的发言来源。来源原型仅保留为实验记录。

### 因果历史与输入预算

解释器读取最新状态与近期完整实际QA，并从已提交状态变化中定位字段最近更新或清除的轮次。在既有预算内补入当时用户原话及其前一条助手消息；普通更新只提供一份最新状态，初始化和明确回访保留所需旧状态视图。历史与来源候选共用一次分词批次，按完整项加入。变化索引临时计算，最终生成仍使用完整原始QA。这一历史定位机制参考Context-aware DST。输入可用性检查已通过；56轮固定输入的[语义比较](../data_and_evaluation.md#因果历史的固定输入语义检查)仍发现keep误判与否定丢失，尚未建立语义优胜。

### 复合任务的证据覆盖

复用[Question Decomposition for RAG](https://arxiv.org/abs/2507.00355)的按子需求召回、合并候选再重排流程，以及[xQuAD](https://terrierteam.dcs.gla.ac.uk/publications/ecir2010_rodrygo_div.pdf)显式考虑需求方面的思想。一次状态解释同时保留子需求，宿主为每项附上当前topic/target/conditions；平台或版本变化作用于全部子查询。首轮以用户原句作为整体问题，后续整体问题由已提交状态构成。

至少两个独立需求时，整体问题和子查询批量送入已有BGE编码器，分别查询固定BM25/dense索引，用等权RRF形成至多50段种子池。有元数据且属于同一检索快照的相邻原文加入候选，按整体任务评分后折叠重复内容，保留至多50个组供各子需求评分，最终选出五个内容组。单一需求沿用普通流程；各阶段工作量进入运行记录。

查询渲染保留需求原句，对主体和对象按完整值消除重复，如v4.1与v4分别处理。命名条件始终保留`名称: 值`绑定，只消除完整相同的行：某值已在需求或其他条件中出现，并不能说明该属性已表达。整体查询与每个子需求共用这一投影；派生查询继续进入生成提示及A/R候选。v16修复了仅按条件值去重导致的属性丢失，原有未命名`legacy`条件仍按原文本处理。输入成本和验证范围见[评价记录](../data_and_evaluation.md#命名条件绑定的投影检查)。

最终预算仍为至多五个内容组。固定整体任务排名第一的来源，再贪心最大化

`F(S) = Σ_j max_{d∈S} 1/rank_j(d)`，其中j包含整体任务和各项需求。

每项需求只计算已选集合中的最佳排名，重复覆盖的边际增益逐渐下降；同增益按整体相关性排序，返回原始分数和正文。对已固定的内容组代表及任务锚点，这是非负、单调的次模覆盖目标，可使用[经典基数约束贪心算法](https://pubsonline.informs.org/doi/10.1287/moor.3.3.177)。该目标衡量排名覆盖，真实任务覆盖与适用条件按原文评价。

`requirement_evidence`保留实际子查询、每路召回、候选及各需求排名、选择增益和模型工作量，保存排名即可重算选择。所选原文继续经过统一依赖、引用范围、A/R构造、真实token成本和提交/退役。旧frame默认requirements为空；空列表不改变旧显式请求身份，无数据库迁移。

### 一次调用的任务与来源联合决策

`evidence_selection: semantic`复用同一解释器，一次返回当前任务状态和至多五个来源ID。初始召回同时使用最近原始用户问题，以及“上一轮已提交任务＋当前原句”视图，经批量编码、BM25/dense/RRF合并候选后重排一次。`query_fusion: single`在此表示这套联合流程，其他融合选项属于排名路径。

联合模型读取真实QA、上一轮状态以及候选的完整正文、版本、父章节和必要依赖，在同一响应中决定任务更新与选材。用户身份、条件和需求来自实际对话；资料用于理解术语和选择适用分支。部分需求有据时选择其支持材料，存在歧义时可保留备选来源。原文在预算内按完整单元装入；合法空选择保持为空，未完成或无效输出使用已有排名。

该改造借鉴[同场景的对话条件重排](https://arxiv.org/html/2606.11945v1)使用实际历史的原则，采用既有BGE和Qwen3。排名路径保留复合需求逐项召回；联合路径在已有候选上同时判断需求与来源，效果见下述开发比较。

回访较早话题时，解释器同时读取选中旧轮次的用户问题和真实回答。完整原文与当前任务一起进入A/R规划、提交、恢复和退役；幂等重放复用已保存结果。

默认仍为`ranked`。v5/v6旧来源选择诊断按原版本保留在`runs/dialogue-retrieval/selection/`；当前v7已完成8个独立选定公开开发会话的配对比较：文档覆盖/排序改善，回答评分相近，辅助阅读成本使完整响应增加。联合路径作为可选能力保留，详细结果见[统一评价](../data_and_evaluation.md#一次联合任务与来源决策的开发比较)。

### 保留主任务资料的条件排序

对于已确认的字面任务承接，完整查询保留原任务和当前条件，当前用户原句另作条件视图。已有BGE先按完整查询评分原候选池，再在同一池上评分条件视图；不重新搜索索引，不生成新的查询或答案。

复用[原始RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)及现有k=60。选择五段时，在“包含完整任务排名第一的段落”的约束下最大化RRF分数和：先保留这一段，再取其余融合分数最高的四段，最终保留真实融合分数与顺序。这样保留主任务资料入口，同时引入当前条件的相关资料。

普通轮次、相同查询视图或候选数未超过返回预算时沿用原排序。条件评分复用已加载的BGE，增加至多50个query/passage评分对。`condition_evidence`记录两份排名、保留段落、最终来源、额外对数和耗时，计入P3检索成本。last3使用原查询流程。

初始候选回放保留了Cloudant的schema说明，同时移除了另一产品的建表资料；实际回答验证见[数据与评价](../data_and_evaluation.md)。

### A/B/C 的共同合同

- `QueryContext` 是派生意图：原用户问题和实际回答保持原样。独立检索问题同时用于召回、主任务评分、证据解析和来源定位，生成提示保留原问题并提示当前条件优先。
- 来源定位表按当前检索排名排列，正文按所选缓存布局排列。
- `context.layout: contextpilot` 下，即使未调用解析模型，也保留当前检索优先级。优先级注释只渲染一次，且仅包含当前可引用来源；被显式版本请求替换的检索版本不再出现在优先级列表中。
- A/R 使用同一当前证据集合、意图和引用范围。意图标签或证据零交集本身不强制重建；实际历史前缀仍可能复用。正确版本、正文可见性和预算决定资格，B 按原生报价与实际成本比较；旧材料仅可留作历史，不获得当前引用资格。
- 明确版本请求替换同一段落的其他检索版本；`v1` 不匹配 `v1.1` 或 `v1-rc2`。显式比较仍允许多版本并存。
- 回访可以指向最近三轮之外的历史。模型选中多轮时，证据恢复对整个先行词集合做判定；能够唯一确定的归档证据用于恢复，其余情况使用当前真实检索结果。
- 解析结果随 PLANNED attempt 固定，回答成功后与原 QA、epoch/outbox 原子提交。无辅助模型、失败/未完成或歧义解析共用上述历史检索投影；已识别的歧义以原始当前问题和`relation=ambiguous`传入生成并持久保存。未完成回答不进入下一轮历史；恢复或幂等重放不重复解析。

### 来源上下文与未决指代

多资料检索的词项相关并不说明资料适用于同一对象。在线输入、索引构建和CorpusIndex统一读取公开的source_uri/source/url及标题。HTTP、控制器CLI与原生对照共享同一元数据索引，在原候选池重排前补全地址和标题，现有BGE/Qwen3重排器以“来源、标题、完整正文”评分；不追加检索或评分调用，不重新编码索引，也不改变正文。元数据无法唯一对应的未选候选保持原样；生成使用的已选证据仍按精确正文与版本解析，显式依赖在此时补齐。

来源定位表在当前问题附近显示地址、标题和章节，供模型结合完整正文判断适用性。共享系统提示集中说明当前条件优先、必要澄清和操作步骤的系统一致性；未决轮次补充具体歧义提示。系统提示改变时，A重建执行版本，C在新回答提交后退役旧引用。

这一步采用[Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)保留片段文档背景的思路，使用已有原始元数据。后续可研究LLM生成片段上下文、[Late Chunking](https://arxiv.org/html/2409.04701v3)的整段编码后分片mean pooling，以及[ContextualRetriever](https://aclanthology.org/2025.emnlp-main.602/)的上下文监督。当前BGE dense使用CLS；采用其他池化方式时需要相应的编码适配与效果验证。

[AsK](https://aclanthology.org/2025.acl-industry.63/)研究歧义下的澄清选择。当前复用已有解析器输出的未决关系，提示说明缺失信息、提供有据共同信息或明确标注的备选方案。该状态进入共同候选与提交链，缓存动作由B按候选可行性和成本选择。

元数据投影复用已有历史与模型结果；隐式追问的解释调用计入完整运行成本。长指代链的对象遗漏和单次解释偏差，通过任务、证据及最终回答评价。

诊断记录所选先行词、对象/条件覆盖和历史轮次；真实多轮结果集中在[数据与评价](../data_and_evaluation.md)。

## 保留的枚举主题选择入口

`source_topic` / `assistant_topic_unverified` 提供枚举主题选择接口；`conversational` 复用其来源资格和 transport，使用上面的独立问题协议。

## 已实现的配置、状态与外部调用边界

`adapters/query_resolution.py`提供`PriorContext/PriorSource`、`snapshot_context`、controller/native provider和`TopicResolver`。`adapters/query_selector.py`提供固定配置的同步`OpenAICompletionClient`；它在已有retriever worker内调用，不依赖bench，避免循环import。真实controller CLI与Native CLI均已接入`run_sessions(...retrieval_context_provider=...)`。旧mode不构造selector、不读取该provider；原query/history不被改写。

新模式为`source_topic`和`assistant_topic_unverified`。两者使用相同SYSTEM/schema、同一模型选择器、同一显式历史窗口；区别是后者允许没有来源支持的assistant实体候选，因此名称和manifest始终标注unverified。两者都要求`current_topic_rrf`，最终候选池与原文通过既有`search_current_topic`和总50重排，未另写融合器。

默认结构协议仍为`static_v1`。可显式设`selector_schema_mode: enum_v2`，把每次实际目录ID写入schema enum，空目录使用`maxItems: 0`；消息、候选提取、预算和历史窗口不改。v2协议身份与每次实际schema另存，语义效果按实际选择的来源另行评价。既有对照中 v2 减少了格式失败，语义选择错误仍存在，因此保持可选。

配置使用 `query_mode: source_topic`、`query_fusion: current_topic_rrf`、`selector_history_turns`，以及 `selector` 下显式指定的本地服务、模型、revision、context_window 和输入/输出预算。该可选路径尚未在当前 Qwen3 交付配置启用。

`tokenizer_path`和`tokenizer_revision`可显式配置；否则只从固定revision的本地缓存读取，`local_files_only=True`。请求model字符串需匹配服务的served-name；配置revision是操作者声明，不能当远端权重验证。temperature固定0，不做自动重试。模型输入使用固定chat template、`return_dict=False`计实际IDs长度；输入超过配置上限或输入＋输出预算超过context window时不发送HTTP，不截断。HTTP响应原文、usage、finish、错误、输入计数和调用耗时保留；`length`、非法JSON、未知/错类ID不得生成新query，保留原query及失败原因。

模型选择枚举entity/user-turn ID，宿主原样拼接当前问题及所选历史原span。候选按同一名字合并多次历史出现，完整history occurrences、EvidenceKey/版本、support/source-window字符偏移保留在trace。模型输入为每个名字的一份明示支持摘录，schema约束候选身份；跨来源同名对象的指代效果按实际任务评价。

`selector_history_turns=3`取最近三个完整QA供selector并形成候选；trace记录省略的更早消息数。生成器仍接原完整QA。来源资格可以追溯全部已提交历史，因此近轮出现的Cloudant名称仍能链接第一轮真正可见的来源。窗口长度可配置；与ConvSearch-R1作者full-history设置比较时，分别报告历史范围、成本和覆盖。

### 在线与离线来源集合的具体差别

在线scope为`committed_required_keys`，来自各历史TurnRecord的required集合，是当时visible的子集。离线`bench/source_topic_eval.py`从较早audit的`visible_evidence_manifest`重建来源，通常scope为`prior_audit_visible`，可包含APPEND保留的其他可见材料。测试覆盖了两者的集合差异。

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
