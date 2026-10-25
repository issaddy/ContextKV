# 多轮意图与来源主题解析

## 当前多轮入口

`query_mode: conversational` 通过 [`adapters/conversation.py`](../../adapters/conversation.py) 在已有检索 worker 内解析当前意图。配置见 [`configs/retriever_conversational.example.yaml`](../../configs/retriever_conversational.example.yaml)。复用同一个 Qwen3 服务、completion 客户端、BM25/BGE-M3/RRF/BGE-reranker；不训练模型，不增加检索服务或第二次重排。

解析 HTTP 等待不持有 encoder/reranker 的锁；独立会话可以先检索，模型 forward 仍串行。共享 tokenizer 计数和调用计数仅持短锁，复用 httpx 的线程安全客户端。关闭前由现有 retrieval worker 排空机制等待所有请求，避免取消后继续访问已释放模型。

第一轮直接检索；无指代/纠正线索的完整具名问题或定义问题可跳过解析。普通无依赖线索的轮次沿用已有 last3 检索。若上一轮是尚未指定具名对象的操作任务、本轮只是 `Use AtlasCloud` 等具名限定，则直接保留两个用户子句，例如 `how to create schema or tables?\nuse IBM Cloudant Query`，不让一次自由改写丢掉原任务。已有具名对象的替换、完整新指令及其他指代/纠正仍走单次语义解析。近期已提交解释可以承接更早条件，明确回访可读取全部用户历史；原 QA 不被改写。

历史名称须有原文来源，不复制 assistant 的数字、步骤或推荐。当前路径直接产生名称与历史位置：找到一个合格来源即可，不再先构造全部来源窗口、字符坐标和长 ID 再丢弃。保留的枚举主题选择接口仍按需生成完整来源诊断。

实现借鉴 [CHIQ](https://aclanthology.org/2024.emnlp-main.135/) 的先解歧再改写思路与 [HAConvDR](https://aclanthology.org/2024.findings-acl.792/) 的历史去噪动机，但采用单次无训练推理，不声称复现论文完整流程。旧 `current/last3` 等低成本模式仍可直接使用。辅助模型的调用量、输入/输出 tokens、耗时和失败均进入 retrieval manifest 与完整响应计时，不能只报告主生成耗时。

字面组合沿用会话查询扩展的基本思路：从历史补齐当前查询缺少的任务信息，而不要求每次都生成一个新句子。[QuReTeC](https://arxiv.org/abs/2005.11723) 以历史词项选择实现查询补全；这里采用明确片段的工程规则，没有引入其分类器或训练流程。是否有效以检索和最终任务结果评价，不以改写句式评分。

### A/B/C 的共同合同

- `QueryContext` 是派生意图：原用户问题和实际回答保持原样。独立检索问题同时用于召回、重排、证据解析和来源定位，生成提示保留原问题并提示当前条件优先。
- 来源定位保留当前 reranker 排名，复用 ContextPilot 的检索优先级注释；正文仍可按缓存布局排列。低名次资料不因证据 ID 的字典顺序抢占定位表首位。
- A/R 使用同一当前证据集合、意图和引用范围。意图标签或证据零交集本身不强制重建；实际历史前缀仍可能复用。正确版本、正文可见性和预算决定资格，B 按原生报价与实际成本比较；旧材料仅可留作历史，不获得当前引用资格。
- 明确版本请求替换同一段落的其他检索版本；`v1` 不匹配 `v1.1` 或 `v1-rc2`。显式比较仍允许多版本并存。
- 回访可以指向最近三轮之外的历史。模型选中多轮时，证据恢复对整个先行词集合做判定，不通过“最近一轮”偷偷选择不同对象；不能唯一确定的归档证据不自动恢复，仍保留当前真实检索结果。
- 解析结果随 PLANNED attempt 固定，回答成功后与原 QA、epoch/outbox 原子提交。失败/未完成的解析沿用原 last3 检索，不丢掉已发生上下文；未完成回答不进入下一轮历史；恢复或幂等重放不重复解析。

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
