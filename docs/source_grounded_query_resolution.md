# 来源支持的对话主题解析：可选工程链与开发准备

2026-10-24，基于 `e43112f` 源码定位后的最小可选工程链已实现：snapshot provider、两种ID selector、P3双路检索与离线入口、本地HTTP合同测试、全41轮冻结模型输入。**尚未执行真实selector或新增检索，不能声称话题选择或质量改善。** 不改当前 `user_topic` 默认，也不改C生命周期/host admission核心。

## 具体缺口与已有状态

`d5t4` 的用户历史没有Cloudant字样。先前assistant基于第一轮来源将“IBM文档数据库”具体化为Cloudant，后续用户以“it / image or pdf with a document / file size”继续。因此，放宽用户大写词regex仍无法找到Cloudant；禁止全部assistant语境也会丢失合理的指代线索。第一轮 `ibmcld_09715-2612-4022` 明确把Cloudant描述为JSON文档数据库，`ibmcld_00513-7-2197`也支持该产品主题。**这只能支持产品名称的来源，不能让历史assistant所说的错误限制数值、SQL函数或附件代码成为事实。**

现有源码已经保存必要信息，无需另建记忆库：

| 位置 | 目前行为 | 最小复用点 |
|---|---|---|
| `bench/run_sessions.py:131–139` | `OnlineSession.current()` → P3 `retriever.retrieve_turn(active)` → handler；检索先于controller/native handler | 在已有检索调用之前接可选的只读context provider，所有新增耗时仍落在原 `pipeline_arrival` 至完成之间 |
| `adapters/mtrag.py:52–72`、`114–136` | OnlineTurn已有真实P2/P3历史；P3初始contexts为空，官方证据不进入检索 | 保留OnlineTurn的当前query/history白名单，不把离线原始row、参考答案或当前官方contexts传给resolver |
| `controller/schema.py:70–80,127–145` | `TurnRecord.evidence_keys`、`raw_history`、`archive`、`reference_map`已存在 | 生成瞬时只读投影，沿用原EvidenceKey/版本/原文字段，无新hash、无新持久化状态 |
| `controller/session_store.py:255–271` | 提交时把candidate.required_blocks写入该轮evidence_keys；完整retrieved材料进入archive | 取**先前已提交轮evidence_keys的并集**并从archive解引用，作为有可见性证明的来源集合。required已在候选构造中验证可见 |
| `controller/cli.py:103–117` | session ID是`run_prefix:conversation_id`，当前handler在检索之后 | closure可按同一ID调用`store.get_snapshot()`；它是只读。不要用`read_snapshot()`充当只读接口，后者会创建session |
| `bench/native_baseline.py:305,322–364` | Native已有`snapshots`、同样的TurnRecord证据键和archive；execution被清空 | provider读取同一native snapshot，无需另建一份历史状态；不能假设native存在execution ledger |
| `adapters/hybrid_retrieval.py:337–367` | 目前只把user questions交给query view，再使用已有BM25/BGE/RRF/rerank | 新模式独立接收投影，最终仍返回只改contexts的OnlineTurn；原生成query、完整QA和正文不改写 |

两个容易误用的集合需要区分：**整个archive不等于历史真正可见**，其中可以有检索但未进入required的材料；**最新execution.visible_blocks也不等于所有历史可见**，REBUILD可能已移除更早的材料。共同最小合同用历史TurnRecord的required证据键。它也适用于没有execution的Native路径。旧记录若没有某项证明，只把那项标为来源不可用，不凭archive里有正文补造可见性。

已只读复核旧off的全部41轮：5个会话首轮没有先前来源，其余36轮有；严格使用较早轮次的可见audit累积后，最多37个不同来源，`d5t4`有15个。每轮actual_history均与此前成功轮的query/answer串联完全一致。此检查未读取test或qrels，未推断产品答案。

## 官方方法：复用什么，不能继承什么

| 方法与固定官方入口 | 可直接采用的机制/实现 | 本项目边界 |
|---|---|---|
| **Faithful Query Resolution，2026-10**：[v1 §4.2.1–4.3](https://arxiv.org/html/2610.15034v1) | resolver本来就接收history并允许消解“先前回答提到的实体”；把最小消解和扩展检索分成不同query。可继续复用项目已有raw/resolved两路RRF，不需另写融合器 | 原文没有“只许user文字”的限制。它没有为每个实体建立过去可见passage的证据绑定，也未独立标注resolver准确率；当前核查未找到论文关联的完整官方代码。只适配原则，不声称复现或移植其效果。论文CC BY4.0；此前详细检索/满意度混杂见[既有核查](query_fusion_research.md) |
| **CHIQ，EMNLP2024**：[论文](https://aclanthology.org/2024.emnlp-main.135/)、[作者固定源码 `4fa68a5`](https://github.com/fengranMark/CHIQ/tree/4fa68a52a984d1fcf9e2414cce497887f03dfdf3) | 先消解历史歧义再改写；`generation/enhancement_generation.py`与`adhoc_query_rewriting.py`把问题/回答上下文及多种派生文字分开。可借用“先解析语境，再组成检索query”的职责划分 | 原实现还用生成回答/解释扩写，并有2048-token截取；它们不能自动充当来源证明。固定仓库未见独立LICENSE，故当前不复制模板/代码或迁移其旧Torch/FAISS环境；不因此禁止重新实现公开的解析思路 |
| **ConvSearch-R1，EMNLP2025**：[论文](https://aclanthology.org/2025.emnlp-main.1349/)、[作者源码 `e7854c8`](https://github.com/BeastyZ/ConvSearch-R1/tree/e7854c89a79c904f331b306d36edde6e9ed12484) | 是可执行的学习式CQR对照：[`src/infer/infer.py:18–28,80–88,150–163`](https://github.com/BeastyZ/ConvSearch-R1/blob/e7854c89a79c904f331b306d36edde6e9ed12484/src/infer/infer.py)有历史Q/A序列化、固定prompt及`<rewrite>`解析，可按[Apache-2.0](https://github.com/BeastyZ/ConvSearch-R1/blob/e7854c89a79c904f331b306d36edde6e9ed12484/LICENSE)直接薄适配。作者发布[Qwen2.5-3B QReCC权重](https://huggingface.co/BeastyZ/Qwen2.5-3B-ConvSearch-R1-QReCC/tree/2756459f017585c71a6ca54316cb9acb43bdfee6)，本次核实revision、非gated、模型卡Apache-2.0 | 它允许assistant历史且有现成权重，无需本项目先训练。它优化检索奖励，不保证每个新增词都来自先前真实来源；自由rewrite应作独立baseline，不能直接标“已溯源”。首版可先用现有本地Qwen服务做同接口开发，专门3B权重比较另行调度；本次没有下载。无需复制作者verl/RL训练栈 |

FQR的“最小”描述的是改动用户意图的范围，不是要求只用一个regex。学习式CQR及语境解析值得作为实际强对照；来源限制是本项目要新增并测量的能力，不是对这些原方法已有保证的转述。

## 建议的首版：模型选择有来源的主题线索，宿主组成query

先采用一次短模型调用的选择式resolver，目标是产品/对象指代，而非再次生成答案。它与后续完整自由rewrite对照分开，避免首版同时引入训练、多步搜索和新状态机。

1. **从已有历史提候选。** prior user原话是意图线索；prior assistant可提供实体名称，但不能提供限制值等事实。可复用当前name-span函数作为**召回候选**的一种方式，包括single-title名字，并结合既有来源title/URI/正文出现的名称。不能直接把所有大写片段拼入query；真正选择由语境resolver完成。无需产品名白名单或Cloudant特例。
2. **先前来源绑定。** assistant实体候选只有在上述“历史已提交且可见”的来源中找到对应原文，才进入支持集合；保留原history span、EvidenceKey/版本、source span及出现轮次。来源引用ID缺失不意味着没有来源，允许在当轮已见材料里匹配。仅找到同一字符串也不证明指代正确，需把来源上下文交给selector并在评估中检查跨产品同名误选。
3. **模型只选线索。** 输入当前问题、真实历史与候选/来源短上下文，输出所选候选ID、对应用户对象span ID以及`resolved/unchanged/ambiguous`。候选ID是本次列表序号；不是新hash。模型不能输出任意答案段落作为检索扩写。数字产品版本/API名称可以保留，不能把历史“50MB上限”等数值claim伪装成产品实体。纯数值/单位不作为assistant主题候选。
4. **宿主组成检索视图。** 保留current query原文，追加被选中的产品/对象原span；历史assistant没有被选中的建议、错误数字、SQL函数和代码不会被整段拼接。`d5t4`的候选例是当前问题＋来源支持的`IBM Cloudant`＋上一用户的`image or pdf with a document`，不是含预期附件数值的改写。这只是待测查询，不是当前正确答案。
5. **原路与新路分开。** 首版复用`search_current_topic(current, resolved, ...)`，保持raw路、每路50候选和总50重排；相同文本只执行一次。先固定等权，不同时按d5t4结果调RRF权重。新来源投影只用于检索，不自动把全部历史材料提升为当轮required证据。

结构输出的宿主检查只在外部模型输出边界做一次：所选ID属于当次候选、所指原文span存在且来源属于先前可见集合。它检查身份/来源，不声称语义已验证。歧义、格式失败、超预算保留原query并记录原因，不取消本轮正常检索。首轮没有历史则直接走原query；不新增“至少多少证据才允许运行”的任意门槛。

允许历史assistant作为语境，不等于信其事实；source绑定也不等于证明话题相关。当前proposal需要通过“错误数字污染、错产品同名、话题转换、否定/比较”的实际对照来评价，而不是只靠schema宣告faithful。

## 最小工程接口草案

实现保持基础数据类型与执行顺序，不改C核心；下列示意不把resolver放入生成控制器：

```python
# 仅在source_topic显式启用时调用；返回已有状态的瞬时只读投影。
prior = retrieval_context_provider(active_turn)
resolved = resolver.resolve(active_turn.query, active_turn.history, prior)
# 最终OnlineTurn.query/history不变，只有检索用视图变化。
active_turn = retriever.retrieve_turn(active_turn, resolution=resolved)
```

- `run_sessions`增加可选provider；默认路径仍是原`retrieve_turn(active)`，不额外读SQLite。controller provider取`get_snapshot(f"{run_prefix}:{conversation_id}")`，首轮没有会话时返回空；native provider取既有`baseline.snapshots`。不调用创建session的接口。
- 新增短adapter承载投影与resolver（例如`adapters/query_resolution.py`），使用现有`EvidenceBlock/EvidenceKey/SourceSpan`。无需给SessionSnapshot或数据库增加字段；无需修改engine quote、host admission或epoch协议。失去原来源时记录missing，不从全语料静默补造“以前已看见”。
- factory为新query mode配置resolver，默认不开启；manifest记录模型/revision、prompt或结构合同、输入预算、实际query、source keys/spans、时长与状态，沿用既有query_view_traces。现有query-mode校验文案“必须user-only”应仅对旧mode有效，不能作为禁止合理新mode的全局规则。
- provider读取与模型解析都在retrieval_done之前，完整进入P3 pipeline latency。当前`per_turn_timeout`只包handler，不包retrieval；新增模型调用应使用明确传输timeout，不能误称原turn timeout已覆盖它。模型与答案生成共GPU时必须记录排队/模型驻留，不能在性能比较中隐藏额外调用。
- 可复用现有httpx/OpenAI-compatible完成接口或固定本地推理入口；只改写检索query不构成judge。无需建立多代理系统或重复验证每个内部类型。预算不足时记录未覆盖，不能默默截断来源后声明全量解析。

## 同41轮的离线对照协议

先用旧off完整41个已保存真实P3输入；不把新六次输出改成历史，不按d5t4是否成功挑样。**对第t轮，来源只能来自audit中同会话`turn < t`的visible证据；不能把第t轮audit_evidence送到检索前resolver。** 当前行的actual_history只包含过去，可复用；同样不能读它的当轮answer作为标签/提示。按序累积先前audit仅用于离线重建已有状态，不变成新在线持久化状态。

建议先做四个预先声明的视图，检索器与候选/重排预算一致：

1. 原current。
2. 现有literal user_topic（保留已知“没有用户命名来源”的负面结果）。
3. 同一模型的history-aware最小消解，无source资格约束，作为语境模型基线。
4. 来源支持的selector；必要时另以ConvSearch-R1原生rewrite作为更强独立对照，不假称其已有source约束。

先冻结所有query/trace/失败，再用现有`bench.retrieval_eval`的BM25/BGE/RRF/rerank；**全部排名完成后**读取dev qrels，复用已有R@5、nDCG@5、R@50和按会话统计。新query-document logits必须实际重算，不能复用不同query的旧logit。保持所有41轮，报告主题新增/未改/歧义/来源缺失，检索改好/变差/不变、模型与重排成本；只有5个独立会话，开发反复迭代不冒充独立test。

还需独立于qrels检查解析：产品/对象是否承接用户所问、是否加入未要求的用途/限制、来源是否实际早于当前轮、错误历史数值是否进入query、显式改题或否定是否被旧主题覆盖、版本和同名接口是否混淆。可用无任务ID依赖的合成结构变体补边界，但不能用这些assistant构造例冒充人工标注。

最后才比较固定历史的生成与完整P3闭环：主题解析是否把Cloudant相关材料取回、是否继续误用Discovery字段限制、引用与拒答是否合理、后续历史是否传播错误。检索改善、source字符串绑定成功和更短输出均不能替代该层检验。本轮已完成下述可测试工程，但没有新方法收益结论。

## 已实现的配置、状态与外部调用边界

`adapters/query_resolution.py`提供`PriorContext/PriorSource`、`snapshot_context`、controller/native provider和`TopicResolver`。`adapters/query_selector.py`提供固定配置的同步`OpenAICompletionClient`；它在已有retriever worker内调用，不依赖bench，避免循环import。真实controller CLI与Native CLI均已接入`run_sessions(...retrieval_context_provider=...)`。旧mode不构造selector、不读取该provider；原query/history不被改写。

新模式为`source_topic`和`assistant_topic_unverified`。两者使用相同SYSTEM/schema、同一模型选择器、同一显式历史窗口；区别是后者允许没有来源支持的assistant实体候选，因此名称和manifest始终标注unverified。两者都要求`current_topic_rrf`，最终候选池与原文通过既有`search_current_topic`和总50重排，未另写融合器。

```yaml
# 路径按已有固定本地快照替换；程序不会下载模型或索引。
sparse_index: data/cloud-bm25s
dense_index: data/cloud-bge-m3
encoder_path: models/bge-m3
reranker_path: models/bge-reranker-v2-m3
stage: hybrid_rerank
query_mode: source_topic
query_fusion: current_topic_rrf
candidate_k: 50
rerank_k: 50
rrf_k: 60
selector_history_turns: 3
selector:
  base_url: http://127.0.0.1:30000/v1
  model: Qwen/Qwen2.5-7B-Instruct
  revision: a09a35458c702b33eeacc393d103063234e8bc28
  context_window: 32768
  max_input_tokens: 24576
  max_new_tokens: 256
  timeout_seconds: 60
```

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

## 全41轮输入预算与待执行的冻结selector对照

已用真实本地Qwen tokenizer、无外部模型调用检查旧off全部41轮。第一次朴素实现把同名实体的多次完整支持记录重复塞入模型，最大达596,681 tokens；这是工程缺陷，已改为实体合并、紧凑模型目录、显式来源窗口。没有借删失败样本或任意qrel筛选掩盖它。

最终固定三轮selector语境、共同24,576输入上限的统计如下。分位数取36个有历史、拟调用模型的输入；另5个首轮不调用。全部41行和候选数保存在`runs/source-topic-preparation/frozen-v1/cases.jsonl`，摘要在同目录`budget-summary.json`。

| 36个拟调用输入 | source_topic | assistant_topic_unverified |
|---|---:|---:|
| 输入tokens均值 / P50 | 7958.8 / 7653 | 8537.2 / 8210.5 |
| 输入tokens P90 / 最大 | 14712 / 19638 | 16139.5 / 20813 |
| 实体候选均值 / P50 | 35.36 / 31.5 | 50.89 / 47 |
| 实体候选 P90 / 最大 | 67.5 / 95 | 105.5 / 126 |
| 输入上限8192时超限数 | 14/36 | 18/36 |
| 输入上限24576时超限数 | 0/36 | 0/36 |

预算覆盖不是优化成功：约19k–20k的输入处理可能比检索本身更贵，尚未测真实选择准确性/成本。组件默认输入上限仍8192；上面的首个完整dev配置显式使用24,576并保留8192敏感性。若后续改成“prior user＋assistant实体语境窗口”输入，那是另一个需要单独声明的view，不能暗中截短后与当前结果混算。

`runs/source-topic-preparation/run_selector.py`默认冻结/plan，已生成41个严格past-only cases与两mode的原始模型messages/schema；当前没有发送请求。显式`--execute`才对每轮交替两个mode顺序，总计预期72次模型调用、10条首轮跳过。它复用当前Qwen OpenAI-compatible服务，核对实际待发messages/schema和本地token计数与冻结输入一致，逐行保存原始模型响应、选择、失败/回退与成本；不生成答案、不检索、不读取qrels、不管理GPU服务。`--request-model`只用于操作者声明的served-name别名，与固定期望checkpoint分开记录。

ConvSearch-R1的独立pure adapter已在`adapters/convsearch_rewrite.py`：完整复用固定作者prompt和Qn/An序列化，保留完整Apache-2.0许可/NOTICE。`build_messages`接真实前序QA，`parse_rewrite`只在正常stop和唯一完整think/rewrite标签时返回可解析的unverified rewrite；raw保留，格式成功不代表来源支持。尚未加载其3B权重或将其设为默认。

最终本地定向验证为 **121 tests通过**：真实SQLite已提交provider、Native/controller两条实际CLI、首轮零调用、未展示archive/当前turn来源不能支持、缺失来源、显式切题、重复实体合并、明示三轮窗口、错误历史数值/代码不进入构造query、未知ID/截断/超预算、默认检索合同、离线visible与required不等价反例、ConvSearch格式，以及冻结driver的默认零客户端plan和fake HTTP原始响应保存。Fake执行只在测试临时目录，不能混为真实模型结果。Ruff通过；后续由统一GPU调度运行冻结selector，不在本地自动启动服务。
