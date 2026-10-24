# 相关工作核查、复用边界与本项目增量

核查日期：2026-10-24。以下结论依据固定源码及官方论文，未将相关论文的性能数值当作本项目结果。固定模型/引擎已完成限定 H200 配置的集成、标定与受控生命周期诊断；正式回答质量非劣与优于强基线的结论仍未建立，实际边界见[验证记录](verification.md)。新增 SmoothAgent、AKV、SCR 的作者代码核查与贡献交叉分析见[独立研究审查](research_review_20261024.md)。

有限双层缓存的 RLT、Continuum、KVFlow 接续筛选见[算法复用研究](cache_reuse_research.md)：分别核对树叶淘汰、保留期限与已知需求恢复的固定作者代码、许可、信息假设和最小可证伪对照，尚未迁移或声称性能收益。

## 复用清单

| 工作 | 固定版本与许可证 | 本项目关系 |
|---|---|---|
| SGLang | `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，[Apache-2.0](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/LICENSE) | 直接复用推理、UnifiedRadixCache、HiCache、原生调度与软引用；在固定源码上应用研究接口补丁 |
| ContextPilot | `b4c80327f6cfae2286839413731fcb6f53578b2f`，[Apache-2.0](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/LICENSE) | 独立原理适配：距离度量、匹配前缀对齐、复用注释；未集成整个可变索引服务 |
| LightRSI / TokenPilot | `9f0f19308a30445438779efbf3108a7d965a55c7`，[MIT](https://github.com/zjunlp/LightRSI/blob/9f0f19308a30445438779efbf3108a7d965a55c7/LICENSE) | 借鉴稳定前缀和批量生命周期门控；未移植其 TypeScript 运行时、任务估计模型或内容压缩器 |
| ReCAP | [论文 v1](https://arxiv.org/html/2609.40118v1) 为 CC BY 4.0；核查时[官方仓库为空](https://github.com/UCSB-NLP-Chang/ReCAP) | warm/idle 仅为受其缓存兼容机制启发的策略基线，不是持久注意力图的复现，无可直接复用代码 |
| EfficientAgent | `b21b12a07a3ccf473c82acf00820b1aa2073044b`，[Apache-2.0](https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/LICENSE) | 工作集与 host 写入准入为后续可选扩展依据；不能将 vLLM/LMCache connector 直接接入 SGLang |
| BGE-M3 / BGE-reranker-v2-m3 | `5617a9f61b028005a4858fdac845db406aefb181` / `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，官方模型卡分别MIT / Apache-2.0 | 直接加载固定权重，复用官方Transformers接口；BM25/RRF融合为方法适配。已经进入真实P3，检索收益与新增系统成本均保留，见[开发分析](hybrid_retrieval_dev_analysis.md) |
| MiniCheck | 官方代码 `b58b9fa69acbd1015ec970fa65dd752413a053d2`，Apache-2.0；DeBERTa权重 `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb`，MIT | 直接复用官方分块/score/inference，仅适配固定本地加载与可见证据输入；真实v2评分及引用增补已运行，不把增补合同通过当整体质量收益 |
| Qwen3-Reranker-0.6B | `fd9fb1d26c07223ced488065909faf522e29cc7d`，Apache-2.0；[归因记录](third_party/QWEN3_RERANKER_NOTICE.md) | 官方模板与末位yes/no打分已适配并运行同池对照；[负结果](qwen3_rerank_validation.md)支持当前保留BGE，不外推为较新方法普遍无用 |

本项目 ContextPilot 适配代码为按公开算法重新实现的独立短实现，没有分发上游源码副本。未来若直接复制或修改上游源文件，应同时保留原许可证、版权和修改说明；这里的许可证记录不是对全部上游依赖的审计。

## ContextPilot：哪些 API 可用，哪些不能直接调用

核查[固定 `live_index.py`](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/server/live_index.py) 后确认：

- `reorder(contexts, initial_tokens_per_context=0, conversation_id=None)` 返回重排上下文和原始索引映射，但也构建/更新索引、更新 `seen_docs` 和轮次。
- `deduplicate` 将文档分成新增/已见并生成引用注释，同时写 `seen_docs`。它的“已见”并不保证正文仍在当前模型输入中。
- `search_batch` 会累计统计；`search` 默认还更新访问时间。不能将这些有副作用方法直接当作候选构造的只读操作。
- `_reorder_with_prefix` 的核心为将命中前缀中的共有项提前，再保留剩余项原顺序。

[`adapters/contextpilot.py`](../adapters/contextpilot.py) 使用已提交的正文顺序和历史必需证据构成只读索引，以[官方标量距离](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/context_index/compute_distance_cpu.py)选择匹配上下文：

`distance = 1 - overlap / max(context_lengths) + 0.001 * mean_shared_position_difference`

匹配项提前，剩余项维持本轮检索顺序，额外条件依赖用稳定 key 顺序补齐。若执行账本记录了某组证据的真实顺序，以其为准，不让历史集合的排序覆盖真实顺序。引用注释只针对 `required ∩ currently_visible`；重建后已移除的正文必须重新插入。候选构造不写入任何已见集合。REBUILD 保留完整原始 QA，APPEND 保持已有消息不变。

依据[ContextPilot 论文 §5.3 的顺序注释](https://arxiv.org/html/2511.03475v4#S5.SS3)，B4 额外在新证据正文之后、当前问题之前写原检索优先序的稳定引用映射。排名不写进正文头，也不改旧消息，从而保留前部证据的共享机会；稳定 key 基线不额外写这个 B4 注释。

配置为 `context.layout: contextpilot`，默认 `stable_key` 仍为本项目稳定身份排序。两者共用同一 required 集合、版本冲突规则、token 预算和稳定引用；布局不能偷偷改变召回材料。这个 B4 适配实现了独立排序/注释路径，未复现上游层次聚类、跨会话索引或批量请求调度。上游 [`process_conversation_turn`](https://github.com/EfficientContext/ContextPilot/blob/b4c80327f6cfae2286839413731fcb6f53578b2f/contextpilot/pipeline/rag_pipeline.py) 辅助路径也没有作为本项目多轮执行器使用。

审查还修正了共同模板的次序：两种布局都先提供证据和复用注释，再放本轮问题，避免变化的问题提前打断相同证据的公共前缀。原始历史 QA 顺序不变，已有执行消息不原地改写。固定 Qwen tokenizer 的测试直接比较两条不同问题的真实 token LCP，确认公共前缀包含完整证据正文；这只能证明前缀机会，不等于实测缓存命中。

## TokenPilot / LightRSI 与 ReCAP

[TokenPilot 论文](https://arxiv.org/html/2606.17016v1)将稳定前缀与生命周期驱逐结合。固定 [`stable-prefix.ts`](https://github.com/zjunlp/LightRSI/blob/9f0f19308a30445438779efbf3108a7d965a55c7/components/packages/features/stabilizer/src/stable-prefix.ts) 把动态上下文从稳定指令部分移出；[`planLifecycleEviction`](https://github.com/zjunlp/LightRSI/blob/9f0f19308a30445438779efbf3108a7d965a55c7/components/packages/features/eviction/src/lifecycle-planner.ts) 按批量轮次触发估计，核对基础版本，在估计失败时旁路。本项目复用这种分离思路，保持 system 模板稳定、完整 QA 持久化、成功提交后再退役旧 epoch。固定周期策略仅对应调度思想，不能称为完整 TokenPilot。

[ReCAP 论文](https://arxiv.org/html/2609.40118v1)还保存注意力得到的重要度和依赖，结合当前请求选择历史。本文的自然回访规则和元数据依赖闭包不依赖注意力，不是该图算法。warm/idle 应使用真实缓存报价和明确阈值，结果表应标注“ReCAP 风格门控适配”；不得使用“ReCAP 复现”作为标签。

## EfficientAgent 与 SGLang

[固定 EfficientAgent connector](https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/efficientagent/kvtier/connector.py) 在 vLLM 的 LMCacheConnectorV1 上改变 CPU tier 保存决策：工作集估计与容量、实际占用和淘汰信号决定压力；新增完整 chunk 数超阈值才跳过保存，空间充足则放行。其 worker 同时避免重复复制已有 chunk。本项目保留 SGLang 原生存取/恢复/调度，后续如加入 host 准入，必须接上 SGLang 请求与实际写入路径，并单列消融；单独实现公式不能算引擎迁移完成。

[SGLang session 文档](https://docs.sglang.io/docs/advanced_features/session_radix_cache)和[固定 `session_ref_tracker.py`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)表明：session 引用是软保护，每次请求仍须提供完整目标输入，释放引用不会保证立即物理回收。本项目的 epoch 退休、报价和日志扩展围绕这些原生语义构建，不重写 attention kernel，也不将虚拟引用计数当作唯一 GPU/host slot 占用。

## 本项目新增模块 A 行为

[`adapters/corpus.py`](../adapters/corpus.py)接受 BEIR 或命名 passage 的 JSONL，使用白名单公开元数据保存真实 document/passage/version/source/span/parent_section，解析显式 `dependencies`、`condition_passage_ids`、`header_passage_ids`。依赖按版本精确匹配，裸 passage 多版本时要求调用方选定。正文不切分、不摘要、不归一化；配置锚点仅从文字中的代码标识提取。`CorpusIndex.resolve` 返回选定材料和依赖闭包，可直接传给服务。

现有检索返回的 `OnlineContext.document_id` 是 passage ID，可用 `CorpusIndex.enrich(contexts)` 匹配其完整原文后恢复 source document/version 元数据并追加条件闭包。元数据文件不能替换固定 top5 的正文；正文不一致或同文多版本歧义会明确报错。无元数据文件时保留原检索路径。

需要保留 B4 检索排名时，先取 `keys = index.retrieved_keys(contexts)`，再用 `index.resolve(keys)`取得正文与条件闭包，并把 `keys` 作为 `retrieval_priority` 传给候选。正文匹配执行一次；支持材料进入 required 集合但不获得虚构的 top5 排名。

显式关系现继续进入共享A/R提示：required材料的dependency refs及parent_section形成一次紧凑注记，避免“正文齐全、关系却在渲染时丢失”。这与[SPIRE §2.3/§3.1](https://arxiv.org/html/2604.20849v1)保留标题、父级结构并在同来源合并上下文的原理一致；本项目只使用已有确定元数据，没有迁移其HTML树检索或LLM过滤，也未确认可复用作者代码。该修复有可逆的字面关系合同，不构成隐式语义关系抽取或质量优胜。

[FullCite](https://arxiv.org/html/2606.07130v1)将文档归属、支持片段定位和claim-citation忠实性分开评价，进一步说明仅保证ref合法或quote存在不足以解决条件错绑。它已在先前调研中收录；本轮不重复实现其额外重试或模糊回绑。新增SDK检索入口及实际成本见[证据包接线](api_evidence.md)。

自然回访需当前问题出现明确指代词，再从有限近期已提交轮次寻找正文支持的唯一主题匹配；多候选不猜，不能利用数据集 follow-up 标签、reference 或未来问题。`context.revisit_mode: explicit` 可关闭该扩展做消融。每个候选的 `evidence_resolution` 记录检索、显式引用、配置锚点、自然回访或依赖原因。这些规则应在所有基线共用，不能把多拿材料的质量收益归因于布局或 KV 策略。

“旧版本是否也适用”等明确旧版请求可通过当前 passage 身份和历史已提交版本恢复唯一前版，保留双方版本说明；普通“那个超时”不会自动恢复已被当前版本替代的旧证据。

候选路径继续遵循入口校验一次、已提交状态可信、每个候选只 render/tokenize 一次。没有为这些扩展新增正文全量重复哈希、运行时镜像状态或多层相同校验。

## 由真实错误推进的新增研究选项

[研究路线](retrieval_research_next.md)纳入TabRAG（2025-11）的行/列/值绑定、FullCite（2026-06）的document+span定位、Deceptive Grounding（2026-07）的来源实体适用性。它们分别针对已取回的限额表被错误组合、历史助手句子被误当外部原文、同数值/相邻产品的错误归属。当前均为尚未实现/验证的实验选项：TabRAG已核作者固定代码及CC BY-NC许可，另外两项仅确认论文/作者说明，未冒称直接源码复用。

长度与决策侧的SSJF/LTR/PARS/TIE原理及固定源码见[长度研究](output_length_research.md)；目前实际训练使用轻量候选特征、NumPy与嵌套会话验证，不能称已经迁移其神经预测器。候选长度、条件误差、收益校准均已有真实pairs上的结果，见[决策验证](length_decision_validation.md)与[收益校准](gain_calibration_validation.md)。C侧则已有32K/16K原生写策略实测，不再把host实验仅列为未来计划。

## 本地检查与后续证据

`uv run pytest -q tests/test_candidates.py tests/test_corpus.py tests/test_tokenizer_optional.py` 覆盖版本/依赖、回访歧义、真实可见性、两种布局差异、完整 QA 和候选纯函数。真实 tokenizer 检查需额外依赖与缓存模型文件，使用 `CONTEXTKV_TEST_HF_TOKENIZER=1` 显式运行。

已有H200矩阵、P3闭环与有限容量实验继续保留；接续将布局、回访、候选决策、epoch退役和原生写策略放入匹配对照，报告质量、完整成本与实际资源。CPU检查、检索qrels、NLI分数和恢复量各自只验证不同环节，整体效果需要联合实测；不以任何单项替代其余证据，也不以一次负结果停止复用和机制改进。
