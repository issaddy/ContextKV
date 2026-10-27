# 原始段落上的混合检索与重排序实验

该入口支撑模块 A 的证据选择：复用固定 BGE 权重和官方 Transformers 接口，保留独立 BM25S 基线，提供 dense、RRF 混合与 cross-encoder 重排序。原文、段落边界与证据身份保持不变。当前多轮主线配置使用这一共享入口，开发数据用于改进，独立测试与开发结果分开报告。

在线最终选择至多五个不同内容组，相同标题/正文不重复占名额；已知版本、父级范围和条件依赖参与分组，所选代表仍是原始来源。普通前五段中的不同正文保留，再从已有评分池补齐，不增加模型调用。显式条件融合保留主任务锚点。`search_variants`与离线`retrieval_eval`仍提供原始评分阶段，便于比较排序器；在线分组结果及新增材料另由`HybridRetriever`记录，不把原始排序报告冒充集成效果。

## 选择依据与具体复用

当前 Cloud P3 的 BM25 基线提供了可检验的起点；真实答案审查同时发现 Cloudant 问题配到其他产品资料，以及引用可见但不支持当前论断的情况。前者需要改进检索，后者仍需 claim/source 支持检查，不能仅靠 reranker 解决。相关性与事实支持分别评价，见[质量评价](quality_acceptance.md)。

| 机制 | 官方依据与复用入口 | 本项目适配与限制 |
|---|---|---|
| BGE-M3 dense | [官方模型卡](https://huggingface.co/BAAI/bge-m3)提供多语言、长输入 dense 表征；本次只用 dense 分支。[固定 pooling 配置](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json)指定 CLS | `AutoModel` → CLS → L2 normalization；query 不加 BGE-v1.5 的检索指令。原始 `title + newline + text` 编码一次，1024 维 float32 memmap，精确内积检索。没有调用 M3 的 sparse/ColBERT 分支，不宣称复现其完整 ensemble |
| BM25 + dense 的 RRF | [Cormack 等 SIGIR 2009 原文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)给出按排名倒数融合的公式，避免直接混合不同量纲的分数 | 两路各取 50，等权 `sum(1/(60 + rank))`，ID 确定性破同分。公式为方法复用、本地纯函数实现；不复制第三方未明许可证代码，也不把 RRF 称为本项目创新 |
| BGE-reranker-v2-m3 | [官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)提供 `AutoModelForSequenceClassification` 与 query/passage 成对推理 | 对 RRF 前 50 重新排序，在线按内容组选择至多 5 段；raw logits 不解释成校准的相关概率、支持概率或回答正确率 |
| 多轮 query 表征 | [IBM 固定检索任务说明](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/mtrag-human/retrieval_tasks/README.md)包含最后问题、全部用户问题和 rewrite 设置，其实验说明检索器与问题表征都值得比较 | 实现 current、last3、all_user；第四个可选视角显式读取冻结 P2/P3 actual_history。没有把数据集给出的 rewrite、参考 assistant、qrels 或未来轮次带入检索，也没有以拼接历史冒充学习式 query rewriting |

选择 M3 的原因是可直接加载的 encoder/reranker 配套、长输入与多语言覆盖、官方调用简单，适合在现有 H200 上以较小工程改动测量收益。不是因模型更新年份断言其效果最强；若它的召回或排序收益不足，可以在同一原文、qrels 与查询视角下替换 encoder/reranker，继续比较较新的方法。原论文或模型榜单中的收益不转记为本项目结果。

固定模型与许可：

- `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`，模型卡 MIT。
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，模型卡 Apache-2.0。

运行只接受预先准备的本地目录，`local_files_only=True`，记录预期 revision、路径、device、dtype、软件版本、token cap 和 batch size。revision 来源明确标为 `operator_supplied_local_snapshot`，不伪称检查了全部模型文件的内容哈希。不安装 FlagEmbedding 整套依赖，不改项目 lockfile 或已有 torch 环境；复用官方兼容的 Transformers/torch 接口。

## 已实现接口与实验产物

[`adapters/hybrid_retrieval.py`](../../adapters/hybrid_retrieval.py)提供 `LocalBGE`、`DenseIndex`、`reciprocal_rank_fusion`、`search_variants`；[`bench/retrieval_eval.py`](../../bench/retrieval_eval.py)是独立 `build/evaluate` 入口。BM25S提供`retrieve(..., top_k=5)`。依赖安装由项目锁文件管理；索引保存时记录实际构建库版本，加载时另记`runtime_versions`，重新保存时更新构建版本。加载检查检索配方和文件内容，运行时不另加精确包版本门槛。

`prepare_queries` 将原始 MTRAG 转成只含当前及先前用户问题的白名单对象。`evaluate_queries` 不接受 qrels 或参考答案。CLI 完成所有排名、写入检索结果后才打开 qrels 做离线分析；`generated_history` 还检查 task/query、P2/P3 来源与完整先前轮次，不能用 P1 参考答案作为“实际历史”。该视角是冻结历史下的离线检索实验，不等同采用新检索器后自然演化的闭环历史。

输出目录保留三个文件：

- `manifest.json`：固定 cohort、查询来源、模型与语料配置、软件/设备、触顶计数与完成/失败状态；不覆盖旧目录。
- `retrieval.jsonl`：每轮、每 query 视角、每阶段的前五原文、完整候选 ID/分数、rerank 输入集合、耗时。先前已完成行在失败时保留。
- `summary.json`：每阶段 Recall/nDCG@5 和候选池指标、缺 qrels/无正例列表、按会话均值及耗时分布。

阶段为 BM25、dense、hybrid、hybrid_rerank。共用已计算的 BM25/dense 结果以节约 GPU，不重复四次相同编码。因共用阶段、索引暖化且没有独立重复测量，耗时是**探索性顺序阶段墙时**，不是正式吞吐/尾延迟实验；模型/索引加载、生成和未来在线集成开销尚未包含。

encoder 默认 8192 tokens。reranker API 默认 512，与模型卡的 Transformers 示例一致；实验可显式选择 1024/2048 等长度。固定 reranker config 的 `max_position_embeddings=8194`，512 不是其硬容量限制。本次首轮建议 1024，以容纳原段落与对话问题；记录触顶数后再判断是否需要长输入对照。`token_limit_hits` 表示达到配置上限，可能包含刚好等长的输入，不伪称精确截断计数。

构建时保留完整原始 passages，与 BM25 相同的语料摘要在加载/阶段边界核对一次；不逐请求重哈希语料或权重。每 4096 个 passage 打印一次进度及完成时进度，不增加恢复/检查点系统。72,442 × 1024 × float32 的 vectors 约 297 MB，查询使用 memory map 上的精确内积；当前规模无需额外 ANN 服务。

## 开发集运行方式

以下路径对应已准备的 H200 工作区；由同一操作者在生成负载结束后串行运行。模型使用明确 `cuda:0` 和 BF16，不自动挑设备。

```bash
python -m bench.retrieval_eval build \
  --corpus data/mtrag-official/cloud.jsonl \
  --output data/cloud-bge-m3 \
  --encoder-path models/bge-m3 \
  --device cuda:0 --dtype bfloat16 --batch-size 16

python -m bench.retrieval_eval evaluate \
  --input data/mtrag/RAG.jsonl \
  --split-manifest data/mtrag/split_manifest.json --split dev \
  --sparse-index data/cloud-bm25s --dense-index data/cloud-bge-m3 \
  --encoder-path models/bge-m3 \
  --reranker-path models/bge-reranker-v2-m3 --reranker-max-length 1024 \
  --device cuda:0 --dtype bfloat16 --batch-size 16 \
  --query-modes current last3 all_user \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/cloud-dev-hybrid-rrf50-rerank1024
```

当前固定 Cloud dev 为 5 会话、41 轮。稀疏语料 snapshot 为 `mtrag-53fef6f5917c7e10c436f697e69a955dbc795d81`，原始 passage 数 72,442，语料摘要 `04fb31f7d4301aa2fd3d12c161cf06265a9bbf9f41e56116d90637f9d053afad`；实际 build manifest 必须与稀疏 manifest 对齐。使用 passage-level 文件，不能把文档级 ID 与带 chunk offsets 的 qrels 混用。

省 GPU 的首轮只做上述检索，不生成 12 组答案。已有 BM25-only 可省略 dense/encoder/reranker 参数；固定真实历史视角可显式加 `--query-modes generated_history --history-audit <P2或P3完整audit.jsonl>`，并另存目录。正式性能实验再独立加载选定检索器，计入实际检索/重排成本和显存占用。

## 共享 P3 生成入口

[`adapters/retriever_factory.py`](../../adapters/retriever_factory.py)同时供控制器 CLI 和原生 B0/B1 CLI 使用。两者新增相同的 `--retriever-config`，接受 JSON 或 YAML，示例为 [`configs/retriever_hybrid.example.yaml`](../../configs/retriever_hybrid.example.yaml)。与其他命令行路径一样，资产相对路径以工作目录为基准；run manifest 保存配置原文对应的解析字段、路径基准、两个索引 manifest 与模型信息。

配置可选 `stage: dense / hybrid / hybrid_rerank` 和 `query_mode: current / last3 / all_user`。hybrid示例保留已实测的 last3 + hybrid_rerank、候选 50、reranker 1024、BF16、检索 GPU1；控制器默认稳定布局。多轮主线使用conversational示例，通过显式selector配置按需复用已有Qwen3服务。`--evidence-corpus`在现有重排前提供原始来源与标题，HTTP和两种CLI共用同一绑定。未传新参数时，现有 `--retriever-index` 或 `--corpus` 行为保持不变。配置文件与旧参数不可同时指定，以免运行来源不明确。

```bash
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m controller.cli \
  --config configs/qwen3_8b_scope.example.yaml \
  --state-dir runs/hybrid-p3-controller \
  bench data/mtrag/RAG.jsonl --protocol P3 --domain cloud \
  --split-manifest data/mtrag/split_manifest.json --split dev \
  --retriever-config configs/retriever_hybrid.example.yaml
```

原生生成使用同一份配置：在已有、经过 manifest 核对的 B0/B1 命令中，将 `--retriever-index` 替换为 `--retriever-config configs/retriever_hybrid.example.yaml`，其余模型、提示词、1024 输出上限与资源条件保持一致。矩阵顶层支持共享`retriever_config`；`comparison_scope: system`也支持方法各自的检索配方，`execution`则共享配方隔离执行机制。配置文件及索引manifest纳入既有`external_inputs`，详见[统一评价](../data_and_evaluation.md)。

`HybridRetriever.retrieve_turn`只接受 P3 的 `OnlineTurn`。current/last3/all_user/user_topic 从 `user_questions` 创建查询；显式 source_topic/assistant_topic_unverified 则使用实际历史和已提交来源投影，详见[来源主题解析](../research/source_grounded_query_resolution.md)。不读取参考 rewrite、qrels 或原始任务对象。P3 会话仍等当前答案提交后才进入下一轮；答案历史由原有 runner 生成，状态恢复、证据归档和提交语义不变。线上不开放离线 `generated_history` 视角。

模型与索引每个 run 初始化一次，`initialization_seconds` 单列；逐轮复用同一模型实例。并发会话对这一模型对的调用以一把锁串行，锁排队也发生在 `t_arrive → t_retrieval_done` 之间。runner 已有的 `time_origin=pipeline_arrival` 以及 `max(t_response_done,t_commit)-t_arrive` 完整响应指标覆盖检索、融合、重排序、排队和生成，因此没有再加一份检索时间。模型初始化与结束释放在逐轮计时之外；run 结束显式删除本进程模型并释放本进程闲置 CUDA allocator，避免下一矩阵 job 残留检索模型显存。

encoder/reranker 与 SGLang 共用 GPU 时，其他会话检索可能与生成竞争；这属于真实系统成本。当前锁只限制检索模型的并发，不阻断 SGLang。正式报告要同时给完整响应时延、检索阶段耗时、失败/未完成率、GPU 资源条件与检索模型驻留；已有空闲 GPU 成本标定不能自动当作新并发负载已验证。

接线测试运行两种真实 CLI 路径（测试模型与 fake HTTP），验证相同前五、每轮实际生成历史、无参考答案进入查询、每 run 只加载一次、模型释放、矩阵来源绑定以及检索排队计时。这些测试仅说明合同正确；真实 P3 的回答质量与整体成本仍需独立测量。

## 原始来源窗口

`source_window: true`显式启用来源窗口。复用[ParentDocumentRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever)的“小片段检索、较大范围阅读”原则及[LlamaIndex窗口检索](https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/)的相邻上下文组织方法：原有索引与50段评分不变，五个内容组代表确定后，补入每个命中前后各一段原文。窗口在同一来源、文档、版本和已声明章节内建立，重叠命中只补一次，不沿补充段落继续扩张。当前示例关闭该项：实测能补齐遗漏包地址，但缺少章节边界的语料会把指定版本和最新版流程一起引入，尚未建立默认开启的净收益。这里保留可复用工程，不据此否定结构化窗口方法，也不继续调半径。

一般语料以`document_id`、`span`提供顺序；固定MT-RAG snapshot使用原有`parent-start-end` passage ID。官方说明这些是[语料入库时的offset](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/README.md)，实际检查也发现不少相邻重叠正文不一致。因此offset只用于顺序，不能据此无损拼接全文。本实现保留每段完整正文、原始身份及独立引用，不引入合成段落或重新编码索引。

`OnlineTurn.contexts`仍是原来的评分命中，`supporting_contexts`另存未评分的窗口原文。HTTP、控制器CLI和Native入口共同补齐显式依赖后交给A/R，只有原命中进入检索优先级，B按真实输入计价，C按既有事务保存和退役。配置需要`--evidence-corpus`，不增加解释、encoder或reranker调用。代价是更多输入和缓存占用，原有精确token预算与原生准入继续生效；当前没有宣称窗口必然改善全部任务或解决无限历史容量。实际质量检查统一记录在[数据与评价](../data_and_evaluation.md)。

## 可选的召回/重排查询分离

共享配置新增 `rerank_query_mode`，仅在 `stage: hybrid_rerank` 生效：

- `resolved`（默认）：继续用召回的 topic/解析后查询重排，保留原行为。
- `current`：候选池仍按原 query_mode/query_fusion 产生，只用当前原问题评分。
- `user_topic`：候选池不变，评分用已有确定性 user_topic 视图，仅从用户问题取词。

例如在已有 `user_topic + current_topic_rrf` 配置中增加 `rerank_query_mode: current`，仍召回原问题和 user_topic 两路、统一取 RRF 前50；只改变同50段分类时的查询。不会把当前问题或历史替换成生成答案，不增加重排候选数量，不按相关性标签选择模式。controller/native P3 与 `bench.source_topic_eval` 共用这一选择函数。

底层 `search_variants` / `search_current_topic` 接受 `rerank_query=None`；None 分别保留原 query/topic，显式字符串仅传给 reranker。输出及 fusion trace 记录实际 `rerank_query`；主链 trace 还记录 `rerank_query_mode`，即使后续 encoder/reranker 失败也保留已产生的 query resolution 原响应、耗时与 retrieval error，原异常继续向外抛出。`retrieval_elapsed_ms` 是包括锁等待、解析和排名的本地墙时，resolver 原有 `elapsed_ms` 仍只表示解析阶段，不能两者相加。

候选池召回和最终重排可能具有不同的查询需求；本接口允许显式分别配置，并保留共同候选与原文，实际用途按目标负载判断。

## 可证伪假设与后续改进

1. **召回互补。** dense/RRF 的候选池召回若没有高于 BM25，先查产品实体、追问、原始语料 ID、query 视角与相关性标签覆盖，不直接堆更强生成提示词。若召回改善但前五不改善，重点检查排序与重排器。
2. **重排序有效。** 在相同候选池上，reranker 应提高前五相关性；若更差，保留负结果，检查多轮 query 噪声、长度触顶与产品版本混淆，再比较 current/last3/all_user 或有依据的 local rewrite。不得只报最优方法而丢失其他视角。
3. **对话条件有效。** 多轮视角可补省略的实体，也可能累积过时主题；按追问与主题切换查看具体排名、按会话汇总。当前 5 会话适合探索，不把逐轮样本当成 41 个独立会话。
4. **检索收益能转化为问答收益。** 选定设置后，在共享问题、语料、生成模型、提示词、1024 输出预算和资源条件下跑完整 P3；逐 claim 检查支持、版本/命令范围和不可回答行为。qrels 改善并不自动保证答案更正确，资料更丰富也可能让输出更长。
5. **质量增强能与缓存协同。** 在选定检索设置内部再共享证据比较布局/APPEND/REBUILD；同时单列检索增强前后完整系统成本。新增检索、重排、grounding 或修订开销全部计入，不把基线材料不同归因于缓存控制器。

本次已通过原文保持、RRF 同分、真实 BM25 默认兼容、白名单/历史来源、标签后置分析、模型/语料对应、离线 CLI 和 41 轮 cohort 等测试。mock encoder/reranker 只验证数据合同，不能用于报告真实检索收益。真实模型实验完成后再填写各方法/会话结果及错误分析。

### 从原始文档恢复阅读上下文

在`source_window: true`下可提供`source_documents: data/mtrag-source-documents/cloud.jsonl.zip`，使用同一固定语料的原始document-level JSONL（或只含一个JSONL的zip）。记录包含`document_id/text`，可带`url/source_uri/title`。Cloud原件来自[固定上游版本](https://github.com/IBM/mt-rag-benchmark/tree/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/document_level)，不更换现有passage索引或向量。

检索仍返回原始片段。阅读阶段通过文档身份和仅允许空白差异的字面匹配定位原文，补完整边界自然段并带上两侧各两个自然段；同源重叠窗口合并。实际模型读取原始窗口，原检索片段及分数另行记录，不将两者重复放进输入。已有完整片段、没有唯一对应原文的片段，以及有显式章节/依赖关系的来源保留原路径。窗口保留原文字符范围和来源，A/R共享同一阅读材料，实际tokens继续进入预算和B/C流程。

原文只在启动时加载，片段定位按固定语料项复用，不增加LLM调用、索引搜索或重排。未提供source_documents时仍使用原有相邻检索片段窗口。生成评价包含实际阅读材料；检索评价从单独的retrieval_contexts投影原排名，不能把上下文扩展当作新的检索命中。该机制的质量和成本需按公开任务验证，默认配置尚未启用。

### 按当前任务选择完整阅读材料

`evidence_selection: semantic`复用已有解释客户端，允许与`source_window`组合。候选阅读窗口及必要依赖在选择前准备，选择器同时看到完整原文、当前问题和结构化任务条件；选中后交给共同A/R的正文与依赖来自这份目录，不再额外扩张未选择的窗口。原文窗口按每个命中独立构造，相同窗口只占一项，避免大候选池经重叠连锁合并成整篇文档。

完整候选及其依赖按实际分词预算装入目录，放不下的项跳过，继续考虑后续完整项。只允许选择已装入的ID；没有可用目录或调用失败时沿用原始排名回退，不截断原文或重试模型。派生窗口保留对应的原始检索片段，检索评价使用这些来源ID，阅读质量使用实际材料。

复合需求仍使用批量编码和联合召回。启用联合语义选择时，取消重复的逐需求及条件重排；可使用`hybrid`直接提供候选，或`hybrid_rerank`保留一次主任务重排。选择HTTP仍在本地检索锁外执行，模型输出不改变事实正文。该组合为显式机制配置；默认配置及整体效果结论以[当前实现](../implementation_plan.md)和[评价记录](../data_and_evaluation.md)为准。
