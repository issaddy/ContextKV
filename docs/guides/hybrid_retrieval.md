# 混合检索配置与运行

该入口支撑模块 A 的证据选择：复用固定 BGE 权重和官方 Transformers 接口，保留独立 BM25S 基线，提供 dense、RRF 混合与 cross-encoder 重排序。原文、段落边界与证据身份保持不变。当前多轮主线配置使用这一共享入口，开发数据用于改进，独立测试与开发结果分开报告。

在线最终选择至多五个不同内容组，相同标题/正文不重复占名额；已知版本、父级范围和条件依赖参与分组，所选代表仍是原始来源。普通前五段中的不同正文保留，再从已有评分池补齐，不增加模型调用。复合需求选择保留整体任务的首个正文组，条件通过统一状态投影进入各需求查询。`search_variants`与离线`retrieval_eval`提供原始评分阶段，便于比较排序器；在线分组结果及新增材料由`HybridRetriever`另行记录。

## 选择依据与具体复用

Cloud P3答案审查发现两类问题：检索混入其他产品资料，以及回答论断缺少来源支持。前者通过查询与重排改进，后者结合实际回答和来源评价，见[质量评价](quality_acceptance.md)。

| 机制 | 官方依据与复用入口 | 本项目适配与限制 |
|---|---|---|
| BGE-M3 dense | [官方模型卡](https://huggingface.co/BAAI/bge-m3)提供多语言、长输入 dense 表征；本次使用 dense 分支。[固定 pooling 配置](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json)指定 CLS | `AutoModel` → CLS → L2 normalization；query 直接编码。原始 `title + newline + text` 编码一次，1024 维 float32 memmap，精确内积检索 |
| BM25 + dense 的 RRF | [Cormack 等 SIGIR 2009 原文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)给出按排名倒数融合的公式，避免直接混合不同量纲的分数 | 两路各取 50，等权 `sum(1/(60 + rank))`，ID 确定性破同分；按论文公式实现为本地纯函数 |
| BGE-reranker-v2-m3 | [官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)提供 `AutoModelForSequenceClassification` 与 query/passage 成对推理 | 使用 raw logits 对 RRF 前 50 排序，在线按内容组选择至多 5 段 |
| 多轮 query 表征 | [IBM 固定检索任务说明](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/mtrag-human/retrieval_tasks/README.md)包含最后问题、全部用户问题和 rewrite 设置，其实验说明检索器与问题表征都值得比较 | 实现 current、last3、all_user；第四个可选视角显式读取冻结 P2/P3 actual_history。检索输入取当前问题和已有实际历史，参考答案与 qrels 用于离线评分 |

方法适配关系见[机制与研究依据](../research_contributions.md)，实际质量和成本见[数据与评价](../data_and_evaluation.md)。

运行使用预先准备的本地目录，`local_files_only=True`，通过 Transformers/torch 接口加载。记录预期 revision、路径、device、dtype、软件版本、token cap 和 batch size；revision 来源标为 `operator_supplied_local_snapshot`，表示操作者声明的本地版本。

## 已实现接口与实验产物

[`adapters/hybrid_retrieval.py`](../../adapters/hybrid_retrieval.py)提供 `LocalBGE`、`DenseIndex`、`reciprocal_rank_fusion`、`search_variants`；[`bench/retrieval_eval.py`](../../bench/retrieval_eval.py)是独立 `build/evaluate` 入口。BM25S提供`retrieve(..., top_k=5)`。依赖安装由项目锁文件管理；索引保存时记录实际构建库版本，加载时另记`runtime_versions`，重新保存时更新构建版本。加载检查检索配方和文件内容，运行时不另加精确包版本门槛。

`prepare_queries`提取当前及先前用户问题，`evaluate_queries`生成排名，CLI保存结果后读取qrels评分。`generated_history`读取冻结的P2/P3实际历史，并核对task/query、来源和先前轮次，用于固定历史下的离线检索比较；闭环会话使用各方法实时生成的历史。

输出目录保留三个文件：

- `manifest.json`：固定 cohort、查询来源、模型与语料配置、软件/设备、触顶计数与完成/失败状态；不覆盖旧目录。
- `retrieval.jsonl`：每轮、每 query 视角、每阶段的前五原文、完整候选 ID/分数、rerank 输入集合、耗时。先前已完成行在失败时保留。
- `summary.json`：每阶段 Recall/nDCG@5 和候选池指标、缺 qrels/无正例列表、按会话均值及耗时分布。

BM25、dense、hybrid、hybrid_rerank共用BM25/dense计算结果，按顺序记录各阶段墙时。该计时覆盖检索阶段，模型/索引加载、生成及完整系统吞吐另行测量。

encoder 默认 8192 tokens。reranker API 默认 512，与模型卡的 Transformers 示例一致；固定模型 config 的 `max_position_embeddings=8194`。主线配置采用 1024，以容纳原段落与对话问题，也可按实际长度配置其他上限。`token_limit_hits` 统计达到配置上限的输入，包含刚好等长的情况。

构建时保留完整原始 passages，与 BM25 相同的语料摘要在加载/阶段边界核对一次。每 4096 个 passage 及完成时打印进度。72,442 × 1024 × float32 的 vectors 约 297 MB，查询使用 memory map 上的精确内积。

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

上述命令执行检索评价。已有 BM25-only 可省略 dense/encoder/reranker 参数；固定真实历史视角可显式加 `--query-modes generated_history --history-audit <P2或P3完整audit.jsonl>`，并另存目录。系统性能实验独立加载选定检索器，计入实际检索/重排成本和显存占用。

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

encoder/reranker与SGLang共用GPU时，检索和生成会竞争资源。检索模型锁限定其自身并发，SGLang独立运行。系统报告记录完整响应、检索耗时、失败/未完成率、GPU配置及检索模型驻留，并在目标并发负载下评价。

接线测试运行两种真实 CLI 路径（测试模型与 fake HTTP），验证相同前五、每轮实际生成历史、无参考答案进入查询、每 run 只加载一次、模型释放、矩阵来源绑定以及检索排队计时。真实 P3 的回答质量与整体成本见[数据与评价](../data_and_evaluation.md)。

## 原始来源窗口

`source_window: true`显式启用来源窗口。复用[ParentDocumentRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever)的“小片段检索、较大范围阅读”原则及[LlamaIndex窗口检索](https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/)的相邻上下文组织方法：原有索引与50段评分不变，五个内容组代表确定后，补入每个命中前后各一段原文。窗口在同一来源、文档、版本和已声明章节内建立，重叠命中只补一次，不沿补充段落继续扩张。当前示例关闭该项：实测能补齐遗漏包地址，但缺少章节边界的语料会把指定版本和最新版流程一起引入，后续重点是版本和章节的适用范围。

一般语料以`document_id`、`span`提供顺序；固定MT-RAG snapshot使用`parent-start-end` passage ID。官方将其定义为[语料入库时的offset](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/README.md)，部分相邻重叠正文存在差异。因此窗口按offset排序，各段保留完整正文、原身份和独立引用。

`OnlineTurn.contexts`仍是原来的评分命中，`supporting_contexts`另存未评分的窗口原文。HTTP、控制器CLI和Native入口共同补齐显式依赖后交给A/R，只有原命中进入检索优先级，B按真实输入计价，C按既有事务保存和退役。配置需要`--evidence-corpus`，复用已有解释、encoder和reranker结果。新增输入和缓存占用计入精确token预算与原生准入。实际质量检查统一记录在[数据与评价](../data_and_evaluation.md)。

同时配置`source_documents`时，读取匹配的原始文档，完成命中边界段落并保留前后各两段；合并重叠窗口后替换对应片段，原排名存入`retrieval_contexts`。带显式章节或依赖的片段保留原有闭包。JSONL或单JSONL的ZIP记录需要`document_id`、`text`，并以`corpus_id`、`snapshot_id`、`version`及`source_uri`/`source`/`url`绑定身份。`snapshot_id`须对应实际检索索引，不一定等于资料业务版本；仅在两者相同时可省略独立`version`。相同句子不能证明来源或版本相同，身份不明或字面不匹配时保留原命中。

发布方的Cloud原始文档ZIP没有上述快照绑定，不能直接把示例路径视为已验证的阅读输入。应先核对发布文件哈希和索引映射，再准备带身份的原文记录。本轮父文档实验同样曾因MultiDoc业务版本与结构化索引快照混淆而未生效，修正经过4110条映射核验；整份父文档及其预算选择目前只在隔离原型中，尚未进入共享配置，见[实验记录](../data_and_evaluation.md#原始父文档补全条件覆盖与完整成本)。

## 可选的召回/重排查询分离

共享配置新增 `rerank_query_mode`，仅在 `stage: hybrid_rerank` 生效：

- `resolved`（默认）：继续用召回的 topic/解析后查询重排，保留原行为。
- `current`：候选池仍按原 query_mode/query_fusion 产生，只用当前原问题评分。
- `user_topic`：候选池不变，评分用已有确定性 user_topic 视图，仅从用户问题取词。

例如在已有 `user_topic + current_topic_rrf` 配置中增加 `rerank_query_mode: current`，仍召回原问题和 user_topic 两路、统一取 RRF 前50；只改变同50段分类时的查询。不会把当前问题或历史替换成生成答案，不增加重排候选数量，不按相关性标签选择模式。controller/native P3 与 `bench.source_topic_eval` 共用这一选择函数。

底层 `search_variants` / `search_current_topic` 接受 `rerank_query=None`；None 分别保留原 query/topic，显式字符串仅传给 reranker。输出及 fusion trace 记录实际 `rerank_query`；主链 trace 还记录 `rerank_query_mode`，即使后续 encoder/reranker 失败也保留已产生的 query resolution 原响应、耗时与 retrieval error，原异常继续向外抛出。`retrieval_elapsed_ms` 是包括锁等待、解析和排名的本地墙时，resolver 原有 `elapsed_ms` 仍只表示解析阶段，不能两者相加。

候选池召回和最终重排可能具有不同的查询需求；本接口允许显式分别配置，并保留共同候选与原文，实际用途按目标负载判断。

## 从原始文档恢复阅读上下文

在`source_window: true`下可提供`source_documents: data/mtrag-source-documents/cloud.jsonl.zip`，使用同一固定语料的原始document-level JSONL（或只含一个JSONL的zip）。记录包含`document_id/text/snapshot_id`，`snapshot_id`必须与证据索引一致；产品版本不同于快照名时还须声明准确的`version`。`corpus_id`如有声明必须一致；片段有明确`url/source_uri/source`时原文必须提供相同来源，片段没有明确来源地址时则须声明相同的`corpus_id`。字段也可放在`metadata`中。缺失身份声明的旧文件仍可加载，但保持原检索片段，不据共同句子推断原文版本。Cloud原件来自[固定上游版本](https://github.com/IBM/mt-rag-benchmark/tree/53fef6f5917c7e10c436f697e69a955dbc795d81/corpora/document_level)，不更换现有passage索引或向量。

检索仍返回原始片段。阅读阶段通过文档身份和仅允许空白差异的字面匹配定位原文，补完整边界自然段并带上两侧各两个自然段；同源重叠窗口合并。实际模型读取原始窗口，原检索片段及分数另行记录，不将两者重复放进输入。已有完整片段、没有唯一对应原文的片段，以及有显式章节/依赖关系的来源保留原路径。窗口保留原文字符范围和来源，A/R共享同一阅读材料，实际tokens继续进入预算和B/C流程。

原文在启动时加载，片段定位按固定语料项复用。未提供source_documents时使用相邻检索片段窗口。生成评价读取实际阅读材料，检索评价使用retrieval_contexts中的原排名。该选项默认关闭，质量与成本按公开任务验证。

默认`ranked`选择至多五个正文及适用范围组，组内真正召回的来源均保留独立引用资格和原始分数；因此来源数量可以超过五个。相同正文由现有evidence alias共享，来源身份、版本、适用范围与依赖仍各自保留。不会从全语料补入同文但未召回的来源，也不把共享正文视作独立佐证。

## 联合语义选材配置

`evidence_selection: semantic`使用同一解释客户端联合更新任务和选择来源，需要`query_mode: conversational`、selector及证据元数据。可与`source_window`和`source_documents`组合；`hybrid`直接提供候选，`hybrid_rerank`增加一次主任务重排。完整阅读目录、依赖、预算装入及无效输出处理见[联合决策机制](../research/source_grounded_query_resolution.md#复合需求与选材)。默认仍为`ranked`，已有比较集中在[评价记录](../data_and_evaluation.md#一次联合任务与来源决策的开发比较)。


## 可选的会话查询适配器

[`configs/retriever_dialogue.example.yaml`](../../configs/retriever_dialogue.example.yaml)使用`query_mode: dialogue`、`stage: dense`和可选`query_adapter_path`。它把当前原问题放在首位，再按由近到远的顺序放入已经提交的实际问答，只用于查询编码；生成器仍收到原问题、完整实际历史和原始证据，`QueryContext`为空。该模式不调用任务解释器或重排模型，也不生成阅读计划。默认conversational配置保持不变。

查询编码沿用BGE-M3的8192-token上限。超限时逐个移除最旧的完整问答，记录保留/移除轮次；当前问题本身超限则报错，不静默截断。省略仅发生在检索视图，不能据此宣称长历史条件已正确承接。实际历史只能从控制器已提交快照或P3 runner进入，参考回答和未来消息不能进入该视图。

查询侧LoRA复用原文向量，不重建文档索引。加载时核对`query-adapter.json`的完成状态、schema、query_mode、基座配置摘要、文档encoder身份、FP32查询归一化及两份适配器文件的SHA256；文档索引仍核对原基座空间。查询适配器不能用于构建文档向量。schema 1的current/dialogue对应`literal-dialogue-query-v1`，训练代码摘要保留为来源记录；变更输入配方需重新验证适配器。

依赖通过`uv sync --extra tokenizer --extra retrieval --extra query-adapter --locked`准备；PEFT固定为0.21.2。训练extra同步采用该版本，已有训练记录的环境身份保留。当前GPU工作区已准备`models/bge-m3-dialogue-query-v1`，其权重SHA256为`951804ff5bc23b18b4cc3772d30ed9775a956a5ef1676be39e77e30f6d5d6caf`，配套原文encoder为固定BGE-M3。示例使用Cloud索引，MultiDoc2Dial改为对应的稀疏/稠密索引和语料，不能更换文档编码空间。

CLI、HTTP及原生P3共享已有`--retriever-config`入口，将其设为上述示例即可选择该组件。GPU实测使用独立目录`runs/query-encoder/integrated`及隔离依赖`runs/query-encoder/deps`，没有替换常规已安装交付。仅删除`query_adapter_path`得到的是未训练历史视图，Cloud对照明显下降，不应将它视为同等效果的简化部署。

该权重仅由MultiDoc2Dial官方训练对话学习，属于有训练依赖的可选检索组件；没有微调8B回答模型。配方、数据摘要和固定实验脚本保存在`runs/query-encoder/`，结果见[查询侧会话检索对照](../data_and_evaluation.md#查询侧会话检索训练与完整系统对照)。这批结果支持继续验证，尚不能证明跨模型、长历史、来源版本更新或并发回访场景的收益。
