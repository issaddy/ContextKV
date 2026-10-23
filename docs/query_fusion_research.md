# 原查询与主题补全双路：近期方法核查及冻结池诊断

2026-10-24。问题来自真实开发结果：`user_topic` 将 Cloudant SDK 支持段排入前两名，却把 WebSocket/customization ID 的一个定义正例从融合第9位推到第53位，超出50篇重排池。[完整五视图结果](user_topic_retrieval_validation.md)仍保留。为保留两种互补视角，现已实现显式可选 `query_fusion: current_topic_rrf`，默认仍为 `single`。CPU冻结池与实现合同已验证，新增神经重排/完整回答收益尚待实测。

## 四项原始方法核查

只采用原论文与论文确认的作者仓库。以下是机制/成本/复用入口，不把其他数据集的提升数字换算为本项目收益。

| 方法与固定依据 | 对本问题有用的机制 | 复用状态与取舍 |
|---|---|---|
| **Faithful Query Resolution，2026-10**：[arXiv v1 §4.1–4.4](https://arxiv.org/html/2610.15034v1) | 将忠实消解与扩大检索分开；保留原话、最小消解及学习式改写的独立检索路径。对 query×BM25/SPLADE 所有列表一次加权RRF，k=60，忠实视图权重固定2、其他1，并去重相同query | 这里的2:1**不是学习到的权重**；原方法融合后没有neural reranker。原文还说明同一iKAT-2023材料参与训练/设计/离线评估，交互实验包含其他修复，检索领先未对应最高满意度。可借用独立路径和显式权重原则，不能声称完整复现或人类满意度提升。论文CC BY4.0；截至本次核查未找到论文关联的官方完整代码/权重入口，未复制代码 |
| **Learning Multi-Step Query Rewriting via Corpus Feedback，2026-10**：[arXiv v1 §3–4.5及附录](https://arxiv.org/html/2610.10955v1) | 以 intent/rephrase/hyde/stop 操作交替查询与观察检索内容；训练目标只用最后排名的相关性奖励，后续伪文档由此前段落提供词汇，不是一次盲目扩写 | Qwen3-4B先SFT后GRPO，蒸馏来源32B；最多5步，每步接5段、每段128 tokens，每次最多生成512。正文明确多步累积上下文会增加成本，单纯增加动作类型并不总有效。对本项目可先比较有来源的一次补充查询/停止，而非直接照搬训练代理。论文为arXiv非独占分发许可；未确认作者公开运行仓库/权重，不能把AReaL训练库当作其完整实现 |
| **ICR，EMNLP 2025**：[正式论文 §3.3–3.6](https://aclanthology.org/2025.emnlp-main.496.pdf) | 内部澄清问题与改写交替，渐进SFT及DPO约束过度推敲、过早结束和分解不足；PRRF按迭代位置i赋权，分数为各轮 `i/(60+rank)` 之和 | 训练轨迹筛选使用gold passage相关性指标，不能把这项停止规则搬到在线。澄清是模型内部过程，不等于逐轮打断用户。后轮较高权重依赖其训练假设，不能因词法topic排在第二路就断言它更可靠。公开论文和附页未提供本次可确认的官方推理仓库/权重许可；只借鉴保留中间视图与诊断过度补全的原则 |
| **CompCQR，2026-09**：[arXiv v1 §2–3](https://arxiv.org/html/2609.14646v1)、[作者MIT固定源码](https://github.com/YunahJang/CompCQR/tree/04c9882ee9ccf17a31670dd406e3b666b48b896d) | 两次LLM构造改写/假设答案/后续问答组件，与原query形成五组件，枚举有序子集并对文档融合，再由LLM选择最佳段落。组件顺序本身会改变召回，是保留不同输入视角的依据 | 可直接复用的小入口是[`variations.py`](https://github.com/YunahJang/CompCQR/blob/04c9882ee9ccf17a31670dd406e3b666b48b896d/compcqr/variations.py)的组合生成与[`fusion.py`](https://github.com/YunahJang/CompCQR/blob/04c9882ee9ccf17a31670dd406e3b666b48b896d/compcqr/fusion.py)的`rrf_score`/`top_documents`；[MIT许可](https://github.com/YunahJang/CompCQR/blob/04c9882ee9ccf17a31670dd406e3b666b48b896d/LICENSE)已核。默认325次检索和k=0针对大量变体，不能称为只需两次低成本查询。本次直接复用项目现有等权RRF纯函数，未复制上游代码；只适配少量互补视图思想，不生成假设答案或未来问句 |

这些工作共同提示：**补全是否忠实、候选覆盖是否增加、最终重排是否有效、答案是否更好是四个不同问题**。本项目当前 `user_topic` 只有原用户span的可追踪性，并不等于语义上已完成最小忠实消解。若后续引入生成式组件，生成假设只能作为检索线索，不能写成原始 evidence 或已确认用户意图；模型/原文许可、输入来源和额外生成成本需单独记录。

## 现在已经执行的CPU诊断

脚本 `runs/user-topic-retrieval-review/frozen_pool_probe.py` 只读五视图的保存排名，构造新50篇池，再读取相同dev qrels计算覆盖；没有GPU、搜索或模型调用，也没有使用标签选择在线规则。输出 `frozen-pools.jsonl` 记录每个候选、来源覆盖及缺失logit，`frozen-pool-summary.json` 保留全部预设策略。

预设策略为原current前50、原user_topic前50、一次性 query×retriever 的RRF（权重raw:topic=1:1、2:1、1:2），以及两路各保留前25、去重后交替补齐到50的quota对照。RRF k固定60；相同query文本只运行一次，避免重复投票。本次41轮只有4轮 current与user_topic文本不同，因此实际部署这个候选方案会增加4次 query编码/双路检索，而总cross-encoder预算仍是每轮50对。它比“每轮双倍重排”更有希望，但额外编码/CPU搜索与模型共GPU成本仍需实测。

| 冻结池策略 | dev R@50 | 相对user_topic改善/持平/退化（39标注轮） | b461t4定义正例在新池中的位置 |
|---|---:|---:|---:|
| current50 | 76.07% | 1 / 38 / 0 | 9 |
| user_topic50 | 74.79% | 0 / 39 / 0 | 不在池中 |
| equal RRF 1:1 | 76.07% | 1 / 38 / 0 | 7 |
| RRF raw:topic=2:1 | 76.07% | 1 / 38 / 0 | 5 |
| RRF raw:topic=1:2 | 76.07% | 1 / 38 / 0 | 14 |
| quota 25+25 | 76.07% | 1 / 38 / 0 | 9（仅构造次序） |

四种方案都找回 `ibmcld_13386-1713-3820`，并保留d5 SDK两段正例，现有标注覆盖不能区分它们谁更好。报告没有按已知正例挑最优权重，也不将quota的构造次序计算为有效排名分。其余37个相同query的任务沿用同一候选集合，新增开销只发生在不同视图任务；另一个变化是附件纠正轮，未使用任务专名路由。

**还不能报告新top5。** equal-RRF在b461t4的新50池中，有22篇没有此前以user_topic query得到的cross-encoder logits；d5t9也有17篇缺失。raw-query logits不能替代topic-query logits，不能将缺项补零或只挑有分段落计算虚假新结果。当前报告只有pool coverage，最终神经重排与回答指标均未测量。

## 已实现的固定预算入口

[`search_current_topic`](../adapters/hybrid_retrieval.py)由离线评估和P3共享工厂调用。仅 `query_mode: user_topic` 与 `stage: hybrid`/`hybrid_rerank` 支持双路，避免将多路结果错称单独BM25或dense阶段。两个文本相同则精确沿用旧 `search_variants`，不重复编码、检索或评分；文本不同则对原current和topic各取BM25/dense前50，一次等权RRF（k=60），取融合前50，**全部以topic query重新评分一次**。没有将旧raw logits混入，也没有任务专名、标签、回答或未来轮路由。P3只改变检索上下文，原生成问题、历史、完整证据正文与证据ID保持原有接口。

配置最小增量是 `query_mode: user_topic`、`query_fusion: current_topic_rrf`，其他BGE路径/预算沿用共享retriever配置。原 `--retriever-config` 同时适用于controller和native。离线新增 `--query-fusion current_topic_rrf --query-modes user_topic`；只输出有明确含义的hybrid/hybrid_rerank两个阶段。现有manifest/结果trace记录两条实际query、各路原排名、实际候选数、去重标记、最终重排IDs、评分query及检索/重排时间，不新增独立日志或跨请求缓存。双路实现复用每路原有hybrid阶段，因此包含这两次局部RRF与最终四榜RRF的成本；端到端检索计时包含这些工作。

验证包括：相同query的原候选/分数/排序完全相等且模型各调用一次；不同query的新池包含原路独有文档，全部用topic评分且总预算不翻倍；共享工厂与两条真实CLI均持久化同一配置，离线qrels在所有排名之后才读取。64项相关测试通过（另1项可选全量数据测试未运行）。`runs/user-topic-retrieval-review/verify_shared_fusion.py`用保存的原排名回放生产纯逻辑，41/41新池与之前冻结equal-RRF池完全一致；预计45次query检索，对应41轮中4次额外查询。75个query-document对缺少旧topic logits，不能由这项CPU回放生成新top5。

## 可复跑的真实模型对照

`runs/run-current-topic-fusion-retrieval.py`已准备：同41轮dev、同BGE固定权重/BF16/cuda:0、8192编码与1024重排长度、batch16、每路50且总重排50；模型/索引加载一次。每个query先 `single` 后 `current_topic_rrf`，两者都重新执行编码和评分，不跨模式复用旧logits。全部排名尝试完成后才打开dev qrels。结果保留原top5正文、完整候选IDs/分数、query来源trace及失败；摘要单列每模式预期/成功/失败任务数，不把失败当成功空排名。固定顺序存在热效应，时长仅作描述。

```bash
python runs/run-current-topic-fusion-retrieval.py \
  --output runs/cloud-dev-current-topic-fusion-v1 \
  --resource-note 'operator-recorded co-resident server/model state'
```

相同模型配置也可单独运行标准离线入口：

```bash
python -m bench.retrieval_eval evaluate \
  --input data/mtrag/RAG.jsonl --split-manifest data/mtrag/split_manifest.json --split dev \
  --sparse-index data/cloud-bm25s --dense-index data/cloud-bge-m3 \
  --encoder-path models/bge-m3 --reranker-path models/bge-reranker-v2-m3 \
  --device cuda:0 --dtype bfloat16 --batch-size 16 \
  --encoder-max-length 8192 --reranker-max-length 1024 \
  --candidate-k 50 --rerank-k 50 --rrf-k 60 \
  --query-modes user_topic --query-fusion current_topic_rrf \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/cloud-dev-current-topic-fusion-standalone
```

首个真实对照固定equal-RRF，不因dev覆盖选择权重。quota仍为CPU候选诊断，尚未接在线。候选保留与排序query是两项独立因素：原query保住其有利定义段，不保证topic评分会把它排入前五，必须保留这一可能失败。已有 `LocalBGE.score`、原dense索引的`by_id`和原始passage直接复用，无需重建索引或扩大重排预算。

如果双路池恢复覆盖却仍排不进前五，再比较评分query策略或有来源的澄清，而非不断增加段数。更复杂的多步模型可作为后续路线，但先明确其训练来源、步骤成本与停止条件，不把ICR的gold轨迹筛选或Multi-Step的训练奖励当在线信号。完整P3仍需保持当前问题、完整QA、原正文、生成预算与资源对等；记录产品适用性、条件/版本和新增错误，不能用R@50恢复代替答案验证。
