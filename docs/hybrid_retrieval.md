# 原始段落上的混合检索与重排序实验

此分支直接改善模块 A 的证据相关性与完整性：复用固定 BGE 权重和官方 Transformers 接口，保留 BM25S，比较 dense、RRF 混合与 cross-encoder 重排序。它不改变默认在线检索，也不改变原文、段落边界、证据 ID、生成提示词或缓存策略。先在已有开发集做无需生成的检索实验，再把有效的设置接入独立 P3 闭环实验；开发集可用于选择与改进，最终 test 保持冻结。

## 选择依据与具体复用

当前 Cloud P3 的 BM25 基线提供了可检验的起点；真实答案审查同时发现 Cloudant 问题配到其他产品资料，以及引用可见但不支持当前论断的情况。前者需要改进检索，后者仍需 claim/source 支持检查，不能仅靠 reranker 解决。相关性与事实支持分别评价，见[支持评分适配](evidence_support_proposal.md)。

| 机制 | 官方依据与复用入口 | 本项目适配与限制 |
|---|---|---|
| BGE-M3 dense | [官方模型卡](https://huggingface.co/BAAI/bge-m3)提供多语言、长输入 dense 表征；本次只用 dense 分支。[固定 pooling 配置](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json)指定 CLS | `AutoModel` → CLS → L2 normalization；query 不加 BGE-v1.5 的检索指令。原始 `title + newline + text` 编码一次，1024 维 float32 memmap，精确内积检索。没有调用 M3 的 sparse/ColBERT 分支，不宣称复现其完整 ensemble |
| BM25 + dense 的 RRF | [Cormack 等 SIGIR 2009 原文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)给出按排名倒数融合的公式，避免直接混合不同量纲的分数 | 两路各取 50，等权 `sum(1/(60 + rank))`，ID 确定性破同分。公式为方法复用、本地纯函数实现；不复制第三方未明许可证代码，也不把 RRF 称为本项目创新 |
| BGE-reranker-v2-m3 | [官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)提供 `AutoModelForSequenceClassification` 与 query/passage 成对推理 | 对 RRF 前 50 重新排序，保留前 5；使用 raw logits 排名，不解释成校准的相关概率、支持概率或回答正确率 |
| 多轮 query 表征 | [IBM 固定检索任务说明](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/mtrag-human/retrieval_tasks/README.md)包含最后问题、全部用户问题和 rewrite 设置，其实验说明检索器与问题表征都值得比较 | 实现 current、last3、all_user；第四个可选视角显式读取冻结 P2/P3 actual_history。没有把数据集给出的 rewrite、参考 assistant、qrels 或未来轮次带入检索，也没有以拼接历史冒充学习式 query rewriting |

选择 M3 的原因是可直接加载的 encoder/reranker 配套、长输入与多语言覆盖、官方调用简单，适合在现有 H200 上以较小工程改动测量收益。不是因模型更新年份断言其效果最强；若它的召回或排序收益不足，可以在同一原文、qrels 与查询视角下替换 encoder/reranker，继续比较较新的方法。原论文或模型榜单中的收益不转记为本项目结果。

固定模型与许可：

- `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`，模型卡 MIT。
- `BAAI/bge-reranker-v2-m3@953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，模型卡 Apache-2.0。

运行只接受预先准备的本地目录，`local_files_only=True`，记录预期 revision、路径、device、dtype、软件版本、token cap 和 batch size。revision 来源明确标为 `operator_supplied_local_snapshot`，不伪称检查了全部模型文件的内容哈希。不安装 FlagEmbedding 整套依赖，不改项目 lockfile 或已有 torch 环境；复用官方兼容的 Transformers/torch 接口。

## 已实现接口与实验产物

[`adapters/hybrid_retrieval.py`](../adapters/hybrid_retrieval.py)提供 `LocalBGE`、`DenseIndex`、`reciprocal_rank_fusion`、`search_variants`；[`bench/retrieval_eval.py`](../bench/retrieval_eval.py)是独立 `build/evaluate` 入口。BM25S 仅增加公开 `retrieve(..., top_k=5)` 参数，默认前五、分数、tokenization 和 tie-break 保持不变。

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

## 可证伪假设与后续改进

1. **召回互补。** dense/RRF 的候选池召回若没有高于 BM25，先查产品实体、追问、原始语料 ID、query 视角与相关性标签覆盖，不直接堆更强生成提示词。若召回改善但前五不改善，重点检查排序与重排器。
2. **重排序有效。** 在相同候选池上，reranker 应提高前五相关性；若更差，保留负结果，检查多轮 query 噪声、长度触顶与产品版本混淆，再比较 current/last3/all_user 或有依据的 local rewrite。不得只报最优方法而丢失其他视角。
3. **对话条件有效。** 多轮视角可补省略的实体，也可能累积过时主题；按追问与主题切换查看具体排名、按会话汇总。当前 5 会话适合探索，不把逐轮样本当成 41 个独立会话。
4. **检索收益能转化为问答收益。** 选定设置后，在共享问题、语料、生成模型、提示词、1024 输出预算和资源条件下跑完整 P3；逐 claim 检查支持、版本/命令范围和不可回答行为。qrels 改善并不自动保证答案更正确，资料更丰富也可能让输出更长。
5. **质量增强能与缓存协同。** 在选定检索设置内部再共享证据比较布局/APPEND/REBUILD；同时单列检索增强前后完整系统成本。新增检索、重排、grounding 或修订开销全部计入，不把基线材料不同归因于缓存控制器。

本次已通过原文保持、RRF 同分、真实 BM25 默认兼容、白名单/历史来源、标签后置分析、模型/语料对应、离线 CLI 和 41 轮 cohort 等测试。mock encoder/reranker 只验证数据合同，不能用于报告真实检索收益。真实模型实验完成后再填写各方法/会话结果及错误分析。
