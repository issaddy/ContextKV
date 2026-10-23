# Qwen3-Reranker 固定候选池对照

这一实验用较新的 Qwen3-Reranker-0.6B 检验 BGE reranker 是否是当前场景合适的基座。它复用已保存的 current/hybrid 前 50 个原始 passage，不重新做 embedding、BM25 或 RRF，不改变正在运行的 P3 主路径。全部任务完成打分后才打开同一份 qrels；默认模型加载只能使用事先准备的本地 snapshot。

## 官方依据与适配

[Qwen3 Embedding 论文 v3](https://arxiv.org/html/2506.05176v3)把 reranking 建模为 instruction/query/document 条件下的 yes/no 判断，训练利用监督数据与模型合并；本文复用公开权重，不复现训练。0.6B 型号适合作为与 BGE 相近规模的较新对照。作者的 reranking 表使用其 embedding 检索的 top-100 候选，不能把那些结果直接代入本项目的 BM25+BGE/RRF top-50。

固定 [`Qwen/Qwen3-Reranker-0.6B@fd9fb1d26c07223ced488065909faf522e29cc7d`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/fd9fb1d26c07223ced488065909faf522e29cc7d/README.md)，model card 标 Apache-2.0。适配沿用官方 Transformers 示例的 instruction 格式、完整 system/user prefix、关闭思考的 assistant suffix、left padding，以及最后位置 no/yes 两个 logits 的归一化。使用 `AutoModelForCausalLM`，没有把它误接到 BGE 分类头，也不调用生成 API。

实现文件为 [`adapters/qwen3_reranker.py`](../adapters/qwen3_reranker.py)。相对官方示例的必要改动是：固定本地加载与设备精度、批量计分、截断前 token 计数，以及通过 [Transformers Qwen3 接口](https://huggingface.co/docs/transformers/model_doc/qwen3)的 `logits_to_keep=1` 只计算末位置的词表 logits，`use_cache=False` 不返回 KV。后者不改变需要的最后位置得分，避免为全部输入位置分配词表输出。未引入额外 vLLM 服务、FlashAttention 安装或新的依赖锁定。

归因与许可副本见 [Qwen3 notice](third_party/QWEN3_RERANKER_NOTICE.md)。固定 snapshot 没有单独 LICENSE/NOTICE，许可依据是其 model card；本仓库不再分发权重。manifest 记录预期 revision 和 `operator_supplied_local_snapshot`，不伪称重哈希核验全部权重。

## 同池、同预算的比较定义

[`bench/rerank_replay.py`](../bench/rerank_replay.py)读取旧 offline run 的 `manifest.json` 与 `retrieval.jsonl`，取 `current/hybrid` 前 50，并要求原 `current/hybrid_rerank` 的 `rerank_candidate_ids` 完全一致。每轮 query、conversation、turn 与源 manifest 核对，cohort 中所有任务都保留。原文从原 BM25 index 的 `passages.jsonl` 读出，使用既有语料标识和该文件摘要检查边界，不加载稀疏索引或向量矩阵。

新旧模型总输入 cap 均为 1024，Qwen 的 prefix、suffix 和 instruction 也占这个预算。不同 tokenizer 和模板意味着覆盖的正文 token/字符可能不同，这不等于相同正文长度。新结果逐 pair 记录截断前/实际 token 数和是否截断；原 BGE run 只有既有触顶统计，不能事后伪造它的精确逐 pair 截断率。Qwen 的 body 先完整 tokenize，再按预算右截断，完整 prefix/suffix 保留，与官方单字符串输入的右侧截断等价。

采用官方默认检索 instruction，不从 qrels 调整 instruction、候选池、模型分数或阈值。后续领域 instruction 可以作为另一个明确开发实验，但需保留这一对照。yes/no 分数用于排名，不能称为本场景已校准的相关性或事实支持概率。

## 运行

```bash
# 只核对并导出冻结池；不加载 torch/模型，不读取 qrels。
python -m bench.rerank_replay \
  --source-run runs/cloud-dev-hybrid-rrf50-rerank1024 \
  --corpus-index data/cloud-bm25s \
  --dry-run --output runs/cloud-dev-qwen3-pools

# 由统一 GPU 操作者在已有生成任务结束后运行。
python -m bench.rerank_replay \
  --source-run runs/cloud-dev-hybrid-rrf50-rerank1024 \
  --corpus-index data/cloud-bm25s \
  --model-path models/Qwen3-Reranker-0.6B \
  --device cuda:0 --dtype bfloat16 --batch-size 8 --max-length 1024 \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/cloud-dev-qwen3-rerank-samepool1024
```

输出 `pools.jsonl` 保存源 query/候选 ID/原 BGE 排名；`reranked.jsonl` 保存新排名、前五原文、逐 pair 长度与新 rerank 墙时；`manifest.json` 保存旧来源路径及摘要、原 manifest、模型/模板/设备/精度、状态。原文件不覆盖、不修改。中途出错保留已写行；逐轮推理失败也单列，分析时保留该轮空排名，不能靠丢掉难例提高均值。

提供 `--qrels` 才生成 `summary.json`：与原 BGE 排名共享 qrels，复用现有 Recall/nDCG@5、候选池和会话汇总。缺标签/无正例仍单列；它不产生答案质量结论，也不进入正式 test 选型。

新耗时仅为 Qwen rerank 阶段；旧 BGE rerank 耗时由其保存的完整阶段减去同次共享 hybrid 阶段得到，两项原时间也保留。它们不是同一时刻、同负载重新测得的成对性能数据，因此本次用于质量/成本探索，不直接宣布 Qwen 更快或端到端更优。选定模型后，才在同一资源和生成条件下做新 P3 对照。

合同测试覆盖完整模板/左 padding/右截断、只取最后位置 no/yes、实际 CausalLM 调用参数、本地加载、候选池与查询一致性、失败不丢弃、源数据不变、qrels 后置和输出不覆盖。模型 fixture 的分数仅验证工程合同。真实权重运行结果另存，未测得收益前不替换默认 BGE 基座。
