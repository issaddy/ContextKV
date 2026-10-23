# 数据、负载与评测验证说明

本项目已经验证固定 MTRAG 文件读取、标签隔离、会话级划分和真实 BM25S 检索，并在 H200 上完成固定模型的 P1/P2/P3 集成、原生 B0/B1 运行与有限标定。完整 Cloud P3 开发集的 qrels 覆盖 39/41 轮，recall@5 为 0.2094、nDCG@5 为 0.1671；正式回答质量与优于强基线的结论仍未建立。mock 输出只用于协议和状态一致性验证。实际证据见[验证记录](verification.md)与[共享提示词开发验证](shared_prompt_validation.md)。

## 安装与已验证数据

在项目根目录运行：

```bash
uv sync --extra retrieval
uv run python scripts/fetch_mtrag.py
uv run pytest tests/test_data.py tests/test_metrics.py -q
```

下载地址固定为 IBM/mt-rag-benchmark 的提交 `53fef6f5917c7e10c436f697e69a955dbc795d81` 中 `mtrag-human/generation_tasks/RAG.jsonl`，而非随时变化的主分支。默认文件写入被 git 忽略的 `data/mtrag/`。已有文件不会被下载覆盖；要仅复算已有文件，运行：

```bash
uv run python scripts/fetch_mtrag.py --offline
```

本次实际下载文件为 10,195,362 字节，SHA256 为：

```text
5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873
```

脚本生成 `profile.json` 和 `split_manifest.json`。若已有结果与本次生成内容不同，脚本拒绝覆盖；应先核对原始文件 hash、固定提交和统计规则。不要修改统计规则使其迎合下表。

| 范围 | 会话 | 任务 | 最大轮次 | 排除首轮后当前块数 | 上轮重复块 | 任一历史重复块 | 非相邻回访块 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全域 | 110 | 842 | 12 | 3660 | 657 | 951 | 294 |
| Cloud | 26 | 205 | 12 | 895 | 188 | 277 | 89 |

每任务恰好包含五个不同证据块，身份定义为 `(document_id, SHA256(text))`。全域 1053 个原始 context 对象带有 `reference` 标注。以上数值是原始数据统计，不能解释为 token 重复率、KV 命中率或 GPU 性能。

会话按域分层，以 `seed=42`、约 20% dev / 80% test 划分。会话 ID 已冻结在 `configs/mtrag_split_manifest.json`，相同会话不会跨集合。当前划分的检索材料中 dev 有 605 个文档 ID、test 有 2553 个，交集为 37 个；这不表示主题隔离。尚未运行主题聚类，`topic_overlap` 为 `null`。

## 在线协议边界

| 协议 | 既往回答 | 当前证据 | 当前实现允许的结论 |
|---|---|---|---|
| P1 | 当前任务官方 input 中的参考历史 | 官方固定 top5 | 与公共任务对齐的逐轮输入、受控成本 |
| P2 | 当前方法已成功提交的真实输出 | 官方固定 top5 | 固定证据下的闭环上下文和缓存机制 |
| P3 | 当前方法已成功提交的真实输出 | 在线 BM25S top5 | 真实检索的脚本化多轮服务 |

P2 的预检索结果可能依赖参考历史，不能据此称为完整在线 RAG。P2/P3 的未来用户问题仍来自固定脚本，不随模型答案变化；正式质量评审需要保留该方法自身历史，并单列脚本预设与自身历史不一致的样本。

`load_rows()` 仅用于离线基础设施。服务 handler 只能接收 `OnlineTurn`，不接收原始行或未来轮次列表。在线对象按字段白名单重新创建：当前问题、允许的先前问答、任务身份以及当前证据的 `document_id/source/score/text/title`。`targets/reference/Answerability/Question Type/metadata` 等标签不会进入该对象。P3 甚至不复制官方当前 contexts。

`OnlineSession` 在当前轮成功提交后才前进；失败时不提交虚构答案，不暴露后续轮次。P2/P3 必须提供每个先前成功轮次的完整自身历史；第二轮不能悄悄用空历史或参考答案启动。P1 在控制器中显式同步参考历史，相关分支记录为 `reference_history_sync`。

## 检索配置与原文保留

检索依赖为 `bm25s==0.3.15`、`numpy==2.3.5`。这些依赖按需导入；使用其他版本时检索器拒绝启动，不自动切换实现。

冻结配置：Lucene BM25，`k1=1.5`、`b=0.75`，NumPy backend，英文 stopwords，无 stemming，无 reranker，top5。query 仅由截至当前的最后三个用户问题用换行拼接，不含参考答案或任何方法的生成答案。使用分数 partition 找出 top5 边界，只对入选的五项排序；同分按 passage ID 排序，包含 top5 边界，不依赖随机 tie-breaking。随机及大规模同分测试与完整 lexsort 的集合和次序完全一致。检索耗时包含实际分数计算和选择。

BEIR JSONL 读取器保留原 passage ID 和原文，不重切段、不清洗正文。返回证据绑定语料快照、内容 hash、source URI 和 provenance。索引用 `title + "\n" + text`，该分词副本不替换原始正文。空或只有停用词的 query 返回空结果，不从标签中补材料。

`BM25SRetriever.manifest` 记录语料内容 hash、ID 顺序 hash、CPU 线程环境和检索参数；`save(directory)` 持久化索引、原文及文件 hash，`load(directory)` 使用上游 mmap 加载已有索引，不重新分词或建索引。文件一致性只在加载边界检查一次，查询路径不做正文重复哈希。正式 P3 应使用固定公开原语料；测试夹具只证明接口与状态链路，不代表 Cloud 语料结果。

离线小型真实检索闭环可运行：

```bash
uv run contextkv --state-dir runs/p3-fixture bench tests/fixtures/mtrag_small.jsonl \
  --protocol P3 --domain all --corpus tests/fixtures/corpus_small.jsonl --concurrency 2
```

这里 BM25S 实际在 CPU 上运行，生成引擎仍是 mock。完整 Cloud P3 应传入固定公开 Cloud BEIR passage 文件，并使用经过硬件验证的 SGLang 配置。

## 负载、失败与时间边界

`bench.run_sessions.run_sessions()` 提供：

- 闭环并发：固定活跃会话数，同会话下一轮从上轮完成后等待 think time；不同会话可并发。
- 会话级开环：固定 seed 的 Poisson 新会话到达；会话内部仍按实际已提交历史串行执行。
- 固定 cohort 的排空、单请求超时和 drain 超时；失败、取消、尚未执行的依赖轮次分别保留。

`RunReport` 保存任务状态、会话 makespan、到达/完成时间、未完成轮次、超时会话和 drain 时间。每个 `TurnRunResult` 保存本轮实际在线对象、生成结果以及 `t_retrieval_done`。P3 的时间原点为流水线到达；P1/P2 的时间原点为固定检索输入就绪，不应把这两种 TTFT 直接混成一列。

客户端没有 token 到达观测时，不能由一次完整响应伪造 TTFT、TPOT 或 ITL。`TurnTiming` 对缺失字段保留 `None`；`output_tokens <= 1` 时 TPOT 不定义。goodput 对单 token 回答使用预先指定的 `ttft_only` 或 `exclude` 规则，零 token 回答不凭空算作满足 TPOT。

报告区分窗口内完成吞吐与整个到达 cohort 的最终结果。控制器在生成完成后的会话关闭和退役排空应单列成本与未完成 outbox；不可据此把物理资源已释放当作默认事实。

## 质量评价与完整审计

`bench.quality` 提供官方记录导出、原始证据完整审计、qrels 读取与 Recall/nDCG 计算，以及配对会话 cluster bootstrap。

官方兼容导出使用本轮检索证据，最多 10 个 context。实际可见证据和原始审计材料放在不同文件，包含完整 manifest、真实输入/输出 IDs 与实际历史；超过 10 个的完整证据不能为通过官方 checker 而截断。当前模块不声称已经运行官方全部算法指标、RAGAS、RADBench 或 LLM judge。

qrels 路径应显式指向实际的 `mtrag-human/retrieval_tasks/<domain>/qrels/dev.tsv`。缺少标签的生成任务在结果中列出 task ID，不当作零分，也不无说明地删除。官方评测适配、judge 配置冻结、人工盲审导出和回填均已实现，操作见下文。正式使用的实际 judge 模型、版本及重型官方依赖环境需由实验配置指定；当前未执行收费 judge 或真实质量实验。

质量统计按配对会话重采样，保留会话内各轮的相关性。输入必须在分组前按 task ID 严格对齐，候选与参考的会话/轮次数不一致时拒绝计算。默认统计量为按轮次加权的平均差，亦可选择会话均值。只有差值置信区间下界严格高于 `-epsilon` 才报告 `noninferior`；样本不足或区间跨越边界报告 `inconclusive`。不把差异“不显著”解释为质量相同。

## Profiling 与运行清单

以下命令生成待执行的参数网格，没有生成任何 GPU 数值：

```bash
uv run python -m profiling.prefill --output runs/calibration/prefill_grid.json
uv run python -m profiling.transfers --output runs/calibration/transfer_grid.json
```

输出明确标记 `status=not_run`、`hardware_measurements=null`。冷/暖 cache 初始化、CUDA/server 计时和具体 GPU 引擎探针需要在硬件端实施。Python 接口 `profiling.prefill.collect(cases, prepare=..., probe=...)` 要求 `prepare` 显式清空/预热，`probe` 返回实际 `Measurement`。它拒绝 mock 记录；传输测量仅接受已完成事件与同一 CUDA/server 单调时钟。

获得真实测量 JSONL 后可汇总：

```bash
uv run python -m profiling.prefill \
  --measurements runs/calibration/measured.jsonl \
  --output runs/calibration/summary.json
```

测量包含 phase、真实输入/前缀/输出长度、并发、传输字节、持续时间与 `source=hardware`；联合 restore/prefill 另记录 `host_load_tokens`。应先采集真实证据再标记 hardware。mock 仅可用显式 `--allow-mock` 作格式演练，不能用于性能结论或硬件成本标定。

`profiling.common.run_manifest()` 对未知 GPU、实际 KV pool、host pool 等字段保存 `null`。`validate_manifest()` 列出正式硬件运行仍缺失的字段。不要用示意容量或占位字符串填充来通过校验。项目当前默认 mock 运行清单也明确标记模拟结果、质量未评测、GPU 性能未验证。

容量指标只接受全局唯一 slot 占用采样，不能将多个 epoch 的共享祖先逐个相加。传输按 transfer ID 去重，并分别保留 completed 与 pending 字节。真实 Prefill tokens 必须来自引擎执行计数，不能由 `prompt_tokens - cache_hits` 代替，因为 chunk 和抢占重算会改变实际计算量。

## 原生 B0/B1 强基线

`bench.native_baseline` 直接调用未打研究补丁的 SGLang 原生 `/generate`，不使用 `controller.engine_client` 的报价、epoch prepare、generation guard 或 receipt 协议：

| 基线 | 每轮提示 | 原生请求 | 服务器启动配置 |
|---|---|---|---|
| B0 | 固定系统提示＋完整原始问答＋共同必需证据，完整 REBUILD | 不发送 `session_id` | GPU prefix cache；关闭 session-aware 与 host cache |
| B1 | 与 B0 相同 | 同一业务会话始终发送同一个顶层 `session_id`，提示重建也不换 ID | session-aware 与 HiCache 同时开启 |

两者都调用共享 `build_candidates()` / `EvidencePolicy`，以 `execution=None` 只生成 REBUILD 候选。历史 archive 和稳定 reference map 仍被维护，因此显式回访、版本比较和条件依赖与研究控制器遵循相同规则。返回的文本只有在原生响应的终止原因及真实 `output_ids` 验证通过后才进入自身历史。`abort`、缺少终止原因、缺少实际 token IDs、HTTP/网络失败均停止该会话，后续依赖轮次保留为未完成。

`length` / `max_tokens` 表示引擎已结束生成，但答案因预算被截断；结果显式记录 `truncated=true`，与研究控制器对最大输出预算的处理一致。该状态不代表语义答案完整，正式质量评价必须保留并报告截断。若服务端启用了 `skip_tokenizer_init` 而不能返回真实 text，此基线会拒绝该响应，不自行用不匹配的 tokenizer 伪造文本。

B1 使用原生会话软引用标签，不调用传统会话续写的 `session_params` 或 `/open_session`。每次依然提交完整 `input_ids`。原生 `/close_session` 的 HTTP 成功仅证明 tokenizer 控制层完成了消息 dispatch；结果始终记录 `acknowledgement=not_scheduler_ack`、`physical_free_observed=false`。关闭失败和未发出关闭请求的 session ID 被保留。没有伪造研究协议的 `CompletionReceipt`、退役 ACK 或释放字节数。

启动配置由实验者分别运行的 SGLang 实例决定；选择客户端的 `B0` / `B1` 不会更改服务器启动参数。正式运行前必须提供 `--engine-manifest`，并将来源代码固定到 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。CLI 会读取原生 `/server_info`，旧入口返回 404 时再使用 `/get_server_info`，比对实际可见的配置字段。官方接口并不提供可靠的远程 Git commit 证明，因此清单明确标记 `operator_manifest_not_remote_attestation`，仍需实验者保存代码 checkout、启动命令和环境证据。

仓库提供 `configs/native_b0_manifest.example.json` 与 `configs/native_b1_manifest.example.json`。两者均标记 `example_only=true`，正式验证器会拒绝；应另存硬件运行清单、记录真实测量与源码/启动证据后再改为 `false`，不能只删除示例标记。清单结构如下；`null` 表示待实测/计算的字段，下面这个示意文件会被正式验证器拒绝，不能直接作为实验数据：

```json
{
  "engine": {
    "base_commit": "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b",
    "research_patch_applied": false,
    "tree_backend": "python"
  },
  "model": {
    "name": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "tokenizer_hash": null,
    "chat_template_hash": null,
    "dtype": "bfloat16",
    "kv_dtype": "bfloat16"
  },
  "server": {
    "enable_session_radix_cache": false,
    "enable_hierarchical_cache": false,
    "disable_radix_cache": false,
    "tp_size": 1,
    "dp_size": 1,
    "page_size": 64,
    "context_length": 32768
  },
  "runtime": {
    "gpu_model": null,
    "gpu_memory_bytes": null,
    "driver_cuda_torch_versions": {"driver": null, "cuda": null, "torch": null},
    "actual_kv_pool_tokens": null,
    "actual_host_pool_bytes": 0
  }
}
```

上述模板表达 B0 的 host 关闭状态。B1 必须将两个 enable 字段设为 `true`，并填写真实分配的大于零的 host pool 字节数。tokenizer/template hash 必须与本次固定 Hugging Face 模板一致；客户端仅加载 tokenizer，不下载模型权重。

原生 CLI 默认发送 `stream=true`，与研究控制器采用相同的客户端 token chunk 观测口径。只在累计 `output_ids` 增长时更新首/末 token chunk 时间，重复输出和终态 metadata 不移动末 token 时间；收到完整终态帧及 `[DONE]` 后才提交历史。SSE 断连、缺失终态或缺失 DONE 都停止会话且不提交部分答案。`NativeTurnResponse.timing` 记录同一客户端 monotonic 时钟中的到达、提交、首/末 chunk、响应结束及历史提交时间。chunk 可以含多 token，因此标签为 `chunk`，不声称逐 token ITL。

固定源码的 [`Serving.incremental_streaming_output`](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/arg_groups/fields/serving.py) 默认为 false。原生基线使用该累计模式，若 manifest 或 `/server_info` 显示 incremental=true 会明确拒绝，要求关闭该选项，不将 delta 误当累计。CLI 可用 `--no-stream`，矩阵用 `native.no_stream: true` 关闭流式；此时首 token 时延保持 null。Python 类默认保留非流式以兼容旧调用，显式 `NativeSGLangClient(..., stream=True)` 开启。

填妥真实清单并启动对应服务后，可使用独立入口：

```bash
uv sync --extra tokenizer --extra retrieval
uv run python -m bench.native_baseline data/mtrag/RAG.jsonl \
  --baseline B0 --protocol P2 --domain cloud \
  --engine-manifest runs/hardware/b0-engine.json \
  --split-manifest configs/mtrag_split_manifest.json --split test \
  --base-url http://127.0.0.1:30000 --output runs/native-b0

uv run python -m bench.native_baseline data/mtrag/RAG.jsonl \
  --baseline B1 --protocol P3 --domain cloud \
  --corpus data/mtrag/cloud/corpus.jsonl \
  --engine-manifest runs/hardware/b1-engine.json \
  --split-manifest configs/mtrag_split_manifest.json --split test \
  --base-url http://127.0.0.1:30001 --output runs/native-b1
```

`--local-files-only` 可限制 tokenizer 从本地缓存读取；`--audit-prompt-ids` 请求原生 `return_prompt_token_ids` 并逐项核对完整输入。全量 token 回传会增加开销，正式比较应给所有方法采用相同审计抽样规则。输入文件 hash、split hash、检索 manifest、真实原生 meta_info、HTTP 往返时间、清理时段与关闭 dispatch 状态均写入结果文件。HTTP 往返时间不冒充 GPU Prefill 或端到端 TTFT，原生 `cached_tokens` 也不被推断为实际 Prefill 计算量。

离线验证命令：

```bash
uv run pytest tests/test_native_baseline.py -q
```

测试使用 `httpx.MockTransport` 模拟原生 HTTP 服务，并包含真实 CPU BM25S 检索的六轮 P3 管线。它验证 B0 不发送会话字段、B1 稳定会话 ID、原始问答完整重建、参考历史同步、显式回访恢复、失败不提交、截断标记和 close dispatch 语义。它没有在真实 SGLang/GPU 上完成吞吐或缓存收益验证。

相关接口依据：[固定源码请求结构](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/io_struct.py)、[固定源码 tokenizer 控制层](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/managers/tokenizer_control_mixin.py)、[原生会话缓存说明](https://docs.sglang.io/docs/advanced_features/session_radix_cache)。

## 完整评测与实验入口

### 小型元数据、完整语料与复用索引

```bash
# 默认仅下载固定提交的 evaluator、qrels 与说明，不下载完整 corpus/模型。
uv run python -m adapters.prepare_data --output data/mtrag-official
# 确实准备完整 P3 时显式选择：
uv run python -m adapters.prepare_data --output data/mtrag-official --cloud-corpus
uv run python -m adapters.retrieval build --corpus data/mtrag-official/cloud.jsonl \
  --index data/cloud-bm25s
uv run python -m adapters.retrieval query --index data/cloud-bm25s \
  --question 'How do I configure request timeout?'
```

控制器与原生基线均可传 `--retriever-index data/cloud-bm25s`，复用固定索引。`--evidence-corpus` 可另传带版本、条件依赖、原始 source 的公开元数据；两种客户端调用同一 `CorpusIndex.enrich()`，按 passage ID＋原文定位版本，再加入显式条件闭包。该步骤不覆盖检索正文，不读取答案标签。

### 官方 evaluator 适配与复用边界

`bench.official` 根据固定提交的实际源码构造参数：算法及生成入口为 `-i/-o/-e`，检索入口为 `--input_file/--output_file`。官方生成流程串联 algorithmic、underspecified、IDK、RAGAS、RADBench 等原有方法；本工程不另写同名近似算法。来源：[MTRAG 论文](https://arxiv.org/abs/2501.03468)、[生成 evaluator](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/scripts/evaluation/run_generation_eval.py)、[算法指标](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/scripts/evaluation/run_algorithmic.py)。

固定源码仍有两处工程问题，适配明确处理：检索脚本使用旧目录 `human/retrieval_tasks_convid`，适配创建指向真实 `mtrag-human/retrieval_tasks` 的目录别名；format checker 在部分格式错误时只打印错误而返回 0，适配直接调用其校验函数，根据 errors 设置退出码。都不改变评测计算或标签。六轮夹具输出已实际通过官方 checker，不能据此推断模型质量。

```bash
# 在线运行产生 official-*.jsonl 后，离线显式 join targets/enrichments。
uv run python -m bench.official prepare --input runs/example/official.jsonl \
  --references data/mtrag/RAG.jsonl --output runs/evaluation/labeled.jsonl
uv run python -m bench.official check --checkout data/mtrag-official \
  --input runs/evaluation/labeled.jsonl --output runs/example/official.jsonl
```

join 按 task ID 保留每个方法自己的 `input/contexts/predictions`，记录历史是否与参考历史一致；只复制离线 `targets/enrichments`。它与服务入口完全分离。

官方重型依赖应装在独立环境，使用下载的原始 `requirements.txt` 和 `constraints.txt`（固定版本包含 torch 2.8.0、ragas 0.3.3、evaluate 0.4.3、pytrec_eval 0.5 等）。上游 requirements 的部分包没有精确锁定，不能称为所有传递依赖均已冻结；在目标 Linux 环境解析并保存 lock/实际环境后保持不变。适配每次保存目标 Python 的实际 package/version 清单及源码 revision、参数、stdout/stderr 与退出码，不把本机控制器环境当成官方环境。

```bash
# 使用已准备好的独立 evaluator Python，不隐式安装/启动 GPU 服务。
uv run python -m bench.official algorithmic --checkout data/mtrag-official \
  --python /path/to/eval-env/bin/python --input runs/evaluation/labeled.jsonl \
  --output runs/evaluation/algorithmic.jsonl
uv run python -m bench.official retrieval --checkout data/mtrag-official \
  --python /path/to/eval-env/bin/python --input runs/example/official.jsonl \
  --output runs/evaluation/retrieval.jsonl
uv run python -m bench.official generation --checkout data/mtrag-official \
  --python /path/to/eval-env/bin/python --provider vllm --model SERVED_MODEL \
  --judge-revision PINNED_REVISION --input runs/evaluation/labeled.jsonl \
  --output runs/evaluation/official-generation.jsonl
```

官方 OpenAI 路径专用于 Azure，且使用命令行传 key；当前适配选择其 hf/vllm 入口。通用 HTTP judge 使用下面单独实现的完整证据审核，不将其称为官方 RADBench/RAGAS 复现。该适配记录 revision 是操作者声明，不能证明远程服务加载了对应权重。

### 完整证据 judge、规则审核与人工盲审

`bench.judge` 从完整 audit 读取当前 query、自身历史、回答、全部 visible manifest 与正文，超过十块也不截断。评分维度为 correctness、groundedness、answerability、conditions_versions、citations；标记 `script_history_mismatch` 与不受支持的断言。分数为 `[0,1]` 或不可判定的 null。provider/model/revision/prompt/temperature 随每条结果保存，API key 仅从指定环境变量读取，不写结果。

```bash
# 没有 --config 时只做规则审核，不发网络请求，不产语义质量分。
uv run python -m bench.judge --audit runs/example/audit.jsonl --output runs/evaluation/rules.jsonl
# 先将示例的 model/revision 改成实际固定模型，再显式执行。
uv run python -m bench.judge --audit runs/example/audit.jsonl \
  --config runs/evaluation/judge.json --output runs/evaluation/judged.jsonl
```

配置示例在 `bench/examples/judge-local.json`。实际 HTTP 由 `httpx.AsyncClient` 执行，测试通过 MockTransport 验证报文、十二块完整正文和配置记录；本轮没有调用付费 API。空答案、未知引用、可见正文缺失和截断被规则审核列为 flags；这些启发式不冒充语义正确率。

```bash
# methods.json 为 {"B3":".../audit.jsonl","Ours":".../audit.jsonl"}
uv run python -m bench.blind_review export --methods runs/evaluation/methods.json \
  --output runs/evaluation/blind --seed 42
uv run python -m bench.blind_review import --key runs/evaluation/blind/private-key.json \
  --ratings runs/evaluation/blind/ratings.csv --output runs/evaluation/human.jsonl
```

将 `review.jsonl` 与 `ratings.csv` 交给评审者，独立保管 `private-key.json`，不要把其方法映射一起发出。随机顺序隐藏方法名，保留回答与全部证据用于判定。回填须含 reviewer ID、五项评分或 `NA`、历史不一致标记及备注。多评审者可各自填写；汇总前应显式处理多评审者分歧，配对统计不悄悄删除缺失评分或合并重复评分。

### 统一矩阵、等预算 dev 调参与冻结 B6

`bench.experiments` 复用既有 runner 和两个真实客户端；每个 job 独立数据库/日志目录，job 顺序执行，各 job 内会话按指定并发运行。支持 P1/P2/P3、并发、seed、think time、会话到达率/窗口/超时、各方法配置、原生服务器和消融。

```bash
uv run python -m bench.experiments plan --config bench/examples/matrix-smoke.yaml \
  --output runs/matrix-plan.json
uv run python -m bench.experiments tune --config bench/examples/matrix-smoke.yaml \
  --output runs/matrix-dev
uv run python -m bench.experiments test --config bench/examples/matrix-smoke.yaml \
  --frozen runs/matrix-dev/frozen.json --output runs/matrix-test
```

此示例是 CPU/模拟工程验收：B2=完整 REBUILD＋epoch 退役，A-only=硬预算内 APPEND＋不退役，B3=固定阈值，另含固定周期、B4 ContextPilot 排序/注释适配、B5 warm/idle、Ours 及成本/滞回/退役消融。Ours 无硬件标定时会记录保守回退，不能解释为成本模型收益。`A+B` 与去掉退役共用 `A_B_no_retire`，避免把同配置重复算作独立实验。

每个 `tune:true` 方法必须提供同样的 `tuning.equal_budget` 个候选，每个候选覆盖相同 protocol/concurrency/seed。失败或依赖轮次未完成的候选不能获选；目标为各实验格等权平均的完整轮完成时间。`quality_eligible` 可在冻结前显式传入离线 dev 质量准入表（method → candidate index → boolean）。未给质量准入时结果明确记录 `quality_gate_applied=false`，不声称非劣效。

dev 选择每个方法的候选，并在 `b6_eligible:true` 中选择 B6。配置、输入、split 与实际 base config 一并冻结；test 拒绝变更，且不参与选择。模拟冻结文件标记 `simulation_only=true`，不能拿它选择硬件测试配置。模拟耗时的选优只证明调参链路可运行，重新上 GPU 时需在相同 dev 预算下重新选择。

每次 generation 开始时保存输入、检索/元数据、标定、原生 manifest 和配置的 provenance，离线选择拒绝用同 job ID 的旧配置结果代替新结果。dev 生成后可以离线评审并将准入表写入配置，再运行 `python -m bench.experiments freeze --config MATRIX.yaml --results runs/matrix-dev/results.jsonl --output runs/matrix-dev/quality-frozen.json`；这一步不再次生成答案。仅离线 `quality_eligible` 可以在这两步之间新增，输入或候选变更均需重新运行 dev。质量不合格的固定消融仍可保留用于分析，但不能成为 B6；可调方法须至少有一个合格候选。

匹配消融使用 `ablation_of: Ours`，只在 `overrides` 写要改变的实验开关。它不参加 dev 候选搜索，也不占用父方法的调参预算；test 从父方法 frozen winner 继承完整参数，再应用该开关，避免将默认参数的消融与调优后的完整系统比较。若要在 dev 上审查消融，先冻结父配置，再运行 `python -m bench.experiments ablate-dev --config MATRIX.yaml --frozen FROZEN.json --output runs/matched-dev`；该命令只运行匹配的消融，结果不参与原始选择。父 winner 改变后应重新运行相应消融。随附 smoke 矩阵使用这一方式，初始 dev 是 33 个 job，冻结后的 test 是 39 个 job，另行 matched dev 消融是 15 个 job。

硬件矩阵通过 `kind: native`、`native: {baseline: B0/B1, base_url: ..., engine_manifest: ...}` 加入原生对照。host 三种写策略必须是实际预先启动的服务，不能靠客户端选项假装变更引擎；候选可用 `native` 覆盖其服务器和 manifest。控制器候选可以覆盖 `engine.base_url` 指向各策略服务器，从而把上下文候选与 host 策略组合一起纳入 B6 的等预算搜索。资源/model/tokenizer 必须通过真实 manifest 对齐后再作性能比较。

硬件服务器跨 job 的残余缓存也必须控制。矩阵支持显式 `lifecycle: {before_job: ["python", "scripts/my_dedicated_server_setup.py"], cache_state_protocol: "cold_reset_then_fixed_dev_warmup", warmup_input: "data/warmup.jsonl"}`。这里 argv 直接执行，不经过 shell；脚本可通过 `CONTEXTKV_JOB_ID/JOB_SPLIT/JOB_PROTOCOL/JOB_METHOD`（每项均带 `CONTEXTKV_` 前缀）选择本 job 的专用服务。钩子的输出、退出码及耗时保存到 `before-job.json`，失败时不执行生成。预热材料应预先冻结且不读取未来 test 轮次；该文件的内容随实验清单冻结。

研究引擎的 cold reset 仅用于显式启用 `CONTEXTKV_ENABLE_PROFILING=1` 的空闲专用服务；原生服务可由操作者脚本明确重启。默认不会 flush 共享 GPU 服务。未配置时硬件记录 `cache_state_protocol=unspecified_hardware`，不能凭实验矩阵完成就宣称公平性能；mock 每 job 创建独立实例，记录 `fresh_mock_instance`。同样，存在钩子只证明其执行过，报告仍需结合实际启动/测量证据确认资源和预热条件。

### 自动报告与缺失值

每个矩阵 job 自动写 `summary.json`、`summary.md`；整组写 `results.jsonl` 与可配对的会话级完成时间差置信区间。失败保留，测试输出不回流调参。也可对单个 raw run 追加质量与引擎采样：

```bash
uv run python -m bench.report runs/example/benchmark.json --output runs/evaluation/report.json \
  --quality runs/evaluation/judged.jsonl --reference-quality runs/evaluation/reference-judged.jsonl \
  --reference-run runs/reference/benchmark.json --epsilon 0.01 \
  --occupancy runs/example/occupancy.jsonl --transfers runs/example/transfers.jsonl
```

缺少相应采样时省略其参数。报告覆盖 TTFT/TPOT/完整轮/队列/控制时间分布、throughput/goodput、失败/超时/依赖未完成、截断、实际 Prefill、唯一 slot 峰值与面积、传输、质量各维度与脚本历史冲突。`TurnRunResult.t_arrive` 覆盖控制器到达点，将 P3 检索时间包括进端到端时延。原生非流式客户端无法提供 TTFT 时保持 null，不用 HTTP 总时长代替首 token；没有真实 transfer/slot 采样时也保持 null。质量比较必须具有相同 task coverage，不可把 judge 失败静默丢弃后计算非劣效。

标定中 `first_token` 表示包含排队/host restore/prefill 的联合观测；`timing_scope` 随记录与分桶保留。它不等于纯 CUDA prefill，也不与不同 host 恢复量或不同时钟范围的样本混桶。

raw run 若附 `telemetry.before/after`，报告自动解析 `/context/metrics` 的全局 slot 和 completed transfer 事件。积分只在 server monotonic 的样本首末之间进行，不跨客户端时钟外推。报告保留事件缓冲截断、采样范围与模拟来源；`scheduler_boundaries` 的观测峰值不冒充 allocator 精确瞬时峰值。服务端 TTFT/TPOT 另列在 `server_metrics`，不混入端到端客户端 TTFT。

质量报告检查 task/conversation 覆盖，新 judge 和人工回填记录还携带被评分的原回答，防止把另一方法同 task 的评分误用于本次运行。缺失评分标为 `partially_evaluated` 并列出 task ID；缺失覆盖或模拟评分不能生成质量非劣效结论。未知任务、跨方法重复行和不同回答的评分须先纠正输入文件。

所有分布包含 `count/p90/p95/p99`，p99 是样本插值统计；小样本时必须连同 count 阅读，不能解释为稳定总体尾部估计。`receipt_work` 仅对有执行 receipt 的轮次报告实际 Prefill tokens / prompt tokens，并按业务会话和分支类型汇总 TTFT、完整轮时延、完整重算比例。抢占重算可使该比例超过 1，不截为 1；cold-prefill 定义为实际 GPU/host 命中均为 0 且实际 Prefill 至少达到 prompt 长度。缺失 receipt/命中计数保持无观测，不以理论前缀或 `prompt-cached` 推算。该观测设计借鉴 [RR-Evict v1 对少数会话完整重算与 TTFT 尾部的分析](https://arxiv.org/abs/2609.32278v1)，没有引入或声称复现它的淘汰算法。

### 计划器本地控制开销优化验证

对 252 个 DecodePoint（63 个 context 点 × 4 档并发）、8192-token 输入、输出先验 mean=128/p90=256 的单候选预测做 15 次本机 CPU 测量。旧实现每个输出 token 都重建坐标表/轴并全表排序近邻；现实现一次建立固定 decode 表的几何结构，完整插值单元直接计算，只有稀疏或越界情况才查近邻；区间测量先寻找同并发覆盖点。没有增加请求缓存、近似积分或修改误差/范围定义。

| 测量数据形态 | 旧实现中位数 | 优化后中位数 | 输出 cost/error/risk/support |
|---|---:|---:|---|
| 无区间的 252 个点 | 38.22 ms | 0.58 ms | 相同 |
| 带覆盖区间的 252 个点 | 15.55 ms | 0.46 ms | 相同 |

原始比较在 `runs/cost-profile/comparison.json`，仅证明本地 CPU 控制路径开销改善，不是 GPU 时延收益。`tests/test_decode_performance.py` 使用冻结的旧实现作为数值基准，验证随机稀疏点、密集点、重复坐标、覆盖区间边界/最近中点/同距次序、mock 过滤、越界与完整均值/p90成本的等价性。单 token 输出仍无 decode 项。更换标定表时新建/加载 CostModel；预处理结构与其固定测量点一起生存。

### 标定来源与运行配置绑定

```bash
# 在推理服务器本机、实际服务器 Python 环境中只读采集；不生成/重置缓存。
python -m profiling.runtime_manifest --base-url http://127.0.0.1:30000 --research \
  --source-checkout /path/to/sglang --output runs/hardware/runtime.json
# 将 profiler 的原始 .manifest.json 与测量一起绑定到成本 artifact。
python -m profiling.fit runs/calibration/measured.jsonl \
  --profile-manifest runs/calibration/measured.manifest.json \
  --development-report runs/dev/benchmark.json --output runs/calibration/cost.json
# 同配置补采后合并原始观测，分别绑定每次运行的原始清单。
cat runs/calibration/measured.jsonl runs/calibration/extended.jsonl > runs/calibration/combined.jsonl
python -m profiling.fit runs/calibration/combined.jsonl \
  --profile-manifest runs/calibration/measured.manifest.json \
  --profile-manifest runs/calibration/extended.manifest.json \
  --development-report runs/dev/benchmark.json --output runs/calibration/cost-combined.json
```

`runtime_manifest` 保存采集主机的 GPU 名称、UUID、总显存、驱动、PyTorch/CUDA 编译版本及本地源码 checkout revision；`--skip-torch` 可在无 PyTorch 的环境只采集其余事实。原生服务只读 `/server_info`，研究服务额外只读 `/context/engine/info` 与 `/context/metrics`，保存其真实 pool、layout、write-policy 与 fingerprint。缺失信息保留 null；本机 GPU 信息始终标记 `local_process_host`，不被推断为 URL 对应的远程机器。KV pool token 数和 host pool token 数来自实际 allocator；没有明确的 allocator 字节数据时，不拿用户配置值冒充实际 host 分配字节数。`launch_command` 与已知 credential 字段不直接导出。

fit 的 `measurement_provenance` 记录 measurement/profile manifest/dev report 的原始文件 SHA256，以及各 phase 的真实 prompt/prefix/output/host-load/concurrency/transfer 范围、cache state、时钟范围与样本数；这些范围只描述观测，插值支持仍由具体测量 cell 决定。`--profile-manifest` 可重复：合并运行时，每条 measurement 都必须有 run_id，测量 run_id 集合须与清单一一对应，且所有清单的 fingerprint 与 cache_config_hash 必须存在并完全一致；遗漏、多余、重复的清单或配置冲突都会拒绝。metadata 的 `profile_manifests` 原样保留全部清单，`measurement_provenance.files.profile_manifests` 逐份记录 run_id、原文件路径和 SHA256；`profile_manifest` 与原来的单文件记录继续保存第一份以兼容已有读取端。单份清单的旧数据没有 run_id 时仍标记 `operator_declared_unlinked`；未提供清单时标记 `not_provided`。加载/再次保存保留全部来源。

`CostModel.check_calibration_scope(info)` 供控制器启动或引擎身份变化时调用，比较绑定的共同模型/template/tokenizer fingerprint 与 cache_config_hash，返回 `matching_scope`、`mismatch` 或 `legacy_unbound`，并在 `profile_run_ids` 列出全部来源运行。cache identity 的覆盖范围由引擎清单明确记录；匹配仅表示引擎配置兼容，**不证明 GPU 型号、驱动、拓扑相同，也不是跨硬件验证**。多次运行的原始 runtime_manifest 分别保留，硬件一致性仍需核对其中的实际记录。明确 mismatch 时保留诊断成本，但将预测的 calibrated 标记设为 false，走既有固定策略回退。legacy 未绑定 artifact 不被突然禁止，只记录来源不足；正式实验需附真实硬件清单。所有文件 hash 只发生在离线拟合/实验入口，不加入每请求数据路径。

### 标定误差按阶段与特征单元解释

`validation.by_phase` 分开报告 first-token/prefill 阶段时延（ms）和 decode TPOT（ms/output token），各自包含样本数、支持域覆盖率、全部观测 MAE/MRE 与 `supported_only` 误差。原有顶层 MAE/MRE 保留兼容；包含多个阶段时明确标为 `aggregation=mixed_phase_diagnostic`，不能用这个混合值替代论文中的阶段误差。

每个阶段还提供 `feature_cells_equal_weight`：先平均同一实际预测特征单元内的重复观测，再对单元等权平均，同时报告单元数、支持覆盖与 supported-only 误差。其分组与 holdout 一致：prefill/first-token 不因 output cap 不同成为独立单元；decode 不因先前 prefix/cache state 不同成为独立单元。这样不会让多个缓存状态重复测得的同一个 decode 条件获得额外权重。观测加权和特征单元等权都保留，二者回答不同问题；测量值为零时 MRE 不定义，报告明确给出有效相对误差样本/单元数，绝对误差仍保留。

真实 H200 dev 标定的离线阶段复算保持原有 34 个训练/8 个 holdout feature cell 及全部原始数据、参数不变，顶层 219 条观测/171 条支持的旧汇总完全一致。新分解如下：

| 阶段 | holdout 观测 / feature cell | 观测支持覆盖率 | 观测加权 MAE / MRE | supported-only MAE / MRE |
|---|---:|---:|---:|---:|
| 首 token 联合阶段 | 84 / 5 | 42.9% | 72.41 ms / 63.2% | 55.56 ms / 69.2% |
| Decode TPOT | 135 / 3 | 100% | 0.461 ms/token / 6.8% | 0.461 ms/token / 6.8% |

按 feature cell 等权后，首 token 的覆盖率为 60%、MAE 55.87 ms；decode 的 MAE 为 0.347 ms/token。全局约 78% 的覆盖率被重复 decode 观测抬高，不能代表首 token 预测可靠。首 token 即使处于几何支持域内，误差仍较大；例如 2048 输入、1792 GPU 前缀、并发 1 的 heldout cell 实测均值约 10.54 ms，预测约 24.51 ms。支持域与误差准确性应分别报告。复算结果保存在 `runs/remote-h200-cost/calibration-dev.phase-diagnostic.json`，没有覆盖原始 `cost-dev.json`。

小规模真实配对诊断共有 9 个 pair，全部生成完成，7 个处于标定支持域，其中 5 个的成本差符号与实测一致；动作原因是 7 次滞回、2 次回退，已选动作的平均 regret 为 126.7 ms。这只能用于诊断，不能证明方法有效：三个场景均在早期轮次，滞回直接阻止重建；两个场景的 APPEND/REBUILD 实际输出长度分别为 422/363、303/256，而当前两候选共用同一个历史输出长度先验，因而会漏掉候选条件下的生成长度差异。

离线分项进一步显示：422/363 的冷态 pair 首 token 差仅 1.14 ms、实际 decode 差为 273.91 ms，两候选实测 TPOT 都约 4.648 ms/token，而模型预测 decode 差仅 0.55 ms。303/256 的 GPU-warm pair 实际首 token 差为 +3.15 ms、decode 差为 +225.23 ms，模型预测分别为 −17.33 ms、+9.66 ms。这里既有首 token 估计偏差，也有共享输出长度先验遗漏的明显成本差。第三个场景两候选都为 512 tokens，但终止原因为 `length`，不能把它当作自然输出长度相同的证据。分项保存在 `runs/remote-h200-cost/phase-and-pair-diagnostic.json`，真实输出长度只用于离线诊断。

后续研究可先固定当前参数，扩大阶段、上下文长度、并发、缓存状态和会话年龄的分层采样，再单独评估候选条件的输出长度先验。任何此类先验只能从已完成的 dev 生成训练，不得访问当前测试答案长度；需要独立验证其预测误差及决策收益。本轮没有依据这 9 个 pair 调参、改变预测算法或宣称统计显著提升。

### 有边界的离线模型比较与可选正系数估计

技术方案第 5 节已允许正系数解析近似，并要求留出未见长度检验。随后仅使用现有 1500 条合成长度标定记录比较三种 first-token 估计：原 `(P,U,L,c)` 查表、`(n=P+U,P/n,L/n,c)` 坐标查表、以及非负基函数 `[1,U,UP+U²/2,L]` 与 `(c−1)` 的四项交互。比较不读取三组配对生成的输出长度，不采集新 GPU 数据；保留原 seed=42 holdout，另固定 seed=2026 的五折 whole-feature-cell CV，以及分别留出 2048/4096/8192 输入长度的三个 fold。所有模型共享相同拆分，没有为每个模型单独挑 seed。

| 评价集合 | 原查表 MAE | 归一化坐标 MAE | 非负基函数 MAE |
|---|---:|---:|---:|
| 原 holdout，84 条 first-token 观测 | 72.41 ms | 101.44 ms | 48.78 ms |
| 五折 feature-cell CV，450 条观测 | 71.55 ms | 71.34 ms | 38.52 ms |
| 三折 leave-length-out，450 条观测 | 90.95 ms | 65.40 ms | 38.04 ms |

非负基函数的五折 cell 等权 MAE 为 24.67 ms（原查表 52.74 ms）；观测加权 p95 绝对误差由 243.07 降为 156.43 ms，但最大误差仍为 338.04 ms。单纯归一化坐标没有稳定改善，因此没有加入生产路径。两个端点长度的 leave-length-out 属于外推诊断，不能据较小 MAE 将它们标成支持域。比较记录包含每个 fold 的覆盖定义、MAE/MRE、大误差、局部 CPU 复杂度及误差启发式失效率，保存在 `runs/remote-h200-cost/offline-model-comparison-final.json`。不同支持域定义的覆盖率不能直接当成准确性优劣。

基于该有限证据，工程新增 **显式可选** 的 `positive_basis` first-token 估计；`lookup` 仍为默认，原 artifact 可继续加载。离线拟合复用 **SciPy 1.17.1 `scipy.optimize.nnls`、NumPy 2.3.5**，安装归入 `calibration` 可选依赖。每个训练 feature cell 的均值等权进入八项非负最小二乘，系数以原单位保存；推理只计算八项基函数及系数点积，并查询预处理的原查表支持域，不导入 SciPy/NumPy，不扩大 `(P,U,L,c)` 的已知支持范围。已有无 SciPy 的 `python -S` 加载与推理测试。

```bash
uv sync --extra calibration
uv run python -m profiling.compare_models runs/calibration/measured.jsonl \
  --output runs/calibration/model-comparison.json
uv run python -m profiling.fit runs/calibration/measured.jsonl \
  --prefill-estimator positive_basis \
  --profile-manifest runs/calibration/measured.manifest.json \
  --development-report runs/dev/benchmark.json --output runs/calibration/cost-positive.json
```

初始原型的训练残差式误差量在部分 CV fold 只覆盖约 32%/56% 的实际误差，因此没有用于运行时决策。可选 artifact 改用完整 feature cell 的五折 OOF：每个 heldout cell 取其中 request 的最大绝对误差，再对这些 cell 值计算 P95/P99，运行时使用较保守的 P99。外层验证集不参与其训练子集的内层 OOF。误差来源、seed、fold、cell 数、覆盖和逐 cell 结果均保存；这只是有限样本的保守经验量，不是置信保证。少于 10 个独立 first-token feature cell 时不标为 calibrated，误差至少取当前成本，继续使用固定策略回退。

独立新 artifact `runs/remote-h200-cost/cost-dev-positive-v1.json` 保留原输出先验 433/512，并记录其源 artifact hash，没有改变输出长度模型或覆盖原始 `cost-dev.json`。最终拟合的 30 个 OOF cell 给出 P95=151.17 ms、P99=269.53 ms；原 holdout 支持覆盖仍为 36/84，仅点估计改善，supported-only MAE 从 55.56 降至 37.39 ms。较大的误差门槛会保守地抑制切换。这些结果足以支持增加可审查的选项，尚不能证明在线决策收益、质量非劣效或跨硬件泛化。

2026-10-24 追加 16384 长度标定后，两个原始采集 run 共 2278 条记录，其中 first-token 和 Decode 各 660 条，D2H 774 条、H2D 184 条。合并时保留两份原始 manifest，校对相同模型指纹、cache configuration、host layout、实际池容量与硬件清单；前后 server restart ID 不同，不将启动标识混同配置。输出先验仍为 433/512，仅扩大输入标定范围，没有用新的成对回答拟合。

扩展集合的固定 seed=42 whole-feature-cell 留出中，first-token 支持为 78/132；查表 MAE 58.41 ms、可选正系数拟合 28.82 ms，supported-only 分别为 62.91 / 34.67 ms。Decode MAE 均为 0.292 ms/token。16K 留出部分支持仅 6/60，可选模型 OOF P99 为 472.57 ms；仍有明显覆盖和不确定性限制。新 artifact 与详细来源在 `runs/remote-h200-expanded/`，没有覆盖旧 v2。该批数据的拆分不同于旧集合，不能用不同留出任务的总平均来宣称扩展采样提高了准确率，更不能据此宣称决策或质量收益。

后续独立审查修正了三个校准边界。第一，原查表仅检查独立 P/U 轴，可能把 P=6000、U=4000（总输入 10000）标为支持，即使实际只测过 2048/4096/8192。现在 lookup 与 positive 共用实际总输入 n=P+U 的最小/最大范围约束；越界仍保留诊断点估计与误差，但不允许成本策略接管，没有额外扩大或细分比例包络。第二，`prefill` 与 `restore_prefill` 按运行时同一预测特征规范化 cell key，避免同一个 (P,U,L,c) 因 phase 别名跨越训练/留出集、虚增独立样本数；每个预测器不兼容的 timing_scope 会在离线拟合/比较入口被拒绝。第三，positive spec 记录实际 `fit_sources`；含 mock 的系数在普通 serving 中不再因过滤 mock 查表点而变成 calibrated，诊断点估计仍可读取。旧 positive spec 缺少此字段时，仅在已有上层 `measurement_sources` 明确全为 hardware 时维持校准资格，否则回退。以上不增加请求路径 hash。原 v1 记录保留，比较和 OOF artifact 应以独立 v2 文件重新生成，不覆盖历史结果。
