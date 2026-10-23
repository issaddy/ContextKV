# ContextKV

面向多轮文档问答的上下文与 KV 缓存协同优化研究原型，按 [技术方案](llm_qa_context_kv_technical_plan.md) 组织。项目提供版本化证据、APPEND/REBUILD 候选、成本计划器、SQLite 双账本与 epoch 生命周期、固定 SGLang 源码补丁，以及 MTRAG 数据和实验工具。

**可直接运行 CPU 模拟演示；固定 Qwen2.5-7B 与 SGLang 已在 H200 上通过真实 GPU/host 协议、P1/P2/P3 集成和受控生命周期验证。当前尚未建立正式质量非劣或吞吐优势结论。** 模拟数据、真实观测与限定实验结果均有明确标记，详见 [验证记录](docs/verification.md)。

## 快速启动

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```bash
uv sync --locked --extra retrieval --extra calibration
uv run contextkv --config configs/demo.yaml demo
uv run --extra retrieval --extra calibration pytest -q
```

演示包含证据重叠、追加、版本变化后的重建和历史回访。输出位于 `runs/demo/`：SQLite 账本、按 hash 保存的证据、JSONL 事件、运行 manifest 和逐轮结果。重复运行会新建演示会话，不覆盖已提交历史。

启动本地 HTTP API：

```bash
uv run contextkv --config configs/demo.yaml serve --port 8000
```

API 文档在 <http://127.0.0.1:8000/docs>，健康检查为 `/health`。默认只监听 localhost。

```bash
curl http://127.0.0.1:8000/sessions/example/turns \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"turn-1","query":"请求超时是多少？","evidence":[{"document_id":"timeout","text":"请求超时为30秒，仅适用于v1。","source_uri":"docs://timeout","version":"v1"}]}'
```

同一会话重复提交相同 `request_id` 会返回已提交结果；内容不同则报冲突。生成响应丢失时保留未决 attempt，查询引擎终态后恢复，禁止盲目重新生成。`POST /sessions/{id}/recover` 用于恢复，`POST /sessions/{id}/close` 用于结束与退役。

## 已实现的工程路径

| 路径 | 内容 |
|---|---|
| `controller/schema.py` | 不可变证据、候选、账本、报价与 receipt 合同 |
| `controller/evidence.py`, `templates.py`, `candidates.py` | 版本/hash 身份、稳定引用、完整正文 span 校验、全量 QA 与纯候选 |
| `controller/cost_model.py`, `output_length.py`, `planner.py` | 测量查表、可选非负拟合、候选长度预测、并发条件误差、滞回与显式回退 |
| `controller/session_store.py`, `service.py`, `epoch.py` | SQLite WAL、CAS、单未决 attempt、同结果恢复、持久 outbox、退役 ACK |
| `controller/mock_engine.py`, `engine_client.py` | CPU 协议模型与研究引擎 HTTP 客户端 |
| `engine_patches/` | 指定 SGLang commit 的源码补丁、原文件 SHA256 和静态验证记录 |
| `adapters/` | MTRAG 白名单、P1/P2/P3 历史门控、BM25S、可选 BGE dense / RRF / reranker |
| `bench/`, `profiling/` | 多会话负载、失败保留、性能汇总、质量区间、MiniCheck 支持评分、引用增补、会话 OOF 与真实标定 |
| `tests/` | 覆盖版本/回访、缓存只读、生命周期/并发/恢复及 P3 集成 |

事务与恢复说明见 [架构文档](docs/architecture.md)，数据和评测细节见 [数据文档](docs/data_and_evaluation.md)。论文与开源方法的原理、复用范围、适用条件及可证伪贡献见 [贡献与实验设计](docs/research_contributions.md) 和 [调研记录](docs/related_work.md)。

## 固定数据与协议

```bash
uv run python scripts/fetch_mtrag.py
uv run --extra retrieval contextkv --config configs/demo.yaml \
  --state-dir runs/cloud-p2 bench data/mtrag/RAG.jsonl \
  --protocol P2 --domain cloud --concurrency 4 \
  --split-manifest configs/mtrag_split_manifest.json --split dev
```

数据固定为 MTRAG `53fef6f5917c7e10c436f697e69a955dbc795d81`，原文件 SHA256 为 `5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873`。已复算 842 个任务/110 个会话，Cloud 205/26；按会话分层的 seed=42 划分已写入 `configs/mtrag_split_manifest.json`。在线请求不包含 targets、reference 或未来轮次。

- P1：官方参考历史与固定证据；每次历史变化显式同步并强制重建。
- P2：各方法自身已提交输出与相同官方固定证据。
- P3：自身已提交输出与真实 BM25S；检索只使用最近三个用户问题。

P3 需要原始 BEIR passage 语料，通过 `--corpus PATH` 提供。最小真实检索管线验证可用内置样本：

```bash
uv run --extra retrieval contextkv --config configs/demo.yaml \
  --state-dir runs/p3-fixture bench tests/fixtures/mtrag_small.jsonl \
  --protocol P3 --domain all --corpus tests/fixtures/corpus_small.jsonl --concurrency 2
```

这条命令运行真实 CPU BM25 检索和模拟生成。不能据此声称完成了 Cloud P3 的真实模型实验。基准输出另外保存官方兼容 JSONL 与完整输入/证据审计 JSONL，关闭和退役排空耗时单列。

## 策略与公平比较

`configs/normal_rag.yaml`、`append.yaml`、`periodic.yaml`、`warm_idle.yaml` 和 `cost.yaml` 可直接运行。研究控制器各策略共享证据策略与 epoch 基础设施；`normal_rag` 是 B2 风格布局基线，不等同于关闭 session/HiCache 的原生 B0。

`fixed_threshold` 按输入长度重建，`fixed_period` 按轮次重建，`warm_idle` 使用实际 GPU ready 报价比例门控。稳定排序/引用与 warm/idle 属于原理适配，不能称为完整复现 ContextPilot 或 ReCAP。

原生 B0/B1 有独立入口 `python -m bench.native_baseline`：B0 不传 session ID，B1 在整个业务会话中保持同一个原生 session ID；二者都不使用研究报价或伪造 receipt。分别在未打研究补丁、按基线启用缓存的服务器上运行，并根据实际启动日志补齐 `configs/native_b0_manifest.example.json` / `native_b1_manifest.example.json` 中的未知字段。示例文件会被验证器拒绝用作正式实验。完整命令与服务器参数见 [原生基线说明](docs/data_and_evaluation.md)。

研究控制器的硬件开发矩阵及专用服务冷态准备见 [硬件 dev 运行](docs/hardware_dev_matrix.md)。示例包含 11 个同预算开发任务及 5 个继承已选配置的消融；原生 B0/B1、host 策略组合和质量准入仍须独立补齐，不能将这一子集命名为完整最强 B6。

`cost` 只有在相应 prefix/suffix/host-load/concurrency 桶存在可信硬件标定时才启用成本切换；否则结果中明确记录 `uncalibrated_cost_fallback`。`shadow: true` 仅记录成本建议，执行固定回退。`calibration_path` 可指定真实 Measurement JSONL。正式启用前还需用本机 profiling 将 `delta_abs` 写为控制开销 P95 的两倍。

可选 `positive_basis` 复用 SciPy NNLS 离线拟合，推理只需保存的八个系数；默认仍为查表。按完整特征单元留出，超出实测总长度、混入模拟数据或误差样本不足时不标为硬件校准。共享提示词 Q1/Q2 也仅作为实验配置，真实开发对照发现错引后保持原默认；见[成本验证](docs/data_and_evaluation.md)及[提示词验证](docs/shared_prompt_validation.md)。最新论文与固定作者代码的复用分析见[研究审查](docs/research_review_20261024.md)。

可选候选长度模型与条件误差已完成真实控制器集成及按会话 OOF 的联合决策重评分，结果与局限见[长度与决策验证](docs/length_decision_validation.md)。检索增强的模型、构建命令和对照见[BGE 混合检索](docs/hybrid_retrieval.md)，较新方法的原理与复用入口见[后续检索研究](docs/retrieval_research_next.md)。[证据支持工具](docs/evidence_support_proposal.md)直接复用固定 MiniCheck；真实误例已促使引用修改改为保留旧来源的增补，逐项审查见[评分与引用审查](docs/grounding_v2_assistant_audit.md)。这些能力均为可选择的研究路径，不以模型分数替代独立回答质量验收。

进一步验证包含[嵌套增益校准](docs/gain_calibration_validation.md)、[实际 P3 集成](docs/p3_retrieval_integration.md)、[纠错查询](docs/correction_query_view.md)、[新版重排器的负面结果](docs/qwen3_rerank_validation.md)及[16K/32K 原生缓存压力](docs/native_policy_validation.md)。保持逐 token 等价的[候选分词优化](docs/candidate_tokenization_profile.md)已在本地和 Linux 验证，并记录线程预算对 CPU 消耗的影响。

## 真实 SGLang 接入

在 Linux GPU 主机使用独立环境安装固定 SGLang 的实际依赖，控制器依赖由本项目 `uv.lock` 管理。主实验固定 Python TreeCore、TP=1、DP=1、FULL attention、BF16 权重与 KV，不启用 L3 或推测解码。完整安装、分阶段 GPU/host 验证和标定命令见 [部署说明](docs/deployment.md)；`bash scripts/bootstrap_sglang.sh` 可准备独立环境、固定源码补丁与模型。

```bash
# 下载完整固定源码并应用补丁；源码准备不安装 CUDA 依赖。
uv run python scripts/prepare_sglang.py --checkout vendor/sglang

# 控制器环境只需加载 tokenizer，不下载模型权重。
uv sync --locked --all-extras
mkdir -p runs
uv run --all-extras contextkv --config configs/sglang.yaml fingerprint > runs/fingerprint.json
```

在 SGLang 独立环境中设置 `CONTEXTKV_MODEL_PATH` 指向固定 Qwen2.5-7B-Instruct revision 的本地模型快照，将 `runs/fingerprint.json` 的完整 JSON 作为 `CONTEXTKV_FINGERPRINT_JSON`，再执行 `bash scripts/launch_sglang.sh`。可通过 `CONTEXTKV_HICACHE_GB` 指定 host 容量；默认 16GB 只是启动示例。无 host 的阶段验证使用 `CONTEXTKV_ENABLE_HICACHE=0`，不会通过 size=0 伪关闭 HiCache。

启动后在控制器环境运行：

```bash
uv run --all-extras contextkv --config configs/sglang.yaml serve
```

能力握手会拒绝未补丁服务器、错误模型/tokenizer/template 与不支持的缓存模式。补丁说明及验证边界见 [引擎说明](engine_patches/README.md)。

macOS/CPU 可单独检查补丁，不安装引擎或模型：

```bash
uv run python scripts/prepare_sglang.py --source-only \
  --checkout vendor/sglang-source-check --report runs/patch-verification.json
```

## 验证与正式实验边界

```bash
make lint
make test
# tokenizer 文件已经缓存后，额外验证真实固定 Qwen 模板：
CONTEXTKV_TEST_HF_TOKENIZER=1 uv run --all-extras pytest tests/test_tokenizer_optional.py -q
```

CI 运行 CPU 测试、lint 与离线演示。验证记录见 [本地验证](docs/verification.md)。生成标定网格：

```bash
uv run python -m profiling.prefill --output runs/prefill-grid.json
uv run python -m profiling.transfers --output runs/transfer-grid.json
uv run python -m profiling.control --config configs/demo.yaml --output runs/control-profile.json
```

Prefill/transfer 网格明确标记 `not_run`，不含伪造测量；`profiling.control` 真实测量本机候选、报价与计划耗时，输出中的 engine_kind 区分模拟接口与 SGLang RPC，模拟值不会自动写入硬件配置。真实测量由 `profiling.sglang_probe` 采集，`profiling.fit` 按独立网格单元留出、报告误差和覆盖率，再冻结开发集成本模型。

研究客户端与原生 B0/B1 默认使用 SSE，记录实际首/末新增 token chunk；非流式或恢复重放不补造客户端 TTFT。服务端联合首 token 阶段、Decode 与 CUDA ACK 传输时间分别保留，不混用客户端和服务端时钟。开发集等预算调参、冻结测试、外部 judge 和人工盲审均有独立入口；最新执行证据与未验证范围以 [验证记录](docs/verification.md) 为准。host 写入准入、第二模型等增强依实测瓶颈决定。
