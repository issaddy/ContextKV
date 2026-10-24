# ContextKV

面向多轮文档问答的上下文与 KV 缓存协同优化研究原型，以 [技术方案](llm_qa_context_kv_technical_plan.md) 为起点，持续研究[完整交互的联合优化](docs/joint_optimization_model.md)。项目提供版本化证据、APPEND/REBUILD 候选、成本计划器、SQLite 双账本与 epoch 生命周期、固定 SGLang 源码补丁，以及 MTRAG 数据和实验工具。

**可直接运行 CPU 模拟演示；固定 Qwen2.5-7B 与 SGLang 已在 H200 上通过真实 GPU/host 协议、P1/P2/P3 集成和受控生命周期验证。当前尚未建立正式质量非劣或吞吐优势结论。** 模拟数据、真实观测与限定实验结果均有明确标记，详见 [验证记录](docs/verification.md)。当前投入顺序和阶段交付见[研发策略](docs/research_strategy.md)：优先复用已有证据与小规模诊断，复杂机制须证明整体净收益。

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

用户现有 Qwen3-8B/32B 已通过[显式本地模型配置](docs/model_scope.md)接入同容器的控制器、启动脚本、引擎握手和生命周期探针，并完成[有限H200协议验收](docs/qwen3_scope_gpu_validation.md)。另有[159＋159轮同输入能力参考](docs/qwen3_reference_validation.md)与四域诊断；新模型独立成本标定、质量机制与正式综合实验仍在推进，旧7B正式基线与结果保留。

近期扩展包括[来源支持的历史主题解析](docs/source_grounded_query_resolution.md)、[固定ConvSearch-R1适配](docs/convsearch_validation.md)、[版本化SDK证据包](docs/api_evidence.md)、[定向host/mixed标定](docs/native_budget_targeted_validation.md)和[业务会话感知host备份准入](docs/host_admission_design.md)，均保持可选。[八方法328条检索对照](docs/frozen_query_retrieval_validation.md)区分候选召回与最终重排，保留长改写的退化；[SDK生成对照](docs/api_evidence_validation.md)保留参数局部修正与初始化代码回归；[host准入对照](docs/host_admission_validation.md)记录了保住另一会话前缀的收益和自身重访的代价。开源复用、实现和效果证据分别说明，A/B/C继续根据结果改进。

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

`configs/normal_rag.yaml`、`append.yaml`、`periodic.yaml`、`redundancy.yaml`、`warm_idle.yaml` 和 `cost.yaml` 可直接运行。研究控制器各策略共享证据策略与 epoch 基础设施；`normal_rag` 是 B2 风格布局基线，不等同于关闭 session/HiCache 的原生 B0。

`fixed_threshold` 按输入长度重建，`fixed_period` 按轮次重建，[`redundancy_threshold`](docs/redundancy_baseline.md)按非本轮required的可见证据正文token比例重建，`warm_idle` 使用实际 GPU ready 报价比例门控。正文比例不等于prompt长度差或可释放KV比例。稳定排序/引用与 warm/idle 属于原理适配，不能称为完整复现 ContextPilot 或 ReCAP。

原生 B0/B1 有独立入口 `python -m bench.native_baseline`：B0 不传 session ID，B1 在整个业务会话中保持同一个原生 session ID；二者都不使用研究报价或伪造 receipt。分别在未打研究补丁、按基线启用缓存的服务器上运行，并根据实际启动日志补齐 `configs/native_b0_manifest.example.json` / `native_b1_manifest.example.json` 中的未知字段。示例文件会被验证器拒绝用作正式实验。完整命令与服务器参数见 [原生基线说明](docs/data_and_evaluation.md)。新 `--run-config` 入口可复用控制器的显式本地模型 scope、context 和 sampling，见[同模型基线入口](docs/native_model_scope.md)；预启动服务支持此路径，owned 启动器仍保留旧 7B/greedy 范围。

研究控制器的硬件开发矩阵及专用服务冷态准备见 [硬件 dev 运行](docs/hardware_dev_matrix.md)。示例包含 11 个同预算开发任务及 5 个继承已选配置的消融；原生 B0/B1、host 策略组合和质量准入仍须独立补齐，不能将这一子集命名为完整最强 B6。

默认 `cost` 只有在相应 prefix/suffix/host-load/concurrency 桶存在可信硬件标定时才启用成本切换；否则结果中明确记录 `uncalibrated_cost_fallback`。显式实验选项 `admission_estimator: native_workload` 使用实测执行阶段和解析等待；`native_budget` 进一步将原生容量预留与经验服务长度分开，并按已有请求 ID 关联预算释放。两者单列 `selection_eligible`，不会把未校准等待标成 `calibrated`；配置与边界见[阶段成本模型](docs/stage_cost_model.md)和[原生工作量估计](docs/admission_wait_model.md)。`shadow: true` 仅记录成本建议，执行固定回退。`calibration_path` 可指定真实 Measurement JSONL。`delta_abs`应对应动作间的增量切换成本；两候选共有的控制开销P95不能直接充作重建惩罚。`plan_selected`保留实际Plan的收益、margin与稳定期状态，便于区分预测排序和策略门控，不额外重算或请求报价。

可选 `positive_basis` 复用 SciPy NNLS 离线拟合，推理只需保存的八个系数；默认仍为查表。按完整特征单元留出，超出实测总长度、混入模拟数据或误差样本不足时不标为硬件校准。共享提示词 Q1/Q2 也仅作为实验配置，真实开发对照发现错引后保持原默认；见[成本验证](docs/data_and_evaluation.md)及[提示词验证](docs/shared_prompt_validation.md)。最新论文与固定作者代码的复用分析见[研究审查](docs/research_review_20261024.md)。

可选候选长度模型与条件误差已完成真实控制器集成及按会话 OOF 的联合决策重评分，结果与局限见[长度与决策验证](docs/length_decision_validation.md)。检索增强的模型、构建命令和对照见[BGE 混合检索](docs/hybrid_retrieval.md)，较新方法的原理与复用入口见[后续检索研究](docs/retrieval_research_next.md)。[证据支持工具](docs/evidence_support_proposal.md)直接复用固定 MiniCheck；真实误例已促使引用修改改为保留旧来源的增补，逐项审查见[评分与引用审查](docs/grounding_v2_assistant_audit.md)。这些能力均为可选择的研究路径，不以模型分数替代独立回答质量验收。

进一步验证包含[嵌套增益校准](docs/gain_calibration_validation.md)、[实际 P3 集成](docs/p3_retrieval_integration.md)、[纠错查询](docs/correction_query_view.md)、[新版重排器的负面结果](docs/qwen3_rerank_validation.md)及[16K/32K 原生缓存压力](docs/native_policy_validation.md)。保持逐 token 等价的[候选分词优化](docs/candidate_tokenization_profile.md)已在本地和 Linux 验证，并记录线程预算对 CPU 消耗的影响。

[质量验收入口](docs/quality_acceptance.md)支持多评审聚合、双方实际答案绑定、完整评分覆盖和固定 dev/test 划分核对；已生成三组真实 P3 共 123 份待盲审样本，尚无人工评分。自动模型评分、开发探索和最终人工验收分别标记。

[有限容量控制器入口](docs/controller_capacity_sweep.md)复用现有引擎与矩阵工具，比较六种固定方法并要求每个容量使用自身的标定。原生和控制器运行器均保留失败结果，并在生成失败时返回非零退出码。

16K 独立标定及六方法的 246 轮真实运行已完成；[标定分析](docs/capacity_calibration_validation.md)和[联合验证](docs/controller_capacity_validation.md)定位了容量不确定标记导致的集中回退，以及输出预算与调度等待耦合。这些结果用于继续改进 B/C，尚不构成完整方法优于强基线的结论。

可选[表格关系注释](docs/table_relations.md)保留完整原文，为对象、列、值和单位附加字面 span。候选和原生对照共用 `context.table_relations`；默认关闭。新增 `table_annotation_style: focus_refs` 复用 CRAFT 行文本并按当前 query 选择焦点、逐行附原引用；[12 次冻结历史验证](docs/table_focus_validation.md)在两个开发历史中改善了已知对象/引用错误，[完整 P3 三臂验证](docs/table_focus_p3_validation.md)已完成123轮，保留目标修复及后续条件/产品错误。独立模型的[原生协议](docs/open_judge_plan.md)和[GroUSE 元评测](docs/grouse_meta.md)已实现；[真实首轮结果](docs/open_judge_validation.md)暴露范围漏判与格式失败，当前不能用于宣布质量非劣。 新增[原子声明诊断 CLI](docs/atomic_judge_proposal.md)独立保留限定条件与来源核验结果，已完成两轮真实29例开发诊断，格式完成率提高但仍有量词、数值与来源适用性误判；[验证记录](docs/atomic_judge_validation.md)保留全部失败。可选JSON Schema模式用于约束输出结构，语义可靠性另行验证。

[用户主题查询](docs/user_topic_query_view.md)已接入共享检索入口；[五视图真实开发对照](docs/user_topic_retrieval_validation.md)完成820条结果，最终R@5从current的52.31%到54.87%，同时保留WebSocket问题候选召回下降。[82轮完整问答](docs/user_topic_p3_validation.md)已完成并保留具体改善与新错误；[固定总重排预算的双路融合](docs/query_fusion_research.md)已实现为可选实验；[真实重排](docs/current_topic_fusion_validation.md)恢复了候选池覆盖，最终top5指标持平；[随后82轮P3](docs/current_topic_fusion_p3_validation.md)保留定义改善与后续回答回归，默认保持单路。

[阶段成本闭环验证](docs/stage_workload_validation.md)完成四方法两重复共 328 轮。工作量模型提高了成本选择覆盖，但完整响应仍慢于对照，等待误差和门槛影响已按实际轨迹定位。模型计算的 Linux CPU 优化有单独逐字段等价证据，未混入此前的 GPU 性能数字。随后[优化后代码的门槛对照](docs/stage_gate_validation.md)又完成 328 轮，gate0 相比 gate2 的等会话均值降低 19.88%，仍慢于固定阈值；这个开发结果继续用于纠偏。新增[原生预算与匹配观测](docs/native_observation_validation.md)已在H200 GPU/host受控负载中完成复算和生命周期验收，默认计划器保持原行为，显式 `native_budget` 分支使用实际预算与请求身份，历史 match 仅用于声明清楚的服务时间情景。

新增 [LettuceDetect span 诊断](docs/lettuce_judge.md)和 [Granite Guardian 原生风险诊断](docs/granite_judge.md)均复用固定模型与作者协议，读取完整实际可见证据，保留失败、截断和原输出；二分类结果不能补造五维质量分数。当前剩余验收和投入顺序见[研发策略](docs/research_strategy.md)；[目标审计](docs/goal_acceptance_audit.md)与[扩展设计](docs/ac_extension_priorities.md)保留历史边界和机制依据。

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

[通用自有服务器矩阵入口](docs/b6_owned_matrix.md)支持开发、冻结测试和匹配消融，并按实际写策略启动服务；系统重复与随机种子分开，质量按真实job/block绑定。已有[13格开发筛查](docs/b6_joint_validation.md)完成533轮生成和[2665维独立评分](docs/b6_quality_validation.md)，完整审计通过，但12个候选均未获得五维联合非劣资格，未选出B6。[369轮计时分解](docs/b6_timing_diagnosis.md)将主要TTFT差异定位到首次forward之前；新增[默认关闭的原生准入诊断](docs/native_admission_diagnostic.md)，后续[peer](docs/peer_interference_validation.md)与[共享前缀](docs/shared_prefix_admission_validation.md)的有限真机验证已完成。冻结B6采用赢家别名，避免相同job重复生成；真实repetitions不合并。新入口的CPU验证不能替代正式heldout结果。

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
