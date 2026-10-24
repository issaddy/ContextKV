# 既有矩阵的自有服务器入口

`bench.b6_joint_sweep --matrix` 在不重写既有 matrix/base config 的前提下，复用现有 owned-server 执行器运行 `dev`、`test` 或 `dev_ablation`。不提供 `--matrix` 时，原 13 格 phase1 构造器和默认 plan 行为保留。本入口没有新增规划策略、质量判据或调度器，也不自动选择/冻结赢家。

```bash
# 默认仅 plan；读取声明与小型 provenance 文件，不启动服务或加载模型。
RAYON_NUM_THREADS=4 .venv/bin/python -m bench.b6_joint_sweep \
  --matrix configs/declared-development.yaml --stage dev \
  --output runs/declared-development-run

# 冻结测试仍默认 plan；确认后显式追加 --execute。
RAYON_NUM_THREADS=4 .venv/bin/python -m bench.b6_joint_sweep \
  --matrix configs/declared-development.yaml --stage test \
  --frozen runs/declared-development-run/frozen.json \
  --output runs/declared-test-run
```

`--matrix` 必须给独立的新 `--output`，不能同时提供 phase1 的 `--config / --model-path / --input / --split-manifest / --port / --patched-checkout / --stock-checkout` 覆盖。模型、endpoint 和 checkout 来自既有声明；相对资源路径仍相对当前工作目录解析，与其他实验入口一致，声明文件保持原字节。输出仅包含本次计划、实际 server/job 记录和结果，不重新生成 `matrix.yaml` 或 `controller.yaml`。

`test` 与 `dev_ablation` 在加载模型、检查/启动自有服务器之前验证完整 frozen provenance 和非模拟来源，并通过原 `run_matrix(stage=..., frozen=..., executor=...)` 执行。`matrix_jobs` 只取冻结的 winning candidate，保持已声明的重复 ID/seed；不重跑候选搜索。原有 B6 alias 行为保留。source/config/评分文件/重复计划变化会拒绝，不能归一化旧 implementation 来绕过。当前远端 phase1 `0893134` 的运行及后续冻结仍属于它自己的源码快照；本地新入口不是对其追认。

## 明确支持的服务器范围

本入口支持固定 SGLang commit、现有 patched/clean stock 两类 checkout，以及固定 Qwen2.5-7B revision 或显式 `ModelConfig.scope_path` 的已支持本地 Qwen3 scope。BF16、32768 模型上下文、page64、GPU0、FA3、chunk8192、FCFS、max running8、host admission off 保持。HiCache 仍使用 kernel / layer_first / TMA0，大小由既有 `hicache_gb` 明确声明；B0 的 host0 与 session radix 关闭是基线定义例外。未知 profile 字段/模式拒绝，不把声明的 conditioned 或其他 layout 静默替换成 off/default。原13格 phase1 构造器仍只接受 legacy7B/标准贪心，不因本入口扩展自动改为新模型/采样。

一个 profile 必须明确提供现有七个字段：`kind`、`checkout`、`write_policy`、`hicache`、`session_radix`、`gpu_tokens`、`hicache_gb`。另外支持 `mem_fraction_static`：显式 local scope **必须填写**，旧无scope声明缺省仍为0.15；必须为 `(0,1)` 内有限值。新模型不会暗用旧7B的0.15。controller 使用 patched HiCache；native 使用 stock B1 HiCache 或 B0 无 host。同一 checkout 不能同时冒充 patched 与 clean stock。

同一矩阵全部 profile 的 `gpu_tokens`、有效 `mem_fraction_static` 必须一致，且所有启用 HiCache 的 `hicache_gb` 相同（B0 为0的例外保留）。GPU tokens 与各 job 合并后的 engine 配置相同、按64对齐。可以声明16384、32768或其他实际参考值；仍显式使用 `--max-total-tokens`，例如121920只代表“按自然参考容量做受控复现”，**不是无 cap 自然启动**。host slots 继续由真实分配日志和 metrics 测量，不能用请求的 GB 取代实际容量；同矩阵 HiCache 格的实际 host slots 应一致。

共享 prompt、完整 context policy、输出预算、模型路径/revision/namespace均保留，不强改为 Q0 或1024。原实验入口继续检查方法间材料/预算一致性。此处额外检查 controller/native 的实际 endpoint 和模型；实际公共 server info 的模型/tokenizer路径、revision、dtype、缓存 flags，以及研究 fingerprint 必须匹配。native manifest 仍只在对应真实服务器验证后生成；不接受预填静态 manifest 替代。

矩阵 `seed` 保持原 `run_sessions` 的工作量/到达顺序种子含义，显式 repetition 与 seed 分离。此支持范围内 SGLang 启动 RNG seed 固定42并实测记录；不把工作量 seed 默默改作服务器采样参数。一次运行仍只有一个自有服务器进程，方法之间串行；GPU/driver/CUDA/torch和实际容量检查不能用 cache hash代替。

## 成本、消融与完整分母

每个 job 先合并 base、方法和冻结候选，再判断实际策略：

- `cost` 必须绑定本容量 artifact，即使 `shadow=true` 最后执行固定 fallback，也要验证用于影子预测的成本模型。
- `warm_idle` 需要 quote，不因此要求时间拟合；`stable_layout`、`fixed_period` 等直接复用原控制器。固定策略已有但不参与决策的 artifact 不被误当作成本准入条件。
- 已知缺文件/容量不符，在启动前拒绝；实际 cache/fingerprint scope 不符，保存启动证据并在任何生成前拒绝该 job。空或无 hardware timing samples 的伪绑定 artifact 也不能使 Ours 静默全回退。加载采用该 job 的真实输出预算、admission estimator、uncertainty和host/decode/control消融设置。

这些检查证明 artifact 配置匹配，不证明所有实际请求都在已测支持域。运行中的 sparse-domain fallback、等待估计限制和质量不确定性仍由原 planner/events 保留，不能宣称已全部校准。

`ablation_of` 默认继承父方法 server_profile，与冻结参数继承一致；显式声明相同 profile 可接受，声明不同 profile 拒绝，避免 noC 同时改变硬件。要研究资源变化，须另建明确配置，不能伪装成单开关匹配消融。

完整任务集合来自原 split/domain/limit_sessions：dev与dev_ablation取dev，test取test。只有 phase1 构造器保留5会话/41轮要求；通用入口不硬编码这一计数。每个 job 按完整任务ID、成功/unfinished核验，保留失败记录并清理自己的进程，然后继续其他 job；有失败 CLI 非零。当前是完整有限 cohort 入口，不支持 `arrival_rate` 与 `arrival_window` 同时声明的截窗开环调度，以免把未入窗会话误计为生成失败。完整到达的负载、think time和已有超时设置仍照原 runner执行。

## 评分产物与责任边界

通用 dev 且 `require_quality_evidence=true` 时，输出现有v2 `quality-evidence.template.json` 和 `quality-templates/block-*.json`。每块 methods以实际job_id为键，包含全部候选/重复/失败，reference只引用同块唯一真实参照。没有唯一参照时保留空reference及说明，不自行选择最快参照。评分与frozen evaluator留空，待真实独立评分补齐；不生成资格布尔值。

`test` 与 `dev_ablation` 不输出冒充dev冻结的模板。尤其ablation-only通常没有同次父run，必须随后绑定真实匹配父来源；本入口不补造参照或质量通过。完整 job/原始 audit/模型与服务器记录仍保存，heldout评分另走原 `quality_acceptance(scope=heldout)` 协议。

每个job使用新owned进程，保留启动日志、实际runtime前后快照和结束状态。公共skip-generation-warmup约束不等于清除磁盘JIT缓存；原spec显式before-job hook仍由 `execute_job`执行并记录，初始空KV观测发生在该hook之前。父进程tokenizer/BGE资源按现有路径管理，不新增研究排名缓存。P3共同retriever_config透传原生/controller入口，retrieval排队/执行仍在pipeline计时内。

本工具不停止/恢复外部大服务。操作者仍负责外层owned supervisor、GPU独占及原服务恢复；这里仅清理自己的process groups。默认plan、CPU fake-owned合同或一次硬件完成均不构成性能或质量胜出证明。


## 显式本地模型与共同采样

`--matrix` 读取已有 base config 的 `model.scope_path` 和 typed `sampling`，与[正式 native B0/B1 入口](native_model_scope.md)共用 `create_model_template`、`LocalModelScope`、原生 client。没有另一套模型发现或采样框架。当前支持范围依然是 scope 明示的 dense Qwen3、无 SWA/RoPE 扩展、TP/DP/PP1；32768 窗口与原执行器调度约束未泛化。controller/native 或候选不能各自换模型/采样/输出预算；标准贪心显式写法与 None 仍等价。

scope 必须能在执行机器上验证完整本地快照。启动环境先移除 shell 继承的 `CONTEXTKV_MODEL_SCOPE_PATH`，再按共享声明设置；模型路径和 fingerprint 同样来自共享配置。现有 `launch_sglang.sh` 已支持这个路径，因此无需修改 shell 或拼造 `--revision`：local scope identity 仅用于 fingerprint/manifest，实际 HF loading revision 为 None。两种引擎实际 model/tokenizer 路径、revision、dtype、无 quantization/JSON override/dummy、PP1均在生成前核对；patched engine 还须报告同一 scope identity/path。

新模型资源由操作者在 profile 明确选择。例如已有 `hicache_gb: 8` 和新增 `mem_fraction_static: 0.8` 都是请求参数，**不是推荐最优值或可用显存保证**。计划不读取 GPU；真实启动失败会保留日志并清理自己的进程。启动后检查 actual memory fraction、GPU token pool、host GB，并由真实 host allocation log得到 host tokens；32B的KV字节换算使用 config显式 head_dim128，不把hidden5120/heads64=80误作head dimension。同一矩阵的启用host格还核实际host slots一致。

stock B0/B1 的 actual native manifest 在服务验证后才生成：保留 `model.revision:null`，另记 scope identity/kind/record、真实共同 sampling recipe 和实际池/硬件。旧无scope/无sampling的 native manifest保持原字段与7B身份。研究controller和native都由原`execute_job`消费同一共享配置；通用非owned executor仍不能绕过 `verified_server` 合同直接把不同profile发给一个现有服务。

质量模板也从共享模型生成，不再固定写7B：Qwen3模板的 `model` 是canonical path，`revision` 是实际 `local-snapshot:…` 身份，同时明确 `identity_kind=local_snapshot_metadata`、`hf_revision=null`、`weight_content_verified=false`、`base_family=Qwen3`。这里 quality 格式中的 revision 是既有 opaque identity 槽位，**不声称HF revision已知**。现有quality parser接受该格式，继续区分同模型/相关family/独立声明；不改评分规则、不添quality通过布尔值，也不把声明当权重内容证明。

固定策略可以立即走此入口而不要求旧cost表；cost/shadow cost路径仍须真实同fingerprint/cache/sampling scope和hardware点。旧7B或不同recipe artifact不会因为GPU token cap相同而获得新模型校准资格；匹配失败在生成前拒绝，不暗中全回退。任何实际运行、评分和freeze必须绑定本次实现与声明，不能重写历史13格或跨模型沿用其资格。

本次本地CPU合同覆盖新scope真实factory→owned参数→实际runtime→native validator→quality template、显式head128/实际池、B0/B1 flags、继承环境清理、资源/模型/recipe/cost不匹配零生成、失败cleanup和旧默认；没有新增GPU运行。本入口是否在某一真实资源组合运行成功，仍须后续单独的最小功能验收，不能用既有Qwen3 scope/probe或静态B0检查追认整个新owned矩阵。

随后已完成[8B最小真机验收](owned_local_scope_validation.md)：stock B1与patched Controller各两轮，共4次真实生成全部自然stop，Controller实际R→A，容量/身份/历史与恢复均留证。它关闭此配置的新入口功能缺口，不扩大到32B/P3/成本拟合、质量或性能结论。
