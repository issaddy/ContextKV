# 有限联合 B6 开发执行器：phase 1

`python -m bench.b6_joint_sweep` 将文本整理策略与真实 HiCache 写策略组成预先声明的开发矩阵。默认只打印计划，不加载权重、不连接引擎、不创建结果目录；`--execute` 才在当前机器逐格启动、验证和清理自有服务器。它复用 `native_policy_sweep` 的启动/原生容量证据、`controller_capacity_sweep` 的研究引擎配置检查及 `experiments.run_matrix` 的串行执行/结果绑定，没有新增线上调度或成本模型。

本阶段是 **Cloud dev / P2 的固定配置筛查**，尚不是完整 B6 或 heldout 结论。P2 使用固定检索材料和本臂真实生成的完整历史 QA，不调用 retriever；后续 P3 共同 A 计划采用低成本 `user_topic` 单路，不能把该检索策略计入本轮 P2。

| 文本/原生家族 | 真正启动的写策略 | 格数 |
| --- | --- | ---: |
| Rebuild | write_through / write_through_selective / write_back | 3 |
| Fixed4096 | 同上 | 3 |
| Redundancy40 | 同上 | 3 |
| stock B1 | 同上 | 3 |
| stock B0 参照 | 禁用 HiCache 和 session radix，保留原生普通 radix | 1 |

合计 13 格，每格完整 5 会话 / 41 轮，即计划 533 轮。`Redundancy40` 使用已有 evidence body span 的 tokenizer 长度比例，不把 prompt 长度差视为正文冗余。12 个 `b6_eligible` 标记仅表示预声明候选资格，**不是质量通过**；B0 是共同质量/延迟参照，不参与 B6 候选选择。每个家族显式枚举三个写策略；`equal_budget=1` 表示每个已命名固定格只有一个参数候选，不意味着家族搜索次数相等。

共同设置为 GPU KV 16384 tokens、host 请求 16 GB、page 64、固定 Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`、BF16、模型上下文 32768、Q0、输出上限 1024、正常 EOS、c4、seed 42。三个控制器家族均开启旧 epoch retirement，host admission 均实际关闭；shadow/在线标定和成本/输出长度 artifact 均关闭。固定策略通过原有路径选择动作，不请求成本 quote。

每格是新的服务器进程、空 KV、跳过 generation warmup，记录为 `fresh_process_empty_KV_no_generation_warmup`。父控制器进程、磁盘 Triton/torch 扩展缓存等仍可跨格保留，因此不声称完全相同的 JIT 冷态。只实现第一重复；计划保存今后三次整格顺序的轮换，但未把它们执行、合并，或用不同 seed 冒充重复。一次有序开发运行不足以证明性能显著性；P2 历史和实际输入也可能随输出/策略分叉。

## 执行

在对应 Linux/CUDA 环境激活已有 venv，准备固定 patched / clean stock 两份源码和现有模型文件。先检查计划：

```bash
RAYON_NUM_THREADS=4 .venv/bin/python -m bench.b6_joint_sweep \
  --model-path models/Qwen2.5-7B-Instruct \
  --patched-checkout vendor/sglang --stock-checkout vendor/sglang-stock \
  --output runs/b6-joint-dev-phase1 > runs/b6-joint-plan.json
```

确认外层操作者已停自己拥有的现有服务，且 GPU 0 没有并行实验后，运行同一命令并追加 `--execute`（不要把标准输出重定向到尚未创建的结果目录内部）。父命令必须显式 `RAYON_NUM_THREADS=4`；只设服务器环境不会配置已初始化的客户端 tokenizer。

本执行器 **不停止或恢复已有大服务**。端口占用会拒绝当前格；外层 owner/supervisor 负责原服务的停止及最终恢复。执行器的 `finally` 只清理自己新启动的 process group。单格启动、运行或结果不完整会保留 `server.json`、已有原始结果和错误，其余格继续；整个 CLI 在有失败时非零退出，不生成 B6 结论。

实际服务器必须通过固定 commit/checkout 检查及公共 `/get_server_info` 核验：FA3、chunk 8192、FCFS、max running 8、TP/DP 1、page、上下文、GPU token cap、seed 和 skip warmup。运行时另记录 GPU UUID/型号/显存与 driver/CUDA/torch，矩阵内需一致。GPU 容量取原生 scheduler 观测值；host slot 取真实启动分配日志，字节数用固定模型 KV 几何计算，不将 `16 GB` 参数当作实际 allocator 字节数。HiCache 各格实际 host slots 必须相同；B0 要求观察到 HiCache 关闭且无 host 分配日志，host 为 0。

研究格还核实际 cache 配置、host admission off 与初始 unique GPU/host KV 为 0。stock 格的 native manifest 只在对应服务器成功启动验证后生成并交给 native baseline；全部 native 格保存 prompt IDs，支持后续答案/输入绑定。`server_profile` 的普通 `experiments tune/test` 调用在生成前拒绝，防止三个写策略被悄悄发往同一个现有服务。

主要产物：

- `controller.yaml`、`matrix.yaml`、`plan.json`：完整共同配置、13 格及源码检查结果。
- `servers/<job>/`：自有 PID、启动日志、实际 runtime 前后快照、研究 metrics 或原生 metrics、完成/失败记录，stock 格另有实际 native manifest。
- `matrix-results/results.jsonl` 及各格原始 benchmark/native、audit、manifest/summary：失败不删行，成功格保留实际输入/生成历史/输出。
- `completion.json` 与 `quality-evidence.template.json`：后者只有真实文件引用，评分和 evaluator protocol 均留空，没有伪造质量资格。

cache configuration hash 只证明配置范围匹配，不证明 GPU/driver 等所有环境相同；本轮固定策略不借用大池或其他写策略的成本标定。

## 离线质量与冻结

`require_quality_evidence=true` 从预声明开始绑定；reference=B0、epsilon=0.01、13 个方法/输入/划分/生成配置/实现源码都不能在生成后悄悄替换。后补的只有离线 `tuning.quality_evidence` 文件引用与评分，不从 generation provenance 剥离任何生成或协议约束。

生成后可复制模板为真实 quality acceptance 配置，使用完整 cohort 的五维答案绑定评分，并按 [quality_acceptance.md](quality_acceptance.md) 保存独立 evaluator 协议。当前入口接受人工盲审或已声明独立模型，相关家族自评仅诊断；不接受手填 `quality_eligible` 布尔值。冻结边界核每格实际 run/audit/controller manifest/owned server record 的路径和完成时字节绑定，并调用已有 `bench.quality_acceptance.evaluate` 核对输入/输出、缺分、划分、协议及五维非劣。完整 dev 也不等于最终人工质量通过。

把实际评分配置的路径作为唯一后添字段写入已生成 `matrix.yaml`：

```yaml
tuning:
  equal_budget: 1
  require_quality_evidence: true
  quality_constraints: {reference: B0, epsilon: 0.01}
  quality_evidence: /absolute/path/quality-evidence.json
```

然后在同一生成源码与路径环境执行：

```bash
.venv/bin/python -m bench.experiments freeze \
  --config runs/b6-joint-dev-phase1/matrix.yaml \
  --results runs/b6-joint-dev-phase1/matrix-results/results.jsonl \
  --output runs/b6-joint-dev-phase1/frozen.json
```

有真实但未完成/不合格评分时保存 `frozen.readiness.json`，非零退出且不产出 frozen；路径/协议不符直接拒绝。所有 13 格及完整任务分母保留，不只给最快格补评分。符合本开发 evaluator 的候选才进入现有时延选择；目标是每格完整 turn 时延均值（`max(t_response_done,t_commit)-arrival`），不是质量胜率。评分文件与协议在最终 freeze 一次绑定；无逐请求新 hash。

本 driver **只有 dev owned-server 入口**。其动态 native manifest 不能由普通 `experiments test` 自动启动重现，当前不提供 heldout 执行器；不能把 dev freeze 解释为通用 test 命令可直接运行。下一阶段须复用 owned executor 做受控 heldout 接线，或先声明静态服务器 manifest 方案并重新完成其 dev 冻结。

phase 1 结果不能拼入后来扩大 B4/B5/Ours/P3/消融/自然容量的 spec 再补 provenance。后续需要新的自足预声明矩阵、少量共同 anchors，以及与相应 dev 覆盖一致的最终 test。成本型 Ours 可先用已有匹配 write_through/16K targeted-stage + 显式 native_budget 与同写简单策略比较，再比 phase 1 最佳完整配置；若选 write_back/selective 或 conditioned host admission，须实际重标定该 cache 配置，不复制改 hash 或借 legacy 模式绕过。

本地验收覆盖 13 格计划、fake owned process 生命周期/真实配置传递/失败继续与非零退出、B0 启动 flags、动态 native manifest、错误 generic executor 拦截，以及真实文件形式的质量正反例。这些 CPU 合同测试不构成 GPU 13 格性能或质量结果。

## 后续矩阵的显式重复与分组质量文件

后续 `bench.experiments` 支持显式 `repetitions: [1, 2]`，与 `seeds: [42]` 分开：每个系统重复仍使用声明的 seed，job ID 增加 `-r1` / `-r2`，结果及实际 raw `experiment_protocol` 保存 repetition；before-job hook 还接收 `CONTEXTKV_JOB_REPETITION`。重复按列表顺序外层展开，每次内部保持原方法/候选顺序，本接点不自动轮换或声称顺序无偏。未配置 repetition 时，旧 job ID、结果字段和执行顺序保持不变。时延配对只匹配相同 protocol / concurrency / seed / repetition 的参照。

这是通用矩阵入口能力，**尚未扩展本页 phase1 owned driver 的单重复计划，也未增加 owned heldout 执行入口**。多候选或多 workload/repetition 的质量冻结使用下列新索引文件；原单格 quality acceptance 配置继续可用。

```json
{
  "schema_version": 2,
  "blocks": [
    {"protocol": "P2", "concurrency": 4, "seed": 42, "repetition": 1,
     "quality_config": "/absolute/path/quality-P2-c4-s42-r1.json"},
    {"protocol": "P2", "concurrency": 4, "seed": 42, "repetition": 2,
     "quality_config": "/absolute/path/quality-P2-c4-s42-r2.json"}
  ]
}
```

矩阵仍用 `tuning.quality_evidence` 指向该索引。每块 `quality_config` 是原 `bench.quality_acceptance` 格式，但 `methods` 的键必须为该块全部真实 **job_id**，例如 `B5-c0-P2-n4-s42-r1`、`B5-c1-P2-n4-s42-r1`、`B0-c0-P2-n4-s42-r1`；`reference` 必须是该块预声明 reference method 的唯一实际 job ID。参照自身若有多个 candidate，不猜选第一个或最快者，直接拒绝。没有显式 repetitions 的矩阵使用 block `repetition=1`，对应 job ID 仍无 `-r1` 后缀。

每块分别保存实际 run/audit/scores/controller manifest/owned server record 引用及固定 evaluator protocol，按原格式评分和核绑定。评分合并文件如使用 `score_method`，应以实际 job ID 区分各次重复；不能按相同 task_id 覆盖多次生成。所有块必须使用相同的完整 evaluator 声明、五维定义和 epsilon，不能某次重复换宽松 rubric。每块允许不同的 methods/文件路径，因为它们对应不同的真实执行。

索引 blocks 与预声明矩阵的 `(protocol, concurrency, seed, repetition)` 集合必须完全相等，每块 methods 与真实 job 集合也完全相等；缺块、重复块、漏 job、错配重复的参照/文件均拒绝。生成失败或 unfinished 的 job 仍在 block 报告中，该块标记 `generation_incomplete`，不拿其余完整块做交集。全部评分文件和协议在最终 freeze 边界进入现有 provenance 绑定，不添加请求路径 hash。

资格按 `(method, candidate_index)` 在**全部预声明块上取 AND**。报告保留每块原有 session-cluster 区间，不把多次重复当作新增独立会话，也不输出合并重复的总体 CI。证据不齐与某候选完整评分后未通过是两种不同状态；前者阻止整个冻结，后者可使该候选落选。延续原行为：某个需要 tuning 的方法若没有任何合格完整候选，整体开发选择仍拒绝，不悄悄删除这个方法。所有块 AND 是预声明的开发准入规则，不是新的统计保证。

格式兼容不代表跨实现结果兼容。当前 phase1 远端 `0893134` 的结果若以后评分/冻结，仍须在该源码快照下处理；本地新增 repetition/block 实现改变了 implementation manifest，不能给旧 13 格补身份或改标成同源新实验。下一轮必须保留新矩阵、新实现及其完整原始执行记录。
