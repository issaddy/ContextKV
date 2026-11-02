# 开发冻结、rolling 评测与负载顺序入口

`training.plan_gain_workflow` 先生成实际反事实监督并拟合收益资产。
`bench.rolling_study_workflow` 接收该资产和 C 的实测成本资产，依次执行开发集、冻结、固定测试与可选动态负载，不启动训练。

当前 R0 的实际配置标签是 `qwen3-r0-source-first-windows-32k-nonthinking`，reader 为 GPU 3 的 `http://127.0.0.1:30002`，helper/judge 为 GPU 2 的 `http://127.0.0.1:30001`，冻结 query/history/support 模型驻 GPU 1。标签来自本轮显式配置；模型 checkpoint、GPU UUID、端点、进程 roots、常驻模型和缓存身份仍由实际资产及运行时核对。

在指定容器和项目根目录运行：

```bash
.venv-sglang/bin/python -m bench.rolling_study_workflow \
  --config <本轮实际study配置.yaml> \
  --matrix configs/study_matrix.yaml \
  --gain-asset <plan_gain_workflow生成的plan-gain.json> \
  --execution-profile <C成本工作流>/execution-assets.json \
  --native-assets <C成本工作流>/native/native-assets.json \
  --reader-profile-id qwen3-r0-source-first-windows-32k-nonthinking \
  --output runs/study/rolling-r0-final-v1 \
  --baselines B0 B3 B6 \
  --lambda-time 0,0.000001,0.00001,0.0001 \
  --read-outputs 512,1024,2048 \
  --failure-unit conversation \
  --load-manifest <既存动态goal工作负载的prepare_manifest.json>
```

输出目录必须是新目录。R1 用它自己的实际 study 配置、收益资产、reader profile、native timing 与 execution 资产调用同一入口。入口从 execution 资产读取原生端点和物理进程/GPU范围；不会把 R0 资产改标签给 R1 使用。

`--dev-conversations-per-track N`、`--test-conversations-per-track N` 显式冻结每轨按 conversation ID 排序的前 N 个会话，默认不缩减。已存在的原生分区、MT-RAG 项目会话划分、MultiDoc 开发抽样会先应用，再进行这个显式子集选择。子集写入 `cohorts/` 和 `matrix.json`；缩减结果只代表该预先登记子集。开发选型只使用 `selection.development_weights` 指定的轨道，UN 的 test-only 资产不参与。可用 `--skip-test` 单独完成开发冻结；`--load-schedules` 只执行显式列出的到达时间表，其余表不会被标为已运行。

开发候选是三个声明的固定对照和四个 S3 λ 候选。对照显式 `plan_gain=None`；S3 绑定真实收益、成本和输出预算。native timing 的实际系统提示只应用于 rolling run 配置，Dense5 保持它本来的系统提示。初始化时保留输入配置和资产 SHA；实际 provider 加载后继续核对规范化常驻模型身份。

默认每个会话一个子进程，同一会话内按真实已提交 QA 顺序推进。某轮失败，该会话后续依赖轮不执行，其余会话继续；不重试、不填补 QA、不改变原始输出。所有调用命令、PID、退出码和 stdout/stderr 都在 `jobs/` 留存。若生成阶段失败但已有完成回答，且尚未开始评分，则对这些已存回答进行一次离线评分。已经开始但失败的评分保留其日志，不自动重试。缺失回答、评分或费用保持缺失。

此时缓存冷启动边界与 `--failure-unit` 一致：`conversation` 会按实际 study 的 `cache_start` 对每个会话执行；`dataset` 将完整轨道交给一个既有 `bench.study` job，失败时该 job 中未完成会话仍是未完成。该协议写入 `binding.json`，不能把两种边界的性能结果混称同一批次。

完整预定开发 cohort 均完成、原始主指标评分齐全且资源约束满足后，才按连续 Q、near-equal 阈值及既有成本次序产生 `frozen.json`。开发结果出来前，`prespecified-test.json` 已登记全部固定对照、S3 λ 候选和测试 task IDs。不存在可行开发方案时保留 `development_freeze.status=ineligible`，仍执行这些预先登记的独立测试与动态负载，标记 `unselected_prespecified`；测试结果不回流选型，也不补造冻结结论。动态负载结果独立保存在 `jobs/load-*`，不进入公开 Q，也不据此修订已冻结方案。

可选负载读取既有工作负载 manifest 的 `diagnostic_matrix_path`、`goal_policy_path` 与 `schedules`，保留完整到达顺序、停顿和真实回答驱动的分支，不拆会话。并发 4/8 超出实测 profile 支持时，该 rolling job 将记录不可行或失败；入口不会把并发 1 的成本资产宣称为并发 4/8 已校准。

关键产物：`binding.json`、`run.yaml`、`baseline-run.yaml`、`study.yaml`、`dev/results.jsonl`、`dev/summaries.json`、可行时的 `frozen.json`、`test/results.jsonl`、`report/`、每个原始 generation/state/scoring 目录和 `workflow.json`。workflow 的完整物理积分包括初始化、空闲与离线评分；每个回答的在线成本仍来自原记录。两个重叠口径不能相加。

S3 使用和 `build_read_plan` 相同的 `read_output_reservation` 对当前完整包计算逐 Need JSON/fact 输出预留；低于预留的 512/1024 等选项不会被编译或进入未来前瞻候选，不通过偷偷改小原文包取得资格。READ DAG 和逻辑状态保持原结构。

S3 当前使用实测 kind/prompt 与 output-cap 绑定的 native timing；最终 ANSWER 将完整控制先验替换成实测 `mean_remaining_control_ms`，此前编译/报价已在 paid ledger。CPU 未来输入容量从 execution 资产的 `claim_capacity` 计算，绑定实际 tokenizer、必需 claim keys 与原生 grammar 协议；当前已完成前驱用实际 claim 数，未来确定性 CPU 操作按其真实单个输出 claim 计上界。未提供该证明时保留原 token 数保守上界。未来 READ/REDUCE/ANSWER、GPU/Host、并发与正文上限仍必须满足所声明的实测或统计预测范围，预测范围与实际观测范围分别留存。
