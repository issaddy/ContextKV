# 动作收益与完整成本接口

`controller.rolling_policy` 复用 `ReadingOutcome`、`ContextPlan`、已读事实、coverage 和 open_needs。当前 `observed_actions` 候选为直接 ANSWER、一次 READ、一次 REDUCE、一次 GAP。只执行选中的动作，再根据真实结果重新决策。GAP 绑定实际 provider 与已声明材料限额，返回的新来源及父事实进入同一状态。全来源覆盖要求先由现有固定流程完成。

`training.plan_gain_workflow --mode observed_actions` 使用显式 cohort 和真实观测包，执行 DIRECT 与候选分支，保存原始回答、回执、质量和完整费用。候选分支按 `infer_bridge_ready_or_one_gap_one_terminal_read_v3` 执行首动作与固定有界后续：已有终端来源优先，必要时一次 GAP，最多再执行一次终端 READ；直接 ANSWER 不续接。GAP 的已读范围与新增 passage 额度与线上共用。确认终止而没有交付回答的分支取零交付效用，未决调用保留缺失。源请求后来失败时，其早先真实观察仍可参与采集。最终资产默认为 `plan-gain.json`，相邻的 `.development.json` 记录开发配对质量差的误差。

质量模型只学习当前语义状态下的实际最终质量差。`pre_action_native_absolute_v1` 成本模型使用动作前 GPU/Host 空闲量、压力、并发及可编译首步的命中/回载报价。DIRECT/READ 使用对应的首步输入，GAP 和 CPU REDUCE 使用该时点批级状态。标签为新动作至最终回答或确认终止的绝对分支费用，共同已付状态、准备和清理费用各自保留账本。

在线 `action_policy.py` 用同一批原生观测分别预测候选和 DIRECT，按两者绝对费用之差计算增量代价。已付成本加绝对分支成本约束预算。质量特征独立于原生缓存特征。费用训练标签使用 ms、GiB·s 和 token，在线预算转换成 ms、GiB·ms 和 token。C 的缓存效果已进入当前命中与费用预测，同一缓存节省只计一次。

`RunConfig.plan_gain` 的 `observed_actions` 配置包含收益资产、实际 reader checkpoint/profile、采样 GPU/PID、采样间隔、输出预算、前缀目录及 GAP 限额。运行时从 provider 绑定检索身份。旧 `whole_plan` 对照继续使用整计划收益、执行 profile 和原生时间标定，比较完整计划及其后续成本。

完整联合入口 `bench.capacity_workflow --full-queue` 读取冻结 P0 共享 A 配置，固定其支持、查询、阅读、预算和生成条件。c4 执行 E0 及固定 B / 收益 B × native / value 的 E1—E4；c1 只执行 E1/E4。每格使用原 350 题声明清单和各自实际历史，失败、缺失及已付费用随完整分母保留。配置和命令统一见 [运行手册](final_runbook.md)。
