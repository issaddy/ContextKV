# 接续与阶段验收

依据：2026-11-02 用户目标文件与《ContextKV 核心模块重构与实施指导》。
当前实现作为 Incumbent 候选；Dense5 仅为历史参考。新增训练已恢复授权。

| 能力 | 所在模块 | 代码 | 训练与真实证据 | 当前缺口 |
|---|---|---|---|---|
| 历史、任务、条件、版本、来源 | history_state、evidence_pool | 已实现 | 教师任务预测已实际运行；任务训练入口已有 | Windows 自动样例遗漏流程需求；需任务绑定替换诊断 |
| 单段、组合支持及预算集合 | support_model、teacher_support、evidence_planner | 已实现 | 容器已有 actual-prefix 支持资产、prefix12 偏好资产；尚无合格持出集收益证明 | 按当前 reader 实际回答重新产生偏好 |
| 阅读与持久化运行 | reading、execution、session_store | 已实现 | R0新balanced长文全scope 36/666通过；旧v10截断保留原始记录 | R1匹配配置复跑、自动链路必要点与多跳引用 |
| 多集合与原生报价 | candidates、planner、context_cost | 已实现 | 局部测试通过，原生报价已记录 | 收益驱动滚动选择及对应实测监督 |
| Reader R1 | training/reader_data、train_reader | 已实现并实际训练 | 1773 train examples/1544 turn，19 validation；56steps/7,800,825tokens/1,405,076.37ms；共同12开发状态E0/inferred为R0 .8708、R1 .7333 | 首轮未提升；精确topic六格、四格与正式QA heldout；长文/多跳尚未通过 |
| R1绑定P1 utility | training/train_evidence_policy、preference_rollout、a_workflow | 已实现并训练prototype22 | models/study-support-r1-prototype22：22真实答案/5状态；四格各14状态生成完成待score | 四格质量、P3开发比较与持出收益 |
| 来源干预与R2混合 | reader_curriculum、reader_constructed_curriculum、reader_computed_curriculum、reader_mix | 课程复核/审计完成，新P1按缺失ID复核 | 新增153；parent1773train+19validation后共1945样本/1544train turns；课程3x、新P1 8x为配置 | 新P1复核和实际混合统计、R2训练及效果；validation字节保持 |
| B v3收益模型 | collect_plan_gains、train_plan_gain | 真实采集和fit已完成 | train8/dev4，176 model calls、721867ms | 滚动控制质量/成本和独立联合效果 |
| 机制离线审计 | bench/audit_reading_mechanisms | 已实现执行/覆盖/原文/claims/REDUCE/最终量值分层审计 | 4项本地测试通过；实际R0/R1文件仍待在容器生成报告 | 受控literal/provenance结果，不替代正式质量指标 |
| 独立前缀所有权 | prefix_retention、原生补丁 | 已实现端点、持久化和执行回调 | 固定token配对已实测；完整READ/ANSWER/REDUCE成本采集和fit阶段在推进 | 最终成本资产、未来价值与联合实验 |

远端已核实训练轨迹：task-a/turns.jsonl 为 10004 轮、1539 会话，split=train，
对应 state/sessions.sqlite3 存在。模型、数据、日志、结果均留在容器。
TaskState、ContextPlan、EvidenceSet、ReadResult/ReadingOutcome 共同表达逻辑工作状态，
不再建立平行 AnswerState 副本。

R1 训练分区持出 loss 为 `0.1608121544`。这与正式 QA heldout 的回答质量不同。
当前 R1 四项受控流程均完成执行；长文 REDUCE 为 36 条/655000 g，最终却答
3 条/6 kg（期望 36 条/666 kg）；同代码 R0 REDUCE 为 36 条/596000 g，最终同为3条/6kg。
因此 READ 总量提取和 ANSWER 忽略全范围 Computed 结论分别审计，不能视为纯 R1 退化。
多跳未得到 1997，均保留为失败。pair/correction 的预期词项生成还需引用与来源审计。
最新 R0 同代码配对用于归因；尚不能据此选出最强 Incumbent 候选或宣称 R1 有质量提升。

| 阶段 | 依赖 | 验收及证据 |
|---|---|---|
| S0 接续与 Incumbent | 当前工作区、R0 | 源码与权重散列、提示模板、运行配置、正文/输出预算、评分口径；当前候选在相同开发题集实测后选出 Incumbent |
| S1 六格与小规模 R1 | S0、训练分区已复核目标 | R1训练/合并已完成；sixgrid-v1实评152格/缺16，主表共同12状态，首轮R1无提升；2泛topic需精确目标和范围审核 |
| S2 A 交替训练 | S1 | P1prototype22已训；冻结四格各14状态已生成、待score；训练P3首会话5QA后Task length终态，第二会话首轮约13分钟CPU temporal热点、无新QA，补丁重启待报告；R2混合/训练尚未完成报告 |
| S3 B 收益决策 | S2 | v3 train8/dev4真实176调用、721867ms并已fit；有限前瞻滚动运行、等累计计算质量曲线仍需实测 |
| S4 C 价值保留 | 原生机制验证、配对数据 | 固定 DAG/token 的原生/保留生命周期与成本对照；共享页边际占用；相对原生新增节省 |
| S5 联合、冻结、测试 | S2—S4 | 当前/更新 B × 当前/更新 C；P3 动态多轮、Loong 预算分析、同容量负载；开发选型后冻结再测试 |
| S6 最终交付 | S5 | 增量代码/调整清单、训练数据与权重清单、逐题答案/来源/步骤/评分、原生回执、完整成本与相对 Incumbent 结论 |

sixgrid-v1共同12状态的E1/inferred较E0留有R0约12.9、R1约26.7个百分点差距，优先改进实际选集。
R1下一轮局部诊断至少应达到匹配E0下R0的0.8708；若以本批R0候选为参照，指导要求的质量提高5—8个百分点
对应0.9208—0.9508。正式目标仍在完整预定cohort与同在线预算验收。
条件/版本与必要点错误减少30%、同物理容量完整响应P95≤基线1.2倍需单独错误计数和负载数据，尚无实测达成证据。

最新已确认部署：R1 `30004/GPU4`、可比R0 `30005/GPU5`、冻结judge
`30001/GPU2`、C `30002/GPU3`。P1采集的BGE/query工作在GPU1，support/utility
工作同在GPU1；在线P1采样集合为`[1,2,4]`，P1 utility训练使用GPU6。
R2拟使用0、3、6、7四GPU，须由root安排服务和训练生命周期后启动；这里不记为训练已完成。

公开多跳来源已完成下载和40/40互斥cohort转换：2WikiMultihopQA官方有标签dev共12576条，
40条用于开发、40条为project-heldout，不是官方隐藏test。MultiDoc、Loong及2Wiki的采样IDs和源SHA
由a_final_batch准备阶段固定；公开生成、评分、05:00开发冻结与09:00交付仍依实际产物更新。

只访问指定主机的 sjy-0518，单 SSH PTY。只同步必要源码与源码配置；
终端诊断单次最多 700 字符。最终完成后退出容器、主机和 Relay，确认本地 SSH 进程退出。

## 2026-11-03 09:00 截止推进

00:43 起按用户要求收敛：已有 1761 项完整回归作为基础，不再反复运行完整或零碎测试；
只保留新数据管线的一状态烟测、原生契约资格检查和最终集成检查。
明确终态的模型预算失败计入成本和覆盖率，继续其他候选与会话。

| 时间窗口（北京时间） | 并行产出与收敛点 |
|---|---|
| 00:43—01:30 | A 完成 P1 首批偏好与课程复核；B 交付真实反事实生成；C 采集并拟合原生及物理成本资产；主 agent 固定原生 wire、完整数据与设备安排 |
| 01:30—03:00 | P1 训练、自身 P3 与下一 reader；四格和完整五场景诊断；保留实测较强 reader，停止无收益的训练分支 |
| 03:00—05:00 | B 实际 ΔQ 训练与滚动运行；C 配对、TTL 与压力数据；当前/更新 B×C 及必要消融，开发选型 |
| 05:00—08:00 | 冻结源码、资产、配置及预定公开测试 cohort，运行测试与同容量负载；只处理影响闭环的故障 |
| 08:00—09:00 | 汇总逐题质量、覆盖、完整成本和取舍，核对交付清单及代码提交，退出并核对本任务 SSH 会话 |

所有时间窗口以交付结果为准；未通过的样例保留为失败数据，不能延迟其他独立工作。
