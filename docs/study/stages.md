# 接续与阶段验收

依据：2026-11-02 用户目标文件与《ContextKV 核心模块重构与实施指导》。
当前实现作为 Incumbent 候选；Dense5 仅为历史参考。新增训练已恢复授权。

| 能力 | 所在模块 | 代码 | 训练与真实证据 | 当前缺口 |
|---|---|---|---|---|
| 历史、任务、条件、版本、来源 | history_state、evidence_pool | 已实现 | 教师任务预测已实际运行；任务训练入口已有 | Windows 自动样例遗漏流程需求；需任务绑定替换诊断 |
| 单段、组合支持及预算集合 | support_model、teacher_support、evidence_planner | 已实现 | 容器已有 actual-prefix 支持资产、prefix12 偏好资产；尚无合格持出集收益证明 | 按当前 reader 实际回答重新产生偏好 |
| 阅读与持久化运行 | reading、execution、session_store | 已实现 | 组合及纠正受控样例实际生成；v10 长文截断 | 自动链路必要点遗漏、长文抽取和多跳引用 |
| 多集合与原生报价 | candidates、planner、context_cost | 已实现 | 局部测试通过，原生报价已记录 | 收益驱动滚动选择及对应实测监督 |
| Reader R1 | training/reader_data、train_reader | 已实现并实际训练 | 1773 train examples/1544 turn，19 validation；56 optimizer steps、7,800,825 input tokens、1,405,076.37 ms；合并与30004/GPU4服务就绪，R0保留 | 六格、四格与正式QA heldout；小规模R1未通过当前长文/多跳受控任务 |
| R1绑定P1 utility | training/train_evidence_policy、preference_rollout | 采集、配对、训练、后继P3代码已实现 | parent=models/study-support-actual-prefix；尚无当前R1实际偏好/P1训练或持出收益证据 | 先审计旧Task标签的原文指针与Need图，再小批真实采集 |
| 来源干预curriculum | training/reader_curriculum、reader_constructed_curriculum | 独立round2数据构造、fresh teacher复核与配对审计入口已实现 | 本地构造/不变性/来源敏感性测试；尚无这批实际teacher标签或round2权重 | 数字/条件/版本、分散窗口、三跳与no-record窗口的真实目标和效果验证 |
| 独立前缀所有权 | prefix_retention、原生补丁 | 已实现端点、持久化和执行回调 | 固定token实际GPU/Host配对各新增128命中，延迟效果分别约+10.4/-17.0ms节省 | 语义READ资格、真实轨迹价值资产与联合实验 |

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
| S1 六格与小规模 R1 | S0、训练分区已复核目标 | R1训练/合并已完成；E0/E1/E2 × R0/R1及任务绑定替换仍待完整实测；含续问、纠正、切换、回访、版本变化 |
| S2 A 交替训练 | S1 | 入口已实现；当前R1实际偏好、P1权重、P0/P1 × R0/R1同历史及各自P3尚缺；数据和收益必须绑定实际reader |
| S3 B 收益决策 | S2 | 相对直接回答的实测 ΔQ 数据与模型；有限候选两步前瞻，逐步重选；等累计在线计算质量曲线 |
| S4 C 价值保留 | 原生机制验证、配对数据 | 固定 DAG/token 的原生/保留生命周期与成本对照；共享页边际占用；相对原生新增节省 |
| S5 联合、冻结、测试 | S2—S4 | 当前/更新 B × 当前/更新 C；P3 动态多轮、Loong 预算分析、同容量负载；开发选型后冻结再测试 |
| S6 最终交付 | S5 | 增量代码/调整清单、训练数据与权重清单、逐题答案/来源/步骤/评分、原生回执、完整成本与相对 Incumbent 结论 |

六格诊断后登记具体数据集与预算点的量化目标。指导中的开发目标为质量提高
5—8 个百分点、条件/版本与必要点错误减少 30%、同物理容量完整响应 P95 ≤基线 1.2 倍；
这些目标尚无实测达成证据。

最新已确认部署：R1 `30004/GPU4`、可比R0 `30005/GPU5`、冻结judge
`30001/GPU2`、C `30002/GPU3`。P1采集的BGE/query工作在GPU1，support/utility
工作在GPU6；物理采样集合为 `[1,2,4,6]`，P1 utility训练只采GPU6。
配置使用完整设备可见性下的 `cuda:1`/`cuda:6`，不能再套单卡可见性后保留这些编号。

只访问指定主机的 sjy-0518，单 SSH PTY。只同步必要源码与源码配置；
终端诊断单次最多 700 字符。最终完成后退出容器、主机和 Relay，确认本地 SSH 进程退出。
