# 模块现状、工程缺口与实现优先级

2026-10-24。目标是目标场景下回答质量、完整响应时间、资源效率与可靠性的综合效果。当前只使用用户已有Qwen3-8B/32B，Qwen2.5-7B已退役。下表的“完成”指功能链路和合同实现，不代表算法或系统优胜。

## 模块盘点

| 模块 | 已完成、可复用 | 待完善或优化 | 可扩展但不作为当前前置 |
|---|---|---|---|
| A：证据管理 | 原文/版本身份、稳定引用、明确依赖闭包、CorpusIndex、完整QA与证据spans；当前required的显式依赖和章节关系传入A/R共同提示 | 语料元数据完整性、对象/时间/版本/条件与答案的语义对应；关系可见不等于支持正确 | 固定SDK包已可接现有检索/HTTP；结构表格、source-local地址按材料需要启用 |
| A：检索与上下文 | BM25S、BGE dense、RRF/reranker、多轮query视图；召回与重排查询可分离；CLI与HTTP复用同一factory、原文metadata闭包与排名；共同A/R构造与批量分词 | 选中来源适用性、追问指代、错误来源进入答案；Qwen3历史重渲染的前缀边界 | 已有复杂改写/selector保持可选；不再扩已失败的两阶段来源计划 |
| B：在线决策 | 固定/周期/冗余/warm/cost策略、原生报价、阶段与长度模型、支持域和显式回退 | Qwen3实际范围内完整成本；先识别有价值候选，不为覆盖率拟合整表 | 必要时对有明显取舍的形状做小规模标定；没有净收益就保留强简单策略 |
| C：KV与资源 | epoch/CAS/attempt/outbox、原生Radix/HiCache、共享pin、GPU/host ready/pending/传输观测 | 自然负载下保护收益与后续重算代价，业务关闭与物理释放的边界 | host admission默认off；不新建影子cache/scheduler，不用受控fixture代替自然净收益 |
| 服务与恢复 | 幂等请求、完整原生IDs/receipt、未知结果恢复、成功一次提交、关闭与退役重试；自动检索在同一会话串行边界内，重放原结果/来源不重检索 | 持续处理实际异常链；保留未知结果和失败分母，不能承诺跨引擎重启exactly-once生成 | 只有保留的回答机制才接入业务；不为负结果预建通用多阶段事务 |
| CLI与HTTP | Qwen3共享配置；CLI完整P2/P3；HTTP自动检索→完整QA/证据→回答→提交/恢复；实际8B两轮正常终态、完整IDs/历史、重放和close已验收 | 故障/跨重启边界分别保留原合同与证据；缓冲响应不等于外层逐token流，正常两轮不授予质量/性能资格 | 现有索引/元数据可复用；未实现上传/索引管理服务，不为此新增独立服务框架 |
| 基线与评价 | 同模型B0/B1、显式owned矩阵、真实任务/失败分母、质量绑定、B6别名、冻结与匹配消融 | 最终共同配置、评价器语义误判、真实独立heldout/人工审查 | 主结论才做充分对照；不增加judge数量追通过、不把助手审查当人审 |
| 多轮监督适配 | Doc2Dial v1/v0.9原文与角色/grounding侧车、显式格式/版本、真实train原样核验；语料接现有检索 | 模糊条件标签、target语义噪声、来源隔离；连续同role不直接套当前QA gate | 按真正历史/条件决策需求再适配训练，不把导出当质量成果 |
| 工程交付 | 固定引擎/许可证、CPU合同、包资产检查、原始输入输出与可复算记录 | 当前入口与文档持续一致、异常路径结果完整、清理无效分支 | 更广硬件/拓扑按实际部署需求扩展，不作为当前单GPU交付前置 |

通用模型无关的CPU模拟、缓存协议、成本拟合和只读历史数据解析继续保留。历史结果不改写为新模型结论；旧运行配置和驱动不因仍有历史报告就继续保留为可执行默认。

## 本轮实际实现与清理

1. **停止旧模型使用。** 容器旧7B owned服务已停止，GPU0释放；旧自动恢复脚本/私有配置移出活跃入口，后续不再恢复。默认ModelConfig不再指向任何真实模型，硬件运行必须显式Qwen3 scope。
2. **移除过时执行分支。** 删除旧模型下载/启动/runtime fallback、固定13格构造器、旧capacity/native sweep执行器；保留被新owned调用的进程、实际池和runtime校验函数。删除9份旧配置和冗余运行指南、无生产消费者的旧manifest生成模板。
3. **统一模板与分析入口。** decision-pairs freeze复用共享模型factory，不再把local-snapshot误传为HF revision；采集核对frozen模型与配置scope并传给client。只读候选导出只解析所需policy，不实例化旧硬件配置。Qwen2专属分词快路移除，Qwen3完整prompt批处理路径保留，真实tokenizer合同改为Qwen3。
4. **补齐P3失败闭环。** 首轮检索失败不清理不存在的session；清理/末次metrics失败先保存原报告再重抛。显式null timing source归为unavailable，使成功/失败混合结果可序列化。未增加隐式重试、重复hash或通用异常吞掉层。
5. **接通HTTP自动检索。** `/sessions/{id}/query`只接收问题和request ID；复用已加载BM25/Hybrid及CorpusIndex，保留原始排名与追加条件正文的区别。历史只取服务已提交的完整QA；幂等检查、未决恢复、读取snapshot、检索与提交共享原会话锁。结果保留原attempt的检索耗时和可见来源，重放不再检索；端到端时序包含检索，控制成本不重复相加。
6. **收束长服务资源与记录。** 检索/元数据只在启动加载一次，部分初始化失败、取消及退出均关闭已创建资源；Hybrid外部trace sink不再积累请求列表。历史来源投影改用已提交QA序号，避免恢复/CAS版本跳号误判可见轮次，原TurnRecord身份不变。默认路径不新增逐请求hash或模型调用。
7. **补齐结构证据的数据链。** 自有语料检索按passage_id区分同document的多个片段，保留source_uri及显式namespace。原本已补入正文却丢失的dependency/parent_section进入A/R共同提示，原生基线同步使用；无元数据输入逐字不变。冻结SDK包的实际BM25→闭包→Qwen3分词已通过，所需额外121tokens计入预算，没有新增模型调用。见[原理与边界](api_evidence.md)。

依赖核查和原始失败复现见[整合审查](../runs/qwen3-consolidation-review/inventory.md)，HTTP配置、失败/恢复及计时边界见[入口说明](http_retrieval.md)。本轮真实BM25、Controller/SQLite与MockEngine验证的是工程合同，不是回答质量或GPU收益。部署、CPU测试与包验证结果在阶段完成时单列，旧四调用只证明当时版本，不冒充本轮HTTP或引擎改动的GPU验收。

## 接下来按实际价值推进

**P0：保持单条完整可用路径。** Qwen3共享配置→现有检索→完整QA/证据→单次生成→原生提交/恢复→结果导出，CLI与HTTP已共用这条主线；本轮集成回归和使用文档已完成。成本/准入等可选研究机制不阻塞基本服务，后续接口实测只验证尚未覆盖的实际边界，不重开GPU矩阵。

**P1：解决可解释的主要成本与质量瓶颈。** 消费已完成的[159轮结构审计](qwen3_candidate_opportunity.md)，先核Qwen3模板边界和原生增量机制；不得复用不匹配KV。A的条件/引用语义错误与B的prefill/decode取舍分别处理，局部指标不代替系统效果。任何新适配都先检查原理和代码合同。

**P2：最小充分实测。** 仅当新实现有明确剩余不确定性才运行Qwen3验证。优先有限协议或代表性真实负载；采用的关键机制才安排强基线/消融，最终结论才冻结独立测试。不重开旧7B实验，不复制模型×容量×并发×提示矩阵。

可运行工程、研究批次完整、整体优胜是三个不同状态。负结果不阻碍交付可用系统，也不允许降低证据标准宣称收益。

## 本阶段检查

新Doc2Dial适配与共享检索取消收尾通过1383项测试及8个子测试，真实Qwen3 tokenizer项开启，Ruff/diff通过。两份作者train的完整转换、角色/文本/引用独立比较及488文档真实索引→五文档→5424-token Qwen3候选通过；没有模型调用。P3取消反例证明关闭前必须等待同一次检索线程，未生成或提交答案，后续未完成分母保留。记录为`runs/pytest-doc2dial-integration.log`与`runs/doc2dial-integration/audit.json`。

实现`185fa7e`已打包并同步容器：87模块、31资产、353个sdist文件、107测试/配置及13许可逐字核对通过；354个远端tracked条目与提交一致，HTTP与新adapter CLI导入/help及编译通过。模型服务未启动，原生补丁保持不变；包和部署原件分别为`runs/dist-doc2dial-integration/`、`runs/remote-h200-doc2dial-integration/`。下述c1cc617两轮GPU记录保留原身份，不追认为新源码的GPU重复验收。

`c1cc617`另完成当前HTTP正常路径的两次真实Qwen3生成，REBUILD/stop各2次，无重试，实际历史/原文/IDs/receipt、幂等与关闭经独立审计0 issues。使用仓库合成语料和现有采样/cap，不作质量或性能比较；范围覆盖和scope连接的局部问题原样保留。详见[真实HTTP验收](http_retrieval.md#qwen3真实http链路验收)。GPU0及本轮端口已释放。

最新结构证据集成回归通过1373项及6个子测试，开启真实Qwen3 tokenizer metadata；Ruff/diff通过。交换依赖边、旧APPEND账本、精确revision/cycle、无元数据兼容及预算计入均有合同检查；另用既有固定SDK包打通真实BM25索引/元数据/Qwen3分词。记录为`runs/pytest-evidence-context.log`和`runs/evidence-context-integration/sdk-v1/result.json`，没有模型/GPU调用，不能据此宣布语义质量提升。

实现`c1cc617`已打包并同步容器：86模块、31资产、350个sdist文件、106测试/配置及13许可逐字核对通过；351个远端tracked条目与提交一致，CLI导入/编译通过。原生补丁未变、GPU0空闲，源码证据为`runs/remote-h200-evidence-context/`，包证据为`runs/dist-evidence-context/asset-check.json`。

HTTP集成后完整回归通过1361项及6个子测试，开启真实Qwen3 tokenizer metadata，无跳过；Ruff和diff检查通过。真实CLI/TCP/BM25两轮验证了条件闭包、完整历史、旧请求重放、冲突、输入白名单与关闭，只有2次模拟生成、0次模型调用。随后补齐的启动来源事件与取消竞态由完整回归覆盖。记录在`runs/pytest-http-retrieval.log`和`runs/http-retrieval-tcp-v1/`；TCP退出完成应用清理，macOS信号退出另有1个semaphore清理警告，尚不宣称长时间资源稳定性已验证。

实现`0d1ae96`已提交、打包并同步容器。wheel/sdist逐字核对86模块、31包资产（含可运行HTTP示例）、348个sdist文件及13份许可，通过且无运行数据、权重、私有环境；记录为`runs/dist-http-retrieval/asset-check.json`。容器349个tracked条目与同一提交一致，CLI导入/help和Python编译通过；原引擎补丁未改动，GPU0空闲，没有启动模型。远端公共证据位于`runs/remote-h200-http-retrieval/`。

以下为上一轮Qwen3整合的已完成记录，不包含本轮新增HTTP自动检索代码：

全仓回归在真实Qwen3 tokenizer metadata下通过：1323项测试及6个子测试，Ruff和diff检查通过。引擎补丁固定上游源码的应用/反向检查、15个修改文件语法与镜像校验通过。全部为CPU/源码检查，本轮没有模型生成；此前硬件结果保留各自身份。

源码`771aef9`的wheel/sdist核对通过：85个模块、28个包资产、340个sdist文件、13份许可，无运行数据/权重/私有环境。容器已同步同一实现，341个tracked条目核对及旧补丁反向/新补丁应用通过；9个旧配置和2份过时文件已移除。Qwen3两份实际本地库存与CLI导入检查通过，GPU0及原服务端口保持空闲；本次部署没有启动模型，也没有恢复旧7B。
