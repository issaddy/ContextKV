# 显式本地模型 owned 入口的有限真机验收

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。`bench.b6_joint_sweep --matrix` 已在 H200 上贯通共同 Qwen3-8B scope、采样、原生 B1、研究控制器及实际产物记录。**两格各两轮，共4次真实生成，全部完成且自然 stop。** 这关闭新入口的功能验收缺口；没有扩大质量实验、选择B6或标定成本模型。

## 实际执行范围

运行源码为 `8d48860b727942914a44656f22d3ee180eb0caad`，346个tracked文件部署核验通过；运行时实现与已通过1328项本地测试的 `af9e139` 相同。固定SGLang commit `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，stock与patched各自核验，研究补丁未变化。

共享配置为 `/media/ssd1/qwen3-8b` 的显式metadata scope、non-thinking、`.7/.8/20/0`、输出64、上下文32768、page64、GPU池16384、host请求8 GB、`mem_fraction_static=.8`。两个方法都使用write-through、SessionRadix/HiCache、FA3、FCFS、chunk8192、server max-running8与seed42，工作量并发1。新owned入口的实际host池均为54272 tokens、8002732032 bytes；请求的GB没有冒充实测容量。

输入是一个自造两轮会话，每轮五段简短材料；第二轮询问第一轮对象的位置。输入文件另含独立非空的synthetic test占位会话，仅满足split合同，不执行它，也未读取真实heldout。P2采用本方法实际生成的上一答，文件内特意设置的离线答案占位文字不能进入实际历史。

| 方法 | 实际动作 | 请求结果 |
|---|---|---|
| 原生stock B1 | 两轮均提交完整重建prompt，保持同一native session | 2/2成功，自然stop |
| patched Controller | 首轮REBUILD，第二轮APPEND | 2/2成功，自然stop |

首轮两臂实际输入相同；后续分别携带自身真实QA，不要求随机采样的答案逐字一致。Controller第二轮receipt记录1088 device-hit tokens；它是本次具体请求的观测，不是吞吐或完整响应收益结论。原生B1的prompt与SessionRadix工作方式保持原合同，没有改成研究APPEND。

## 核验与工程收益

实际服务身份、BF16、无伪HF revision、资源与缓存flags通过现有validator；原生manifest由启动后的真实观测生成。Controller初态metrics确认GPU/host已用tokens均为0；stock初态依据独立新进程和跳过生成warmup，没有伪造零使用量快照。JIT磁盘缓存不声称清除。

独立[离线审计](../../runs/owned-local-scope-probe-review/h200-v1-final/analysis.json)复用真实tokenizer与现有候选/验证器，逐token重建4个输入、核对正文spans和实际自有历史，问题数为0。B1输入1131/1179 tokens，Controller1131/1252 tokens；4次输出均为28 tokens并自然stop。原生采样证据是保存的实际HTTP请求，研究采样证据是客户端提交事件与真实terminal receipt，不声称后者是HTTP抓包。

两格均保留原始输入/输出IDs、五源及实际历史、server/runtime、原生HTTP或研究receipt/events、完整任务分母和质量模板。质量模板绑定真实local-snapshot身份，`hf_revision:null`、`weight_content_verified:false`，评分与frozen protocol保持空；模板不是质量资格。

工程收益是共享模型/采样配置可经正式入口执行B1和Controller，并自动核对实际容量、保留结果、逐格清理。不再需要为新模型绕过矩阵另写生成driver。固定策略未加载旧7B成本表；采样不匹配的cost准入拒绝合同继续保留。

外层supervisor仅停止已确认归属的原7B进程组，部署后串行运行两格，并最终恢复原服务。两格owned进程的最终returncode均为−9，记录原值，不声称优雅退出。研究retire_ref的即时物理释放为0，后续仍有缓存驻留；本次清理指结束owned进程，不把退役ACK解释成页已释放。独立检查确认恢复PID75096、HTTP200、原7B fingerprint、GPU121920/host279040且已用均0。首次独立健康检查早于加载完成，连接被拒；等待ready后才确认恢复，launcher返回没有冒充服务就绪。外层异常/超时/清理失败/恢复失败的8项CPU合同与7项审计边界检查均通过。

## 可复现产物与边界

- [声明、fixture和CPU检查](../../runs/owned-local-scope-probe-prep/README.md)，[外层启动与恢复脚本](../../runs/owned-local-scope-probe-launch/supervisor.py)。
- [真实运行](../../runs/remote-h200-owned-local-scope/runs/owned-local-scope-probe-v1/owned/completion.json)与[独立恢复检查](../../runs/remote-h200-owned-local-scope/restored-health.json)。
- [52文件公开快照](../../runs/owned-local-scope-probe-public-v1.tar.xz)，60744 bytes，SHA256 `28af797b77a2e5814412c41c1fbb7e31682824944712c1bdd86170b2821b701f`；51个payload逐文件验证。快照不含权重、私有环境、私有完整日志或数据库；恢复检查在加载完成后另存。

本轮只验证8B、B1与固定APPEND策略的两轮工程路径。B0之前已有独立实际入口检查，此处没有为凑三格重跑；32B、P3、资源压力、同scope成本拟合、质量非劣及性能优胜不由这4次请求推出。后续文档/分析器变化不触发重跑该验收；随后已使用既有159轮完成[CPU候选结构审计](../research/qwen3_candidate_opportunity.md)，据此缩小必要的真实成本观测。
