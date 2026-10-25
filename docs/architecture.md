# 实现与事务边界

课题场景、A/B/C 的职责与当前推进顺序统一见[项目范围](project_scope.md)。本文描述现有工程接口；可选研究分支不自动成为主流程前置。

控制器为单 Python 服务；SQLite WAL 是跨进程一致性的边界。每个业务会话有串行异步锁，数据库另有一个未决 attempt 的唯一索引。多会话可并发，部署默认一个 Uvicorn worker。无需数据库服务或 GPU 即可运行正确性演示。

```mermaid
flowchart LR
  Data[白名单当前问题与证据] --> Candidates[APPEND / REBUILD 纯候选]
  Store[(SQLite 双账本)] --> Candidates
  Candidates --> Quote[成本或warm/idle策略：同批只读报价]
  Candidates --> Fixed[固定策略跳过报价]
  Quote --> Planner[成本 / 固定策略]
  Fixed --> Planner
  Planner --> Prepare[仅选中版本 prepare]
  Prepare --> Generate[完整 token 输入与代际 guard]
  Generate --> Result[持久化完整结果]
  Result --> Commit[CAS 提交历史与 outbox]
  Commit --> Store
  Commit --> Retire[异步退役 ACK / 重试]
```

## 应用层

- `schema.py` 定义 frozen dataclass，集合与消息使用不可变结构。`serialization.py` 按类型恢复 JSON，不使用 pickle。
- `evidence.py` 以 corpus/document/passage/version/SHA256 定义身份，稳定引用发生短 hash 碰撞时仅延长新编号。回访只从已归档原文恢复；必须证据先统一形成，再构造候选。
- `templates.py` 对完整 prompt 分词；保留证据原文的消息内字符 span、身份和来源。ByteTemplate 只用于模拟；HuggingFaceTemplate 固定 revision，每个候选只渲染一次；Qwen3 在已核实 AddedToken 边界按请求复用共同分词段，其他 tokenizer 沿用完整 prompt 编码。与原生 apply_chat_template 分词结果的一致性放在 tokenizer 测试中验证。
- `candidates.py` 保留完整原始 QA。APPEND 只省略可用正文 span 证明仍在输入中的材料；历史同步、配置变化、冲突版本、恢复或超预算会使其不可行。REBUILD 从原始 QA 和共同证据重建。
- `cost_model.py` 使用已完成历史估计输出长度，按 prefix/suffix/host-load/concurrency 查询标定点。未标定桶、未知 host 状态与不确定传输不能当作准确预测。
- `planner.py` 先满足硬约束，再选择预计完整成本较低者；误差带仅作诊断，成本相同时才复用原生共享前缀锁观测排序 KV 工作集。正常配置使用原生实际预算观测，缺少执行成本支持时采用显式简单回退。`observe` 提供残差更新接口，默认运行冻结离线模型。

校验集中在数据进入、引擎响应和持久提交边界。新传入证据在候选入口校验一次内容 hash，内部共用证据索引并信任已提交的不可变账本；归档时不重复哈希或读回已有正文。报价身份由计划器统一校验，成本模型直接使用已校验的数据。保留用于证据身份、实际输入关联和请求幂等的 hash，以及 CAS、epoch 代际和完整回答提交约束；不为内部正常数据增加重复防御层。

## 持久状态

`sessions` 保存当前快照；`ledger_history` 保留每次提交与 P1 历史同步的审计副本。语义历史与执行消息、实际 input/output IDs 分开。证据文本另按内容 SHA256 存储。

`attempts` 保存请求身份、选中计划、prepare 身份、epoch、完整结果及处理阶段。`request_id` 在同一业务会话内唯一；相同 ID 的不同请求报冲突。跨控制器重复 begin 不会获得执行权。状态只能前进，已选计划和已完成结果不能由旧对象覆盖。计划与新 epoch 在 prepare RPC 前一起落盘；prepare ACK 与提交标记在 generate RPC 前一起落盘，避免无外部调用间隔的重复持久化。

一个事务同时写入新 QA、执行账本、当前 epoch、已提交 attempt 和旧 epoch 退役 outbox。事务内重新检查 attempt 阶段与 snapshot version。新版本提交前绝不退役旧当前 epoch。

多会话的后台退役通知合并到一个运行中的worker。慢退役期间新提交的outbox由下一轮快照继续处理，不为每个已完成请求积压一个等待锁的task；失败仍留在原持久outbox，由既有周期重试或显式drain处理。此处不改变原生淘汰策略，也不将减少控制器任务误报为释放GPU KV。

## 故障恢复

| 状态 | 恢复行为 |
|---|---|
| 构造/报价失败 | 标记失败，语义档案不变 |
| prepare 响应丢失 | 保留 operation ID；恢复时幂等取得原 generation，再清理 |
| 请求已提交但响应丢失 | 查询 scheduler status；只有完整结果可直接恢复提交 |
| 完整结果已落库、提交失败 | 保留 RESULT_READY，同一结果恢复 CAS，不重新采样 |
| 新账本已提交、退役失败 | 保留新答案，outbox 重试；不能回滚 |
| APPEND 结果被放弃 | 标记下一轮强制新 epoch；恢复成功后才退役原 epoch |
| 引擎重启 | 原始档案保留；旧实例引用失效，完整重建 |

Mock 与真实 SGLang 的 status 在有限窗口保留完整输出。真实补丁在完整 HTTP 结果交付前将输出与 receipt 发布到有 TTL 和字节上限的恢复缓存；SSE 必须收到终态 receipt 和 `[DONE]` 才可提交。超过恢复窗口或引擎重启后，无法恢复完整结果时需要显式放弃已确认终态的结果并使用新 request ID 重试。该协议不提供跨引擎重启的 exactly-once 生成。未知或仍在途的状态不会被超时自动当作失败。

准备阶段遗留 attempt 的 `--abandon` 是管理恢复操作，仅在旧控制器已经停止后使用。API 会定期重试退役，CLI 退出前排空；outbox 在进程重启后继续存在。

## 引擎与计量

Mock 模型为 CPU 确定性状态机，用于页边界、共享前缀、软引用、在途副本与故障注入测试。其 token slot 和耗时不能用于 GPU 性能结论。

真实引擎必须匹配固定 SGLang commit 及研究协议能力，配置指纹同时覆盖模型、tokenizer、template、namespace 和 cache key。普通未补丁服务器不能冒充 epoch guard 支持者。原生 B0/B1 使用单独的基线通道。

退役 ACK 表示软引用已撤销，物理 free 为零。真实分配、淘汰和传输仍走原生 admission、锁和完成边界。账本中逻辑 prompt 长度不能当作唯一物理占用。

研究客户端与原生基线默认使用 SSE，记录首个非空输出 chunk 和最后一个新增 token chunk，终态发布开销不计入末 token 时刻。完整响应和账本提交仍计入完成时间。非流式或恢复重放没有客户端首 token 观测，故 TTFT 保持未知；不能用服务端首 token 墙时填补客户端指标。

客户端单调时钟只计算端到端时序；服务端时钟只计算服务器阶段和全局唯一 slot 积分；CUDA event 在原生 ACK 完成点提供传输耗时。探针测得的 `first_token` 已包含服务器排队、恢复、Prefill，不重复累加同一段时间。占用峰值来自观测采样，并非瞬时分配器峰值；缺失数据与事件截断均保留标记。

`ablation.epoch_retirement=false` 让文本 REBUILD 沿用原生 session，P1 历史同步和故障恢复仍创建新 epoch。成本分项消融使用冻结模型，不能再用总延迟残差把被移除的成本学回去。完整标定冻结、开发集选择和测试数据彼此分离。
