# 异步挂起定位

真实 `incumbent-pilot-v2` 已确认根因：第三轮任务的教师输出 `.needs.n0.anchor_spans[1]` 为 `turn_id=0, speaker=user, start=0, end=115`，实际历史只有 turn 1、2，当前为 turn 3。旧 wire 允许任意整数 turn_id；`TaskInterpreter` 解析该锚点时，`history_text` 对不存在的历史行执行 `next(...)` 并抛出 StopIteration。

远端日志显示该异常在线程结果转交 asyncio 时进入 `_copy_future_state → dest.set_exception`，产生 `TypeError: StopIteration interacts badly with generators and cannot be raised into a Future`，目标 Future 一直 pending。于是同步线程已经结束、GPU 空闲，业务仍停在尚未保存检索结果的 STARTED。普通取消还会进入 `run_retrieval_worker` 的 shield drain，继续等待同一个已无法投递的结果。

修复将 TextRef wire 绑定到当前实际消息所有者及其 Unicode 字符边界：只允许已有历史 user/assistant 和当前 user，禁止 turn 0、未来 turn 和当前 assistant；缺失的可选 key/predicate 引用用 null。`TaskInterpreter` 在使用预测前按字段路径核对来源与跨度；解析函数通过真实消息索引抛出明确 ValueError，保留原始输出和已支付教师调用记录，不改写返回的 turn 0，也不通用捕获 provider 异常。wire 版本登记为 `owned_task_graph_v3_message_refs`。

回归包含真实坏引用的字段形状、native schema 的合法所有者、nullable 引用、线程检索失败后保存 FAILED 且不新增 QA。当前本地 CPython 3.12 会把线程 StopIteration 转成可递送 RuntimeError，远端观察到的旧实现会产生回调 TypeError；测试明确区分这两种传输行为。

当前源码中，新请求先获得会话锁、持久化 STARTED，再 await `evidence_provider`；成功返回后才保存 evidence/query_context/retrieval_ms。因此无 plan、无 retrieval 保存的活跃新请求，首先应核对该 provider await 链。退休锁不在这一分支的前置路径上。普通取消会在同步检索线程排空后把无计划的 STARTED 写为 FAILED。

同步 provider 内的 HTTP、BGE 或 threading.Lock 阻塞应留在线程池业务栈里。若线程池确实全部空闲，单靠 Python 主线程处于 selectors.select，无法区分任务正在等未递送的 Future、取消清理、其他异步事件，或账本里的 STARTED 已失去活动调用者。此时不据观察时长重启或补写 QA。

`bench.async_diagnostics` 提供显式开启的观测。新 benchmark run 设置：

```bash
CONTEXTKV_ASYNC_DUMP=/absolute/run/async-tasks.jsonl python -m bench.study evaluate ...
```

先确认文件中 `async_dump_installed` 的 PID，再对该 PID 发送 `kill -USR2 PID`。未安装 handler 的旧进程不能使用该命令，USR2 的默认行为会终止进程。该开关只安装信号观测，不增加定时检查、超时、重试或模型调用。

每次 dump 保存所有 Task 的 `cr_await` 链、cancelling 状态、`_fut_waiter`、asyncio/concurrent Future 状态与回调依赖、默认线程池队列和线程标识，以及会话锁、provider 同步锁、退休状态和未完成 attempt 元数据。它不读取用户文本 locals 或 Future 的实际结果。连续手动观测可区分一次正常的“concurrent Future 已完成、async Future 尚未递送”调度间隙与持续未推进的链路。

此前的观测工具测试还覆盖三个受控情形：同步线程被显式阻塞；取消后等待线程排空并正确保存 FAILED；人为扣住已完成线程结果的递送，在后台线程空闲时保留 STARTED。第三种用于验证观测能力；本次生产归因来自实际教师记录与 asyncio 错误日志。
