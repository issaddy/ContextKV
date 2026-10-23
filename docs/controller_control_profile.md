# 控制路径 CPU 诊断与局部优化

本次诊断使用已保存的真实 Cloud P2/c4/1024 输出上限运行 `9ad65448c01e442aac00b757bc17bcb3`，以及从其提交前状态冻结的 15 个案例。没有调用远端引擎或生成新答案。局部测量在 macOS 15.7.7/arm64、Python 3.12.13 上执行，不是 H200 服务器性能复测。

## 原始计时范围

41 轮的 `control_ms` 中位数为 231.32 ms、P95 为 436.29 ms。源码定义为 `(submitted − entered) + (committed_at − response_done)`：它包含候选构建、事件记录、SQLite/JSON、引擎 info/quote/prepare RPC 的等待，以及异步任务恢复执行前的调度等待；它不是纯 CPU 时间。它排除会话锁等待、入锁前请求哈希、模型生成、提交后事件及最终回复前额外的 Attempt 读取。

存档时间戳的 arrive→submit P95 为 349.48 ms，response→commit P95 为 94.87 ms；两者的分位数不能直接相加。arrive→submit 比 control 的提交前范围多出锁等待，本次 41 轮两种范围的差至多 0.009 ms。c4 时其他任务的同步候选、序列化、提交及回复后审计导出也可能延迟一个正在 await 的请求。原记录没有逐 RPC/同步阶段计时，不能据此把 436 ms 分配给某个远端操作。

## 精确重建候选的本地证据

用同一固定 Qwen2.5-7B-Instruct tokenizer revision、原始在线 query/contexts 和冻结 snapshot 重建 15 个案例。模型/template/tokenizer fingerprint 匹配，两个候选的 input IDs、消息、证据 spans、input hash 及可行性逐一与冻结结果相同。APPEND 最长为 11,217 tokens；最长候选对为 10,206 + 7,720 tokens。

每案例预热后测量 10 次，两候选共同构建的案例中位耗时范围为 7.71–20.11 ms，15 个案例中位耗时的中位数为 10.79 ms。独立 cProfile 的 150 次候选对构建共 1.964 s，其中 tokenizer `encode_batch` 为 1.708 s（约 87%），含 JSON 编码的 token hash 为 0.059 s（约 3%），全部 SHA256 调用为 0.004 s（约 0.2%），渲染后的正文检查为 0.014 s（约 0.7%）。这些证据不支持优先删除哈希或证据检查，也没有据此引入增量 tokenization 或会话缓存。

候选局部测量保存于 `runs/controller-control-profile/candidates-summary.json`，累计 profile 为 `candidates-cprofile.txt`，复现脚本为同目录 `candidates_profile.py`。

## 序列化改动的范围

真实快照的旧实现 profile 显示，重复解析固定 dataclass 注解，以及对每个 token 标量执行 `get_origin/get_args/is_dataclass`，是可避免的 CPU 开销。`serialization.py` 只做两项变更：固定 contract class 的类型注解缓存，以及内置 str/int/float/bool/None 的提前返回。缓存不保存会话、候选、答案或任何请求内容；没有新增配置或哈希。

专项回归先在旧实现通过，再验证新实现：canonical JSON 字节、嵌套 immutable contract 的值与容器类型、未知字段拒绝、固定 tuple 长度、ambiguous union 错误、原有不强制转换的 primitive decode 语义，以及 NaN/Infinity 输出拒绝保持一致。JSON 解析本身和 SQLite 事务没有改变。

同一 15 个真实 snapshot（52.8–196.6 KB）在临时 SQLite WAL、`synchronous=FULL` 下预热 1 次后测量 5 次。以每案例的均值等权汇总，部分存储链从 47.22 ms 降为 13.46 ms：

| 局部操作 | 旧实现 | 新实现 |
|---|---:|---:|
| snapshot loads | 5.75 ms | 1.13 ms |
| snapshot dumps | 1.80 ms | 0.91 ms |
| commit，不含新证据文件 | 17.93 ms | 5.17 ms |
| 最终 get_attempt，未计入上述存储链总和 | 4.08 ms | 0.88 ms |

存储链只测 read/begin/save/commit，排除网络、候选构建、新证据 fsync、新 epoch 分配、events 和并发干扰；用于填充结果大小的是上轮已提交答案，没有生成新答案。15 个 snapshot 与 30 个候选的新旧 JSON 字节及 typed object 全等。结果与可复现脚本保存于 `runs/controller-control-profile/store-summary.json`、`store_profile.py`；旧序列化实现备份为同目录 `serialization-before.py`。

## 暂不实施的存储与计时建议

`begin_attempt` 为检查版本而重新解码整个 snapshot；它可以保留事务检查，仅查询 `version,closed`。正常返回前再次读取已提交 Attempt，也可以研究复用已知提交结果，避免回复路径额外反序列化。`save_attempt` 每个阶段解码旧完整 Plan 并重写完整 Attempt，commit 则重复读取 Attempt、读取 snapshot，并持久化新 snapshot/history/Attempt；这是实际存储放大，但本轮没有拆表或改变恢复协议。新证据归档的 fsync 也未纳入局部存储微测，不能假定其成本为零。

优先在目标部署上重新测量优化后的完整控制路径，单独记录同步构建/存储段与 info/quote/prepare await 段，再评估剩余瓶颈。原始 `2 × P95 = 872.58 ms` 包含大量 APPEND/REBUILD 共有开销，不能把它直接解释为新增切换代价。局部 CPU 改善不能换算成远端控制 P95、GPU TTFT 或吞吐收益，也不作为手动降低门槛的依据。
