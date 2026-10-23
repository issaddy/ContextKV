# 控制路径 CPU 诊断与局部优化

本地诊断使用已保存的真实 Cloud P2/c4/1024 输出上限运行 `9ad65448c01e442aac00b757bc17bcb3`，以及从其提交前状态冻结的 15 个案例。局部测量在 macOS 15.7.7/arm64、Python 3.12.13 上执行，没有调用远端引擎或生成新答案。后文另列实际远端 before/after 复测，两类证据的范围分别解释。

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

远端完整路径的后续复测见下文；尚需单独记录同步构建/存储段与 info/quote/prepare await 段，才能继续归因剩余瓶颈。原始 `2 × P95 = 872.58 ms` 包含大量 APPEND/REBUILD 共有开销，不能把它直接解释为新增切换代价。局部 CPU 改善不能换算成远端控制 P95、GPU TTFT 或吞吐收益，也不作为手动降低门槛的依据。

## 实际 P2 闭环复测：存在历史与动作混杂

远端使用 expanded lookup、Q0、P2/c4、1024-token 输出上限分别执行旧/新 serializer，两次均完成相同 41 个 Cloud dev task。每次先运行专用服务 cold hook；记录证明 GPU/host unique tokens 均为零，且清空后的 restart ID 对应各自运行。实际 pool/layout/write policy 等配置相同。

逐项比较 manifest，配置仅 `state_dir` 不同，继而 config hash 不同；模型/tokenizer/template 指纹、成本 artifact SHA256、engine 配置（除 cold reset 产生的 restart ID）、数据、workload、检索和 Python/平台记录一致。按 `_manifest` 相同算法对本地 git `f5a7cd2` 与 `337dabe` 的全部 `controller/*.py` 重算源码聚合 hash，分别精确匹配远端 before/after manifest；两树的 controller 文件仅 `serialization.py` 不同。聚合 hash 证明源码集合对应，不被误写为 serializer 单文件 hash。

| P2 全部 41 轮，ms | 旧中位数 | 新中位数 | 旧 P95 | 新 P95 |
|---|---:|---:|---:|---:|
| control | 316.75 | 141.01 | 733.55 | 216.55 |
| 提交前 control（不含锁等待） | 215.96 | 106.88 | 635.44 | 179.08 |
| 响应后至 commit | 86.97 | 27.81 | 130.80 | 44.55 |
| complete（controller 到 commit） | 3576.43 | 3614.14 | 5499.34 | 5014.05 |
| runner 完整轮次（含返回及 audit） | 3607.89 | 3624.24 | 5553.70 | 5023.54 |

这组 before 与最初诊断的旧运行使用不同成本标定/执行配置，733.55 ms 和此前 436.29 ms 不是同一次基线。P2 本组 41/41 的当前 query 和检索 contexts 相同，但 audit 的完整 input IDs 仅 6/41 相同（14.6%），动作 32/41 相同（78.0%），output IDs/答案正文仅 1/41 相同（2.4%），规划压力档位仅 24/41 相同。生成历史、动作及缓存访问路径已发生传播变化，因此全组耗时差异不能作为 serializer 的严格逐轮因果效果，也不能证明语义质量一致。

6 个同输入案例包含全部五个首轮和一个第二轮，输入范围 2460–3385 tokens。它们的 control 中位数为 204.60→113.94 ms、响应后 commit 中位数为 36.70→11.97 ms；complete P95 为 3001.38→3059.07 ms。该小子集的生成输出和执行环境压力仍不完全一致，且不代表长会话。唯一同输入且同输出案例的 control 为 176.19→100.20 ms、complete 为 797.07→741.22 ms；单个案例不能外推。

为避免按优化后长度重新挑选样本，分层固定使用 before 的输入长度，并保留每层 after 长度：

| before 输入长度 | task 数 | 同输入数 | control 中位数，旧→新 | after 输入长度中位数 |
|---|---:|---:|---:|---:|
| <4096 | 6 | 6 | 204.60→113.94 ms | 2965.5 |
| 4096–8191 | 22 | 0 | 315.95→118.23 ms | 6524 |
| ≥8192 | 13 | 0 | 369.09→150.88 ms | 9419 |

这些分层仅描述同 task 的观测，后两层不是同长度或同输入的受控比较。完整逐 task、子集和分层结果在 `runs/controller-control-profile/hardware-p2-comparison.json`。分析脚本只读取本地 artifact，并校验 audit token 与 receipt hash、输出与 benchmark 一致：

```bash
PYTHONPATH=. python runs/controller-control-profile/compare_hardware_runs.py \
  --before runs/remote-h200-control/control-before \
  --after runs/remote-h200-control/control-after \
  --before-cold runs/remote-h200-control/cold-job-check.json \
  --after-cold runs/remote-h200-control/cold-control-after.json \
  --output runs/controller-control-profile/hardware-p2-comparison.json
```

## 实际 P1 固定 REBUILD 对照：全部输入匹配

为隔离 P2 生成历史及成本选择的传播变化，又使用 P1 官方参考历史和 `always_rebuild`，同样在 c4、1024 输出上限、两次 cold reset 下先运行旧 serializer、后运行新 serializer。两次均 41/41 成功；上述配置、引擎、源码集合及 cold 记录检查全部通过。相同 task 的完整 input IDs、当前 query/contexts、参考历史和 REBUILD 动作均为 41/41 一致。输入长度为 2460–4231 tokens；该集合没有长 APPEND 样本。P1 还执行参考历史同步，其控制流水线与 P2 不完全相同。

| P1 全部 41 轮，ms | 旧中位数 | 新中位数 | 旧 P95 | 新 P95 |
|---|---:|---:|---:|---:|
| control | 233.14 | 84.45 | 456.11 | 168.68 |
| 提交前 control（不含锁等待） | 176.63 | 66.06 | 407.40 | 151.48 |
| 响应后至 commit | 50.65 | 15.65 | 72.99 | 21.97 |
| complete（controller 到 commit） | 1906.01 | 1723.92 | 3127.59 | 2846.96 |
| runner 完整轮次（含返回及 audit） | 1934.87 | 1728.50 | 3149.13 | 2855.12 |

这次 41 个 task 的 control 均降低，完整轮次到 commit 有 35 个降低、6 个增加。只有 16/41 的 output IDs 和答案正文相同；全组平均输出长度分别为 292.56 和 292.66 tokens，均值接近不表示逐轮输出或质量一致。16 个同输入且同输出案例的补充结果如下：

| P1 同输入且同输出 16 轮，ms | 旧中位数 | 新中位数 | 旧 P95 | 新 P95 |
|---|---:|---:|---:|---:|
| control | 212.90 | 87.19 | 378.55 | 157.07 |
| 响应后至 commit | 53.96 | 16.75 | 71.01 | 20.91 |
| complete（controller 到 commit） | 1613.13 | 1499.42 | 2984.25 | 2841.48 |

输入长度分层中，<4096 的 38 轮 control 中位数为 224.29→82.14 ms；4096–8191 仅 3 轮，为 257.49→97.84 ms；≥8192 无样本。分层样本量及每轮变化完整保留在 JSON，不能据 3 轮估计稳定尾部改善。

本地字节/contract 等价及受控 CPU 微测，与本次固定 P1 输入下 control 降低的观测方向一致，支持保留这项局部实现优化。但远端仍是一次先 before、后 after 的有序实验，没有随机化、多次重复或运行顺序交叉；输出变化、调度和运行时间差仍可能影响端到端耗时。同输出子集按观测结果筛选，只用于补充描述，不能作为无偏效果估计。这里不声称语义质量一致、一般吞吐收益、长上下文收益或总体 P95 的统计显著改善。

```bash
PYTHONPATH=. python runs/controller-control-profile/compare_hardware_runs.py \
  --before runs/remote-h200-control/control-p1-before \
  --after runs/remote-h200-control/control-p1-after \
  --before-cold runs/remote-h200-control/cold-control-p1-before.json \
  --after-cold runs/remote-h200-control/cold-control-p1-after.json \
  --output runs/controller-control-profile/hardware-p1-comparison.json
```
