# 共享前缀锁：一次原生准入机制验证

> 归档记录：保留当时的设计、结果与复用依据；其中“下一步”和执行计划不代表当前任务。当前工作仅以[实现清单](../implementation_plan.md)为准。

2026-10-24。**8 个 episode、32 个模型请求全部完成；独立原始审计没有发现输入、receipt、终态或来源绑定问题。** 6 个正式 episode 中，新观测区分了共享前缀已被 peer 锁住与无关前缀两种情况，旧标量近似把两种情况都判为需等完成。这是一个明确的准入状态缺口及其修正证据，**不是旧/新在线策略的性能对照、QA 质量结果或完整异质队列模型验证**。到此停止本 fixture，不扩大参数网格。

## 运行与独立核验范围

完整结果在 [shared-prefix-pin-v1](../../runs/remote-h200-shared-pin/runs/shared-prefix-pin-v1)，导入归档 SHA256 `9afd9a0ac11e234fd228573972c63d177a2bda31e0b226553d563b1b3dfbd50a`。项目部署 `16e4ef9a13ed1f2974c26f41a9ac704c7b0adfd5`，记录 323 个 tracked blobs，完整 files 字典与固定 Git 树逐项一致；固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`。独立审计将实际记录的 8 个组件 SHA 与该 revision 的 Git blob 或计划固定的 ignored helper 字节核对，全部一致。该绑定是启动观测，不冒充运行隔离或逐请求远程认证。

固定 Qwen2.5-7B-Instruct revision、tokenizer/template/namespace 均沿原 fingerprint；GPU 16384、host 279040 slots，page64、write_through、kernel/layer_first、TMA0、host admission off、chunk8192、FCFS/maxrunning8。原生 scheduler 没有被替换；新增字段只是观测，old/new 等待函数都在同一个实际 quote 上离线式分类。

导出 prep 的原 inputs.json SHA 为 `bea7696c64b6d943172bab27855ae109fdf3147c4e6f9d830d8d6e79b2eb30cf`，与计划声明一致；run 内副本经 dumps 规范化，SHA 为 `4ef9b0585bc9e070ce09a02817ba4d37bf69a5bc13980a195fe23a17a4b2b577`，两者 JSON 语义逐项相同，不能称文件字节相同。

输入是冻结开发 token ID 的受控合成分支，不是原 QA 问题。X/Y 各 4096 tokens，在第一 token 分开；shared peer 与候选共享 X 的完整 4096，unrelated peer 使用 Y。peer 为 6144、候选为 12288，shared peer 与候选在第 4097 token 开始不同。每 episode 独立 cold，然后两个 seed 各输出 1 token，settle 后原生 pool 为 8192，分别验证两个 GPU 前缀；peer/candidate 各输出 512、ignore_eos。全部 32 次输出数量正确，16 seed 与 16 target 分母完整。8 次 candidate 输入和输出 ID 分别全同；peer 输出并非全同，因此也不能包装为等答案的速度对照。

peer 先启动；观测到其已生成至少 1 token、等待队列为空后，再取一次候选 quote 并提交候选。实际最终 quote 的 peer 已生成 2 或 3 tokens。这个触发依赖运行状态，是**准入快照实验**，不具备固定外生到达的性能公平性。

[独立 analyze.py](../../runs/shared-prefix-pin-review/analyze.py) 不读取 runner 的 `classification` 或 `assessment` 作为结论，直接检查：

- 原计划顺序、2 个 warmup / 6 个 formal、全部输入 IDs/hash、输出预算及正常固定长度终止；
- 每次请求唯一 RID、prepared epoch/restart、receipt、原始 terminal event 一一对应，完整 generation/session ref；
- 非负实际 D/H/U 工作覆盖、原生 receive/forward/first-token/completion 单调顺序与 server elapsed 一致；
- seed 准备形状、候选纯 GPU 无 pending、单一已 decode peer、空等待队列、同 epoch quote 与原生 clock；
- 全部请求 drain 后才 retirement，所有 32 个退休 ACK 匹配；原生 cache 保留不被误当退休失败或要求 pool 归零。

主结果 [analysis.json](../../runs/shared-prefix-pin-review/analysis.json) 的 global / source / episode issues 均为空。6 项 CPU 合同另覆盖严格容量边界、首次拒绝不能被后来接受覆盖、缺失请求保留、runner 摘要篡改不改变重算，以及“容量通过但 forward 晚于 peer 完成”不能算无需等待的反例。

## 从 quote 重算，而非把总可逐出量当候选新增锁定量

令 `A=min(capacity, free+evictable)`，`R` 为原生正在运行请求的输出预算，`I` 为候选临时 pin 会新锁住的 GPU tokens。这个固定 FULL 请求的总需求为：

```text
G = (12288 − 4096) + min(512, native_clip) + 64 = 8768
paged_input = 8192
initial_fit = G < A − R − I
              且 paged_input <= min(A − I, A − R − I)
```

原生总需求使用严格 `<`，页分配条件使用 `<=`。8 个 quote 的 `A` 均为 10176。旧近似用 `min(ready_prefix, all_evictable)=4096`，不知道候选前缀是否已被其他请求 hard-lock；新字段只计连续 GPU 命中范围内 FULL lock_ref 为 0 的 matched tokens，因此 shared 的 `I=0`、unrelated 的 `I=4096`。它不是给所有 prefix 免去容量，也没有放松 native admission。

每个首容量 gate 的实际需求都为 8768，较 quote 推算差 0；原生 `total_offset` 比 quote 的运行预算少 2 tokens，因此剩余额增加 2。这保留了 peer 继续 decode 引起的状态变化，不假定 quote 与实际 gate 为原子同一状态。

## 全部正式结果

下面的时间是 **candidate 首次 forward 减 peer completion**，两者同属原生 `perf_counter` 时钟。负数只证明 candidate 在该 peer 完成前开始，不表示相对某个旧策略加速了这些毫秒。

| repetition / arm | quote 旧剩余 / 新剩余 tokens | 旧 / 新 initial fit | 首 native 容量 gate | forward − peer complete（ms） |
| --- | ---: | --- | --- | ---: |
| 0 / unrelated | 5570 / 5570 | false / false | rejected | +1.655 |
| 0 / shared | 5570 / 9666 | false / true | accepted | −2697.054 |
| 1 / shared | 5570 / 9666 | false / true | accepted | −2693.356 |
| 1 / unrelated | 5570 / 5570 | false / false | rejected | +1.733 |
| 2 / unrelated | 5571 / 5571 | false / false | rejected | +1.794 |
| 2 / shared | 5570 / 9666 | false / true | accepted | −2694.205 |

三个 shared 的实际 candidate outcomes 均为 `accepted:OTHER` 一次；三个 unrelated 均先有两次 `not_accepted:NO_TOKEN`，最后 `accepted:OTHER` 一次。`OTHER` 是成功加入 can_run_list 后对后续容量的返回状态，不能当请求失败。所有候选最终工作都是 D=4096、H=0、U=8192；peer 是 D=4096、H=0、U=2048。

关键结论来自两个不同证据：新 quote 在这 6 个正式快照上的初始 fit 与首 native 容量 gate 一致；同时，三次 shared 的实际 forward 都早于 peer completion，确实反驳了旧近似在这些状态下“必须等这个 peer 完成”的判断。仅容量检查通过本身不足以推出后一条，因为原生后续仍有其他门控。

## Warmup、漂移与解释边界

两次 warmup 单列，不加入 6 个 formal 分母：shared 旧/新剩余为 5571/9667、false/true、首容量 accepted，forward 相对 peer completion 为 −2798.807 ms；unrelated 为 5570/5570、false/false、首容量 rejected、+1.647 ms。方向相同，不因此计算置信区间或总体准确率。

8 次客户端 quote 往返为 5.431–10.611 ms；quote 返回到实际 candidate submit 为 0.162–0.211 ms。quote 的服务端观测为 `monotonic_ns`，gate 为 `perf_counter`；本次记录未单独捕获两者时钟映射，所以审计保留原始数值、客户端间隔和上述 token 预算漂移，不默默相减成“精确 quote→gate 延迟”。同原生时钟的 forward/peer completion 比较不受此限制。

新字段仅用于 `native_budget`、无 host/mixed restore/pending、waiting 为空时的首次 fit；后续解锁和排队推进仍沿旧 proxy。此实验没有验证复杂共享图、虚拟排队后状态、DMA、抢占重算、在线质量或外部性，也没有把人为正 service callbacks 的返回毫秒当硬件预测。两个函数没有分别驱动调度决策，故不能根据这份数据宣布新策略的吞吐或 P95 收益。

专用 fixture/restore 的返回码均为 0，supervisor 保存 `server_returncode=-9`，不据此猜测具体 shutdown 原因。恢复后的第一次 health 探针在加载中连接拒绝，后续 HTTP 200；`restored-service.json` 保留这段过程，没有重启或重跑实验。以上恢复记录与生成完整性分开陈述。

本次证据足以接受“候选已共享锁与全局可逐出量不能混为一谈”这一具体修正。下一步回到完整 22 会话 / 159 轮质量基线与来源规划机制，而不是继续围绕此合成布局搜索更有利的参数。
