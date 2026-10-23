# 研究控制器硬件 dev 矩阵

`configs/matrix-hardware-dev.yaml` 是后续实跑入口，不是已完成实验或完整 B6 结论。它只包含 patched SGLang 上的控制器方法，固定 P2、并发 4、seed 42、1024-token 输出上限与相同默认 SYSTEM_PROMPT（Q0）。计划包含 11 个 dev job，冻结后有 5 个匹配消融 job。原生 B0/B1 必须另行使用未补丁服务器、真实原生 manifest 和各自的冷态准备流程；不能把本例的研究服务 cold hook 用到原生服务器。当前示例没有质量准入表，也没有任何 `b6_eligible:true`。

从仓库根目录运行。先核对并按实际环境修改两个 YAML 中的数据、模型、本地 lookup artifact 和专用服务路径。成本 artifact 显式使用 `runs/remote-h200-expanded/cost-dev-expanded-lookup.json`；1024 是生成上限，artifact 的历史长度先验和已有 Decode 标定范围并不会因此扩展。成本预测超出支持域或配置不匹配时仍记录既有回退，不能解释为成本模型收益。两份配置都不是全局默认配置。

```bash
# 仅生成计划，不连接推理服务，不生成答案。
python -m bench.experiments plan --config configs/matrix-hardware-dev.yaml \
  --output runs/hardware-controller-dev-plan.json

# 显式执行一次 cold 准备：会清空指定专用研究服务的 KV 与会话状态。
# 模型/tokenizer/template fingerprint 从固定 revision 的 config 构造并与服务校验。
python -m scripts.prepare_cold_job --base-url http://127.0.0.1:30000 \
  --config configs/hardware_dev_controller.yaml --output runs/cold-job-check.json

# 核对实际服务与材料后才执行；每个 job 前自动运行同一个 cold hook。
python -m bench.experiments tune --config configs/matrix-hardware-dev.yaml \
  --output runs/hardware-controller-dev

# 继承冻结的 Ours 参数后运行匹配消融，不重新搜索消融参数。
python -m bench.experiments ablate-dev --config configs/matrix-hardware-dev.yaml \
  --frozen runs/hardware-controller-dev/frozen.json --output runs/hardware-controller-dev-ablations
```

`scripts.prepare_cold_job` 要求显式 `--base-url`、`--config`，并仅接受 `info.profiling_enabled=true` 的固定补丁 SGLang 服务。服务需已由操作者独立配置并启用 `CONTEXTKV_ENABLE_PROFILING=1`。脚本不启动或重配服务，不生成模型答案，不预热生成。只有 `ENGINE_BUSY` 使用现有 profiler 的短重试（最多 20 次、间隔 50 ms）；身份、配置、传输等其他错误直接失败。

脚本记录 reset 前后的真实 pool、缓存配置、运行身份、活动/排队请求数以及原始 reset 应答。清缓存后再次读取 metrics，只有 GPU/host 的 unique token 数和请求数均为零才成功退出；reset ACK 本身不被当作清空证明。stdout 由既有 `before_job` 保存到每个 job 的 `before-job.json`，钩子失败则不执行该 job 的生成。独立运行可用 `--output` 额外保存 JSON。冷态范围是 KV/会话状态，不代表 CUDA kernel、JIT 或进程初始化状态重置；已有服务的启动与暖机条件仍应单独记录。

调参方法均给两个候选：B3 的 4096/8192-token 阈值、B3_periodic 的 2/4 轮周期、B5 的 .25/.75 GPU-ready 比例，以及 Ours 的两个切换门槛。B2、A_only 和 B4 固定配置；B4 是本项目的 ContextPilot 排序/注释适配，并非完整外部系统复现。去退役（noC）、去 host/decode/control 成本项、去滞回均沿用 `ablation_of: Ours`，从冻结 winner 继承其余参数。

`tune` 产生的冻结结果在本例中只表示 dev 完成轮时延选优；没有语义质量非劣效结论，也不选出 B6。后续应先完成独立 dev 质量审核，再按既有 `quality_eligible`/`freeze` 流程显式冻结质量合格配置。完整强基线结论还需要原生 B0/B1、其资源与写策略对齐及独立测试集评估。
