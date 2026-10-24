# Qwen3 研究引擎接入：有限 H200 验收

2026-10-24。用户现有 `/media/ssd1/qwen3-8b` 和 `/media/ssd1/qwen3-32b` 已分别通过研究引擎协议、四轮控制器完整 QA、GPU/Host 受控生命周期及并发观测检查。本批验证 **模型接入与已有机制在该配置下可执行**，不建立回答质量、吞吐、成本模型精度或一般负载收益。

## 固定实现与实际配置

部署源码为 `a87638a6f76be471e97e9855c9a23bbe358dad0f`，容器中333个tracked Git blobs逐一核验。先确认原研究服务空闲，仅停止其自有进程组71819；保留旧源码归档后，逆向旧补丁并应用新补丁 `b1b58e096c309eef1d7c10b1da3d29bec86503030e72f0f8e3badba9cf9c8a05`。固定SGLang仍为 `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`；原源码SHA、15文件编译、原生调度/传输观测接点和统一scope解析器副本检查通过。未下载权重、改CUDA kernel或重用旧7B成本表。

两模型在GPU0上串行使用独立30001服务：BF16权重/KV、TP/DP/PP1、FA3、Python FULL tree、context32768、page64、GPU pool16384、chunk8192、max_running_requests1、seed42。HiCache为 `write_through / kernel / layer_first`，TMA开关0，host size参数8GB；session radix和profiling显式开启。功能探针保持原research client的temperature0，**不是**此前能力参考的0.7/0.8/top_k20配方。

| 实际配置 | 8B | 32B |
|---|---:|---:|
| layers / KV heads / 显式head_dim | 36 / 8 / 128 | 64 / 8 / 128 |
| GPU容量tokens | 16384 | 16384 |
| Host容量tokens | 54272 | 30528 |
| BF16 K+V每token理论字节 | 147456 | 262144 |
| GPU池理论KV数据大小 | 2.25GiB | 4GiB |

容量来自实际allocator；字节由实际模型几何换算，未计对象/索引等开销，也不是峰值显存。同token数并非同物理内存。模型身份依旧是已核config、分片库存/字节及tokenizer/template的local metadata identity，未新增HF revision或权重内容证明。

## 已观察的合同与边界

- 两模型的smoke均验证了prepare幂等、真实完成回执、终态结果重放、重复quote相等、retire ACK与退役后拒绝。退役即时回收的GPU/Host slot均为零；撤销软引用不冒充物理释放。
- 每模型四轮真实controller运行依次为REBUILD、APPEND、REBUILD、APPEND。检查了实际执行消息重渲染后input IDs一致、此前完整QA按序保留、生成output IDs解码与原答案一致、相同业务request回放及retire outbox排空。输出cap128、合成小材料，不作QA质量评估。
- lifecycle对GPU/Host分别运行keep_old与retire_old配对。两模型、两层均观察到保住新分支128tokens、旧分支保留减少128tokens，共享前缀保留；退役当时占用不变，随后由原生eviction实现差异。这是明确施压的受控归因，不是自然工作负载平均收益。
- 每模型各有GPU-ready和Host-only三请求观测。全部主请求实际prefill为8192tokens，Host-only各请求实际恢复1024tokens；GPU-ready描述初始状态，其第三个请求也因排队期间资源变化从Host恢复1024tokens，不能把准备时驻留状态当永久命中保证。
- 四份观测报告均为合同检查通过、覆盖 **partial**：采到了1 running/2 waiting、等待请求未知match、正前缀命中；未采到已知match的waiting请求。原始状态与等待预算成员保留，不能把未覆盖写成已验证，亦不由此推断原生调度错误。请求输出128是该固定功能负载的长度控制，不是质量截断对照。

scope从实际启动到cold reset后的info持续保留；新模型未冒充旧7B指纹。采样切换、新模型成本/长度标定、正式B0/B6、真实任务P2/P3及完整资源压力矩阵均不在本次验收范围。

## 原始证据与恢复

[公共产物清单](../runs/remote-h200-qwen3-scope/public-export-manifest.json)含36文件（35 payload加清单），归档90,116字节，SHA256 `7466ded22745bef94fa245fb98de330d9228fe39fb7985edace19e7cf5582c7d`。保留两模型probe报告、配置、runtime、四轮原始账本/事件、源码部署记录与驱动；不含权重、私有日志或环境凭据。归档发生在全部探针结束后。

supervisor72723完成，两模型各5个probe进程退出码均为0。旧7B服务以新兼容补丁恢复为PID73118，随后独立health HTTP200、旧7B指纹及空闲metrics检查通过；启动期间11次尚未就绪的检查保留。重启身份为 `93df75e053464d6c986f258fc6050a66`，旧cache配置hash保持 `9e0b9573db2dd06720818bc763b3889b12a7c509fa8c0e516df36e12e533667d`。恢复launcher退出与真正健康分开记录。

本地驱动经过11项独立CPU失败/清理合同检查，源码说明在[独审目录](../runs/qwen3-scope-gpu-review/README.md)。[实际GPU产物独审](../runs/qwen3-scope-gpu-review/h200-v1/analysis.json)核验35个payload、333个Git blobs、8轮实际消息/分词/输出解码与SQLite提交，以及553份quote的原生预算公式，未发现已保留合同的矛盾。幂等回包由运行时probe断言验证，持久账本也只有4次生成/4次提交；重复回包本体没有单独导出，不把它写成离线逐字重放证据。

固定原生源码的running-capacity早期返回位于等待请求的前缀匹配之前；本次max_running_requests1与未见known-waiting相符。由于没有逐请求gate执行轨迹，这仍是有源码依据的解释，不能宣称唯一原因，也不为把partial改成complete而临时扩大实验。后续若需要该状态，另设与原生admission机制相符的有限负载。
