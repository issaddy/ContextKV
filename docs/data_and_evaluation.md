# 数据、实验对比与验收结论

当前工程已完成 A/B/C 联合运行与必要回归。最新对照显示：简洁回答协议消除了本批输出截断，控制链路重复工作明显减少，稳定证据追问中追加具有实际收益；六会话整体尚未优于同基座强原生参照。这里集中维护结果，不再另存逐轮实验纪要。

## 当前主评价口径（指标版本2）

主报告固定为：**回答质量、响应延迟、时延达标服务能力、资源效率**。完整定义见[项目范围](project_scope.md#核心指标与判断方式)。失败/超时/截断、引用可见性/范围与状态一致性作为工程验收项；Prefill、命中、传输、动作分布和控制开销留在JSON解释机制。

当前HTTP对外完整返回；`bench.report`优先复用原件中保存的返回方式/SLO，旧原件未声明时默认`response_mode=buffered`：完整响应采用会话等权均值和P95，另列不含用户停顿的会话累计系统等待。内部引擎SSE的TTFT/TPOT不被当成外部HTTP体验。现有流水线时间仍不包含未采集的外部HTTP网络区间。

Goodput改为`latency_slo_goodput_requests_per_second`，排除失败、截断、零输出及终态不明的完成请求。它只证明正常终态和时延达标，语义质量独立报告。缓冲返回没有已声明的完整响应期限时，goodput为`null`而非0；不按观察结果补选期限。旧的`slo_goodput_requests_per_second`留在旧原件中，新报告不再生成这一含糊字段。

资源主指标是实际采样GPU/Host唯一槽位积分/非截断完成轮次，分母使用整批完成（含排空），不是测量窗口内goodput计数；采样区间包含原记录的收尾。缺少遥测或事件截断时不生成归一效率值。峰值、原始积分和传输明细保留于JSON；槽位不等于整个进程显存。质量不合格的回答不能凭这个数值证明资源优化成功。

已有原件可直接离线复算：

```bash
python -m bench.report SAVED-RUN.json --output runs/metrics-review/summary.json \
  --quality SAVED-QUALITY.jsonl
```

`--quality`可省略，缺失时标为未评估。只有运行前已有明确的完整响应期限，才加`--complete-slo-seconds D`。真实流式交付使用`--response-mode streaming --ttft-slo-seconds F --tpot-slo-seconds P`；该开关声明计量合同，不修改服务行为或补造首字观测。矩阵在根节点声明`response_mode`，`slo`支持`complete_seconds`、`ttft_seconds`、`tpot_seconds`，由同一入口应用于所有方法并纳入既有冻结来源。

本次只用保存的120轮和既有评分重汇总，产物在`runs/mainline-delivery/metrics-review/`。原生成、原评分和原报告不覆盖；没有新增GPU生成或评分调用。本批未预声明缓冲返回的完整响应期限，因此新的服务goodput保持未配置，也不由单一并发点推断最大承载能力。

## 最新共同对照

运行源码为 `8aeeb5b2abc9b6029eba7da691886165e2a2a3e9`，固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b`，H200 / Qwen3-8B / BF16。GPU 池 16384、Host 实际池 54272 槽、page64、FA3、chunk8192、FCFS、write_through。共同输出上限1024、非思考采样 temperature0.7 / top_p0.8 / top_k20，正常 EOS。

负载为原始 MTRAG dev 的六会话、40轮，P2、并发2、think time0.2s、一次重复。保留原四域中位规模会话，再按输入证据的相邻重叠度选择两个稳定追问会话；没有按答案或耗时挑选。原 test 划分不变。本次是开发诊断，未运行候选搜索或新标定。

- **Native**：clean stock SGLang B1，完整重建，共享原生 session/radix/HiCache。
- **Simple**：相同研究引擎与生命周期，固定8192-token阈值。
- **ContextKV**：实测成本＋`native_budget`，按预计完整响应选择 A/R。

三者共用必要证据、实际历史规则、简洁提示和本轮引用范围。各方法保留自己的真实答案；后续输入和回答长度可以因此不同。

| 指标 | Native | Simple | ContextKV |
|---|---:|---:|---:|
| 成功 / 计划轮次 | 40/40 | 40/40 | 40/40 |
| 完整响应，会话等权均值 / s | 1.187 | 1.265 | 1.258 |
| 完整响应 P95 / s | 3.276 | 3.754 | 3.998 |
| 输出截断 | 0 | 0 | 0 |
| 当前不可见 ID 引用的回答数 | 1 | 0 | 0 |
| 正文可见、但超本轮范围的引用回答数 | 0 | 0 | 1 |

内部引擎TTFT均值为Native0.119s、Simple0.145s、ContextKV0.138s，作为机制诊断保留。原闭环完成吞吐为1.220/1.197/1.201请求每秒，不作为缓冲服务的SLO goodput或最大容量。

ContextKV 完整响应比 Native 高约6.0%，比 Simple 低约0.6%；单次六会话不足以作稳定优胜或统计质量非劣结论。33/34 个双候选轮次按成本选择，1次因支持域使用简单回退。120条共同问题/检索材料、各自实际历史、可见正文、每个控制器40个提交状态及退役清理核对无异常。

## 追加的收益条件

| 会话 | 场景与证据重叠 | Native / s | Simple / s | ContextKV / s | ContextKV 动作 |
|---|---|---:|---:|---:|---|
| `c01c8c…` | SDK连续追问，相邻重叠约83% | 1.101 | 0.925 | 1.051 | A4/R3 |
| `5e79e4…` | 许可年龄澄清，相邻重叠约87% | 0.804 | 1.056 | 1.023 | A1/R3 |

第一会话四次追加实际 Prefill 为434、425、562、290 tokens，客户端 TTFT 约50–58ms，完整会话均值比 Native 低约4.5%。Simple 在该会话更快，且输出更少；不能将这些闭环差值全部归因于缓存。

第二会话虽然重叠更高，ContextKV 输出581 tokens、Native460，且其中两次选择重建，整体未获益。**证据重叠是机会，不是收益保证**：还需实际 GPU-ready 前缀、新增后缀小、无明显容量等待，且收益覆盖更长上下文、Decode 和控制成本。证据切换或旧材料增长时，及时重建仍有价值。原生自身也会复用前缀，应保留这一强参照。

## 资源与生命周期

| 主要资源效率（非截断完成40轮/方法） | Simple | ContextKV |
|---|---:|---:|
| GPU唯一KV占用积分 / 轮（slot·s） | 11688.84 | 11691.59 |
| Host唯一KV占用积分 / 轮（slot·s） | 33151.14 | 34864.14 |

诊断原件保留：实际Prefill为94753/115748 tokens，GPU积分为467553/467664 slot·s，Host积分为1326046/1394565 slot·s，观测峰值均为16384/54272；完成D2H为14957936640/18043895808 bytes，H2D均为0。相同模型和槽位大小下才能直接比较；这些诊断量不单独决定胜负。

两种控制器退役 outbox 均清空，采样未截断，终态无 pending transfer。当前 ContextKV 比 Simple 更多重建，Prefill、Host积分和D2H未改善，因此不宣称资源全面更优。Native 没有同口径全局唯一池积分，不能用逻辑长度或缓存命中补造该列。退役 ACK 代表解除引用保护，不代表立即释放物理 KV。

## 重复开销与回答完整性修正

同一容器对29份保存的真实状态进行 CPU 回放：

| 路径 | 修正前 / ms | 修正后 / ms | 等价核验 |
|---|---:|---:|---|
| 持久化/状态处理均值 | 40.91 | 20.14 | 提交状态逐项相同 |
| 50候选成本批次中位数 | 403.02 | 200.21 | 所有 CostEstimate 字段相同 |
| 候选构造与分词均值 | 22.34 | 15.49 | 完整文本、IDs、spans 相同 |

计划与 epoch、prepare ACK 与提交标记各合并一次事务；外部调用前的持久化屏障保留。decode几何与请求内共同分词段复用，不增加内容缓存或重复哈希。

前一版四会话29轮的 ContextKV 完整响应均值3.632s、9条截断；当前同四会话子集为约1.368s、0条截断。共同简洁提示、控制优化与策略变化一起发生，且原生同样大幅变快，因此这只能说明工程版本进步，不能将全部差值算作 A/R 算法的独立贡献。

回答协议优先完成问题、保留条件、行内引用、避免重复结论与来源列表。实际历史仍完整保存。当前“引用可见性”和“本轮引用资格”分开记录；剩余范围违规与语义错误如实保留，不通过删引用、修改历史或隐藏重试消除统计问题。

## 语义质量检查

复用既有 `bench.judge`，以已有 Qwen3-32B 对120条已保存答案及完整可见原文作一次离线诊断。共同冻结 prompt、非思考、贪心、1536输出上限；显式检查必要内容覆盖。它与生成模型同属 Qwen3，只提供模型辅助诊断，不作为独立/人工质量验收。120/120次均正常完成。原始均分如下，不能用其小数差直接宣布质量排名。

| 维度 | Native | Simple | ContextKV |
|---|---:|---:|---:|
| correctness | 0.948 | 0.953 | 0.928 |
| groundedness | 0.964 | 0.969 | 0.949 |
| answerability | 0.988 | 0.995 | 0.978 |
| conditions_versions | 0.990 | 0.989 | 0.985 |
| citations | 0.968 | 0.961 | 0.955 |

保存原文审查发现：儿童活动安全问题有可用的部分材料，ContextKV却整体拒答；评审器又把“没有补充推断”同时罚成groundedness/citations零分，混淆了完整性与忠实性。另一条越范围引用实际有3个ID，评审理由只指出1个；“版本命令查询最新发布”的依据不足也未被可靠识别。原分数保留，不改分、不换judge重跑。故本次质量结论是发现了具体不足，尚不能建立独立非劣或优胜。详细见 `runs/mainline-delivery/delivery-quality-review.json`。

质量包含 correctness、groundedness、answerability、conditions_versions、citations。自然 EOS、合法编号、原文命中和高检索召回均不能代替语义支持。已有人工盲审/来源绑定/会话 bootstrap 接口见[质量验收](guides/quality_acceptance.md)，本次不伪造质量资格或冻结赢家。

## 数据与协议

固定数据为 [IBM MTRAG](https://github.com/IBM/mt-rag-benchmark/tree/53fef6f5917c7e10c436f697e69a955dbc795d81) 的 `mtrag-human/generation_tasks/RAG.jsonl`：110会话、842任务、每题五个证据块，文件SHA256为 `5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873`。会话按域和seed42分成22dev/88test，清单位于 `configs/mtrag_split_manifest.json`。

| 协议 | 历史 | 当前证据 |
|---|---|---|
| P1 | 官方参考历史 | 固定 top5 |
| P2 | 本方法已提交的实际回答 | 固定 top5 |
| P3 | 本方法已提交的实际回答 | 真实检索 |

P2预检索可能依赖参考历史，不能称完整在线RAG；P2/P3未来用户问题仍是固定脚本。线上只接收当前白名单问题、已发生历史和证据，reference、qrels及未来轮次不进入控制器。失败不补造答案，后续依赖轮次保留为未完成。

BM25S、BGE、RRF和重排使用固定索引/原文；manifest在加载边界核对，查询不重复哈希正文。P3包含检索排队和执行，不能与P2直接混合比较时延。缺qrels的任务显式列出，不当作零分。

完整响应取到达至完整响应/提交较晚者；TTFT和TPOT来自客户端实际SSE chunk，非流式或恢复重放保持未知。服务端排队/Prefill/Decode及CUDA传输各用自己的时钟，不重复累加。当前goodput是非截断、有终态且时延达标的请求吞吐，语义质量仍独立判断。

## 复现与证据位置

- 正常运行使用[8B成本配置](../configs/qwen3_8b_cost.yaml)和[部署步骤](deployment.md)，原测量artifact已随仓库提供。
- 当前原始对照在容器 `runs/mainline-delivery/delivery-run/`，本地导入副本在 `runs/mainline-delivery/delivery-evidence/delivery-run/`；每方法保留summary、audit、原始输出、源码和服务器状态。
- 当前原始包为 `runs/mainline-delivery/delivery-evidence.tar.xz`，SHA256 `dde5b6adbeeee8b9e81a17e2d089cddd3466c65b1943bee2fc09386ee516f4ee`。包内有原始矩阵、输入选择和split；源目录相对项目根。
- 对照入口为 `python -m bench.b6_joint_sweep --matrix runs/mainline-delivery/delivery-matrix.json --output runs/NEW-DELIVERY-RUN --execute`，父进程需 `RAYON_NUM_THREADS=4`。按[owned入口](guides/b6_owned_matrix.md)准备固定checkouts；已有结果可直接读取，不需要重复生成。
- `runs/mainline-delivery/summarize_delivery.py` 从原件复算汇总与历史核对；`delivery-comparison.json` 为当前汇总。评分和CPU回放单独保存，不改写原运行身份。
- Linux完整回归1539 passed、8 subtests、2个训练/导出opt-in跳过；真实Qwen3 tokenizer启用。保存答案审阅器的后续接线另通过20项检查。

## 已有工作的复用与边界

历史检索对照支持保留 BM25/BGE/RRF/重排，而不是因较新模型名称自动替换；局部召回改善同时伴随执行成本增加，未据此宣称最终答案更好。表格、SDK原文/依赖、Doc2Dial数据适配和Qwen3 HTTP正常路径已有真实或低成本合同证据，相关接口继续复用。

历史Host准入验证同时发现竞争下的保留价值与自身回访重算代价，当前仍默认关闭。旧模型、旧提示、旧资源或训练产物不充当本次Qwen3质量/性能证据。原始结果保留于对应 `runs/`，已删除的历史页面可在Git提交 `8aeeb5b` 及更早版本查看。

当前可交付的是完整可运行工程、有效的控制开销修正和具有条件的追加机制；“目标场景整体持续优于强原生”仍未得到充分支持。下一步有价值的投入应针对必要内容/来源约束、短回答下的固定成本、输出长度先验与实际资源竞争，而非重复标定、阈值搜索或更多评分工具。
