# Qwen3 同池真实运行审查

2026-10-24 独立核对的结论是：**保留 BGE 为当前重排序基座。** 在固定 Cloud dev、当前问题、同一 RRF 前 50、1024 输入上限条件下，Qwen3-Reranker-0.6B 的官方 qrels 指标下降。没有发现可见的候选错配、查询混入、打分方向颠倒或模板整合错误；这不是 Qwen3 在所有场景较差的结论，也尚未构成答案质量退化的证明。

原始结果：[Qwen run](../runs/remote-h200-newer/runs/cloud-dev-qwen3-rerank-samepool1024/manifest.json)、[冻结池](../runs/remote-h200-newer/runs/cloud-dev-qwen3-rerank-samepool1024/pools.jsonl)、[新排名](../runs/remote-h200-newer/runs/cloud-dev-qwen3-rerank-samepool1024/reranked.jsonl)。独立重算不调用项目的指标函数，使用原 qrels 直接计算 Recall/nDCG；[审查脚本](../runs/qwen3-rerank-audit/recompute.py)、[汇总](../runs/qwen3-rerank-audit/summary.json)、[逐轮结果](../runs/qwen3-rerank-audit/turns.jsonl)保留全部 41 轮及来源摘要。

## 数据与源码合同核对

- 原 offline retrieval 文件、原 manifest 和 qrels 的本地 SHA256 与 Qwen run 所记录来源一致；嵌入的原 manifest 与原文件解析结果相同。
- 41/41 的 query 与固定 `RAG.jsonl` 当轮用户文本逐字一致；task/conversation/turn 一致。每轮原 hybrid 前 50、BGE 的 rerank 输入、Qwen 的 rerank 输入和逐 pair 诊断顺序完全一致，新排名没有增加或丢失候选。
- 41/41 成功；2050 对输入全部未截断，实际长度 81–698 tokens。旧 BGE run 的触顶计数也为 0。因此这次差异不能归因于 1024 cap 截断丢失正文。
- 两个模型都记录 `cuda:0`、BF16、相同 torch 2.14.1 / Transformers 5.19.0；固定模型 revision 与各自计划一致。BGE batch=16、Qwen batch=8，不能据此做严格等批量性能比较。模型来源依旧是操作者提供的固定本地 snapshot，未在本次审查重新下载或校验全部权重。
- [`qwen3_reranker.py`](../adapters/qwen3_reranker.py)与[固定官方 README](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/fd9fb1d26c07223ced488065909faf522e29cc7d/README.md)的 instruction/prefix/suffix、左 padding、末位置 no/yes 顺序一致。运行记录 prefix 39、suffix 9 tokens，no=2152、yes=9693；分数范围约 0.00000963–0.999797，全部有限、非 0/1 饱和，降序及 ID 破同分排序正确。
- `logits_to_keep=1` 在已装 Qwen3 源码中只把末位置 hidden state 送入 lm_head；`use_cache=False` 不返回 KV。没有发现把 CausalLM 错当 BGE 分类头、调用生成、前后端模板不一致的问题。本轮是源码与 artifact 审查，没有额外 GPU 重跑来声称逐 bit 数值等价。

本地没有重新加载全部 72,442 个 passage；运行时原 passage 文件已按保存的 corpus manifest 校验，当前审查核对来源标识并阅读结果中实际保存的前五原文。qrels 和未来答案未参与当时的评分；下述标签与同分敏感性计算全部是离线审查。

## 总体与会话结果

共有 5 个会话、41 轮，39 轮有正例 qrels；缺标签的是 CocoaPods 会话第 6、7 轮，两方法均保留运行结果但不进入这两个质量均值。表中均值以有标签的轮次计算，不把 39 轮当成 39 个独立会话。

| 会话（ID 前缀） | 有标签轮数 | BGE R@5 | Qwen R@5 | BGE nDCG@5 | Qwen nDCG@5 |
|---|---:|---:|---:|---:|---:|
| 查询/存储 `1c041ce4` | 8 | 0.6125 | 0.3729 | 0.5600 | 0.3874 |
| WebSocket/电话 `b4614daa` | 8 | 0.5417 | 0.6667 | 0.4888 | 0.6182 |
| CocoaPods/Carthage `c01c8cf1` | 5 | 0.8667 | 0.8000 | 0.8751 | 0.8281 |
| VA/Container Registry `c6c3b02c` | 9 | 0.4259 | 0.1796 | 0.4265 | 0.1204 |
| Cloudant `d5b1e735` | 9 | 0.3333 | 0.2222 | 0.2491 | 0.2186 |
| 全部有标签轮次 | 39 | **0.5231** | **0.4085** | **0.4833** | **0.3907** |

Qwen 相对 BGE 的 Recall：3 轮提高、12 轮降低、24 轮相同；nDCG：10 轮提高、15 轮降低、14 轮相同。全部前 50 的平均 Recall 同为 0.7607，差异来自排序前五。VA/Registry 会话对总体 nDCG 差的贡献为 −0.07065，查询会话为 −0.03541；电话会话则抵消 +0.02653，不能概括成全领域一致退化。

同分不是主要解释。前五边界出现同分的轮数为 BGE 6、Qwen 10。保持非同分顺序不动，仅在完全相等分数内用 qrels 作离线最有利排列，Qwen R@5 上界仍只有 0.4214、nDCG@5 上界 0.4077，低于 BGE 现值。这个 oracle 上界仅用于诊断，没有应用到新排名。重复段落与数值精度都可能产生同分；本次未做 FP32 对照，不能把所有同分归因于 BF16。

## 六个可复核例子

这些例子按主要 nDCG 损失、标签与内容分歧以及一个改善例选择，属于定向来源阅读，不是全部答案的人工盲审。

1. **VA 第 5 轮，版本 3/4 有何差异：直接对比细节降序。** `ibmcld_01533-6329-8623` 和 `ibmcld_01535-6381-8675` 从 BGE 第 1/2 降至 Qwen 第 33/34。原文明确对比 v3 的新镜像/安全通知/每周触发，与 v4 首次推送建索引、查询时计算评估。Qwen 前四换成默认版本迁移、CLI/API/SDK 更新文档，其中前两段正文完全相同；前五均没有上述扫描触发对比。这是可定位的细节覆盖损失，但那些迁移文档仍与版本差异相关，不能称为全部无关。
2. **VA 第 7 轮，解释 Container Registry：定义被操作教程挤后。** `ibmcld_01408-7-1878` 的架构/服务定义由 1→24，`ibmcld_01447-7-2032` 的概览由 2→9。Qwen 前两名为两份 Registry CLI 段落，随后为 getting started、插件更新、全局 region。它更偏操作内容，qrels R@5 从 0.6→0；但第三段 getting started 仍直接定义私有 image registry，因此“0”不表示输入完全不能解释该产品。
3. **VA 第 1 轮，加密镜像是否扫描：重复正例的名次影响指标。** 两个包含“加密镜像不扫描”结论的来源从 BGE 第 1/2 变为 Qwen 第 6/3，R@5 从 1→0.5；Qwen 第 3 仍提供了直接否定答案。它把一般漏洞/扫描能力排得更高，降低了直接答案的排序，但不能仅由该 Recall 差断言该问题已不可回答。
4. **Cloudant 第 1 轮，IBM 是否提供 document databases：qrels 未覆盖相关替代材料。** 两个标注来源 `ibmcld_00513-7-2197`、`ibmcld_00474-7885-8455` 从 BGE 第 2/5 降至 Qwen 第 8/15，使 R@5 变为 0。实际 Qwen 前五包括 Cloudant 教程、documents 概览、平台服务说明、数据库组织与 Cloudant docs；其中 `ibmcld_09715-2612-4022` 直接说明 document-oriented DBaaS，`ibmcld_12889-3-1157` 说明 document-oriented NoSQL。此处不能把所有未标注段落判为无关；它揭示 qrels 与实际回答充分性之间的限制。
5. **查询会话第 4 轮，query API 细节：细分主题偏移。** 标注的 Discovery 默认 query 设置 `ibmcld_07094-7-1956` 从 1→8。Qwen 前四偏 query API release notes、DQL 和训练数据管理，第五是 VPC 文件共享 API；后者对该会话目标存在跨产品偏移。当前问题只有泛化的 query API 文本，未提供显式产品名，两个方法都受同一个 current-only 视角限制；其余 Discovery 资料并非无关。
6. **电话会话第 6 轮，安全连接与优化交互：Qwen 改善覆盖。** 安全连接段落 `ibmcld_16288-1733-3996` 从 BGE 第 13→Qwen 第 3，包含 secure trunking/SRTP 与 SIP authentication；交互优化段落 `ibmcld_16288-6287-8401` 保留在第 2。该轮 R@5 从 0.5→1，支持“模型取舍与具体任务有关”的判断。

## 成本与采用决定

已保存的重排阶段均值为 BGE 190.82 ms、Qwen 505.87 ms，中位数为 185.84/378.10 ms。Qwen 首两轮为 4276.00/1225.55 ms，剔除它们后的描述性均值为 390.74 ms；不删除原始点、不用剔除后的数值替换主报告。首轮较大符合初始化/预热影响的可能性，但本次没有把其具体来源测量分解。

BGE 的时间来自原 run 保存的总阶段减去共享 hybrid 阶段，Qwen 则在另一次 run 直接测量；batch 也不同。因此不能把均值比值包装为严格的 2.65 倍性能差，或据此宣称端到端生成速度。现有记录只显示这次 Qwen 运行没有同时获得更好的 qrels 指标与更低的观察开销。

当前决定是保留 BGE，把这份负结果作为真实基座筛选证据。该结论限定于 BGE-M3/BM25/RRF 产生的 50 候选、当前问题视角、官方默认 instruction 与 BF16；它不复现作者的 Qwen embedding top-100 全流程。本轮不新增模型、不改线上默认、不从少量坏例调整 qrels 或评分。后续若要宣称问答质量变化，应在实际 P3 答案上检查支持与完整性；本报告没有做该结论。
