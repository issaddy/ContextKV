# 八种冻结查询的真实检索开发验证

2026-10-24。8 方法 × 41 个问题全部检索成功。固定 BGE 重排后的开发集 R@5 以 user_topic 的 .54872 最高；ConvSearch 为 .27009，但它的候选 50 召回率 .81026 高于 current 的 .76068。**这批证据指向召回查询与重排查询的分工问题，不支持把可解析重写或合法 selector ID 当作更好的检索。** 本轮没有生成答案，不作质量准入或在线性能结论。

## 绑定与复算

原始结果在 `runs/remote-h200-frozen-query-retrieval/runs/frozen-query-retrieval-v1/`；独立脚本和完整逐轮分析为 `runs/frozen-query-retrieval-review/analyze.py`、`analysis.json`。脚本先验证完整排名，再读取唯一固定 Cloud `dev.tsv`；未读取 test。核验内容为：

- 328 行分别与冻结 bundle 的方法、原问题、实际查询、生产者结果/失败状态逐字段一致；同 5 个旧 off 会话的完整实际历史，未调用新的 selector/CQR/答案模型。
- 固定源码部署 `20fef314241c60530775a41f07fdba15c40752fd`，三份运行时核心源码 SHA 与该 Git 树相同；独立 helper SHA 为 `96a7d6319312876b6e0208f8136aaf737b9f3692128f37fcb4f2ac412dc42de6`。之后提交的失败 trace 修复不在本次硬件运行中。
- 同一 72442 段语料/index，corpus SHA `04fb31f7d4301aa2fd3d12c161cf06265a9bbf9f41e56116d90637f9d053afad`；BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`，BGE-reranker-v2-m3 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`，BF16/GPU0/batch16、encoder 8192、reranker 1024。模型 revision 延续既有本地快照声明/下载记录，不凭一个 manifest 字段声称权重重新鉴证。
- 从每条 raw BM25/dense 排名重算等权 RRF（k=60），再从实际 50 个分类分数重排，328 行的完整 ID 顺序及分数全部精确相等；重复出现的 top5 原文也一致。每种方法最多重排 50 段，双路没有扩大总重排预算。
- 独立用标准 Recall/nDCG 公式逐条复算，与原 summary 全部一致。每方法 41 行均保留，其中 39 行有正相关 qrels；`c01c8cf11437e6bb3bc93efac26528c2<::>6/7` 缺标签，不从执行分母删除，也不伪记为零质量。

current 为单路原问题；user_topic 有单路与 current/topic RRF 两种；四个 selector 方法均为 current/保存查询双路；ConvSearch 为作者完整 rewrite 单路。所有生产者失败/歧义/首轮跳过仍使用原 current 查询。v1 source/unverified 分别保留 4/7 个失败，v2 分别 0/1 个失败；两版 unverified 各有 1 个 ambiguous，每种 selector 各 5 个首轮 unchanged。不能把后续检索成功覆盖上游失败。

## 总体排名与会话差异

下表均为同 39 个带标注问题的逐轮均值。RRF R@5 是重排前结果；候选 R@50 是实际提交给 reranker 的 50 段，重排只能改变顺序，不能增加这一集合的召回。

| 方法 | RRF R@5 | 候选 R@50 | 重排 R@5 | 重排 nDCG@5 |
|---|---:|---:|---:|---:|
| current | .38974 | .76068 | .52308 | .48327 |
| user_topic 单路 | .39829 | .74786 | .54872 | .51493 |
| user_topic 双路 | .42393 | .76068 | .54872 | .51493 |
| v1 source | .41410 | .75299 | .48162 | .42306 |
| v1 unverified | .36197 | .77094 | .41410 | .37730 |
| v2 source | .40556 | .74444 | .45171 | .40306 |
| v2 unverified | .38333 | .76239 | .40983 | .37337 |
| ConvSearch | .37778 | .81026 | .27009 | .26741 |

按会话列出的重排 R@5 显示下降不限于单一主题。列名为 conversation_id 前缀，标注轮数依次为 8、8、5、9、9；全 ID 和每轮值见分析 JSON。

| 方法 | 1c041 | b4614 | c01c8 | c6c3b | d5b1e |
|---|---:|---:|---:|---:|---:|
| current | .6125 | .5417 | .8667 | .4259 | .3333 |
| user_topic 单路/双路 | .6125 | .5417 | .8667 | .4259 | .4444 |
| v1 source | .4875 | .4479 | .8000 | .4037 | .4074 |
| v1 unverified | .5750 | .3854 | .7333 | .2370 | .2963 |
| v2 source | .4458 | .3854 | .8000 | .3667 | .4074 |
| v2 unverified | .5750 | .3229 | .8000 | .2370 | .2963 |
| ConvSearch | .4063 | .0938 | .5333 | .1889 | .2407 |

current→user_topic 单路只改变 4 条查询，带标注轮中 R@5 为 1 升、38 同、0 降；SDK 问题 `d5…t9` 的两篇标注段从 rank18/23 提升到 rank2/1。单路与双路所有带标注轮的 R@5/nDCG@5 相同，但并不意味着完整候选集合或 top5 全部相同；双路补回一些候选，未体现为这批标签的 top5 增益。

## 枚举结构恢复不等于语义修复

v1→v2 source 恢复了原先失败回退的 4 条查询，R@5 为 0 升、1 同、3 降；unverified 恢复 6 条，1 升、4 同、1 降。这是相同冻结问题、相同资源下实际新查询的排名结果，不能解释为“无效 JSON 本来更好”；结构正确性仍应保留，语义选择需要独立改善。

| 具体变化 | 原始排名证据 |
|---|---|
| source `1c041…t7`：创建 schema/table 增加 `Sort; Storage Space` | `ibmcld_16613-0-1165` 从原 RRF rank25、重排 rank5 变为 RRF rank55，未进入重排 50；R@5 和 R@50 都由 1 降至 2/3。这里先发生候选预算挤出。 |
| source `b4614…t8`：通话/转接失败追加 phone-number-limit 旧问题与 `Phone Connection; Include` | 两篇相关段仍在 50 内；`ibmcld_16288-10521-12298` 原重排 rank4，新增查询后 rank11，R@5 从 1 降至 .5。unverified 追加 `Adding More Phone Numbers` 也降至 .5。 |
| source `c6c3b…t9`：namespace 设置追加加密镜像、Registry 解释和授权三条旧问题 | 原相关段 `ibmcld_01484-4131-5923` 从重排 rank4 变 rank19；另外两篇相关段进入双路 union rank56/62，仍在 50 之外。R@5 从 1/3 降至 0，候选 R@50 仍 1/3。 |
| unverified `c01c8…t5`：当前明确 Carthage，却追加 CocoaPods 与旧安装问题 | R@5 从 2/3 升至 1；其中 `ibmcld_10817-7-1802` 从 rank6 进入 rank2。这证明重叠文档标签可以获益，不能据此认证选对当前操作对象。 |
| unverified `d5…t9`：Python SDK 查询追加 `If; Lite; Lite Plan` | R@5 前后均 0，但两篇 SDK 相关段从原候选 rank25/29 被推至 RRF rank58/62，候选 R@50 从 1 降至 0。只看 top5 会漏掉这类退化。 |

其他原本合法的 query 没有因 enum 改写而自动变好。例如 Data lakes 仍被 source 追加 Knowledge Studio/Admin；`ibmcld_16728-6533-8457` 当前查询重排 rank4，source 为 rank7。名字曾在来源出现，只能证明字符串来源，不能证明它是当前问题的对象。

## ConvSearch 的候选池与重排分离

相对 current，ConvSearch 在 39 个标注问题的重排 R@5 为 2 升、15 同、22 降；nDCG@5 为 5 升、11 同、23 降。与此同时，19/39 轮“候选 50 有相关段但 top5 为零”，current 仅 6 轮。具体例子：

- Data lakes `1c041…t6`：`ibmcld_16728-6533-8457` 已在 ConvSearch RRF rank4，却被其重排查询降到 rank47；该轮没有 token-cap hit。
- 附件纠正 `d5…t5`：标注段 `ibmcld_00510-7123-9213` 从 RRF rank2 降到重排 rank34。重写夹带了旧助手的产品/数字错误和附件操作代码，46/50 对达到 cap；此处不能单凭观察分离语义扩写和截断的影响。
- namespace `c6c3b…t9`：三篇标注段全在候选 50 内，重排为 rank25/26/9；current 虽仅召回其中一篇，却把它排到 rank4。
- SDK `d5…t9`：ConvSearch 把两篇标注段召回到 RRF rank1/2，重排后为 rank14/2；R@5 相比 current 从 0 升到 .5，但 user_topic 能将两篇均保在 top5。

ConvSearch 查询平均 1382.5 字符（462–2976），current 平均 56.2；9/41 rewrite 含代码围栏。实际独特 rerank 计算有 326 个 token-cap hit，其中 325 来自 9 条 ConvSearch 查询，另 1 个来自 source 的 Data lakes 查询；v1/v2 复用这个 source 结果，不是两次独立 cap 事件。所有 dense query 均未达到 8192 cap。

cap hit 只表示达到 1024，不排除恰好等长，不能逐对断言截断。更关键的是，ConvSearch **没有 cap hit 的 30 个带标注问题 R@5 仍只有 .24278**，9 个有 cap 的问题为 .36111；这是不同难度样本的描述性分组，不能当截断无害的因果试验，也不能把全部损失归因于长度上限。未改历史或静默裁剪作者输出。

qrels 也不是完整证据清单：`d5…t4` 所有方法的标注 R@5 均为 0，但 ConvSearch top1 已变为 Cloudant 的 `ibmcld_00581-0-1268` Limits 段，current top5 则混有 CDN、Discovery 和存储文档。`d5…t5` source 取回了明确列出附件大小的 `ibmcld_00558-23465-25360` 和 `ibmcld_12904-23555-25450`，其标注 R@5 仍为 0，且 top1 仍有 Discovery 干扰。这些局部正文观察必须与有限标签分开，不能将 unjudged 一律判错，也不能反过来宣称答案会正确。

## 执行量、限制与下一步

实际执行 147 次 BM25、147 次 dense query、151 次 rerank，共 7550 对，445 个操作全部成功；328 行中 177 行复用同一个完整 pipeline，原始来源明确。每方法逻辑上仍有 41 × 50 个 rerank 对，其实际计算因完全相同查询/候选被共享。初始化约 14.14 秒，独特操作墙时合计约 40.76 秒只是这次研究进程的工作量，不能当任何方法的独立在线延迟或缓存收益。`resources.completed_retrievals=0` 来自没有调用 `HybridRetriever.retrieve_turn` 的直接 helper 路径；有效执行计数来自逐条 ledger，不能误读为没有检索。

上游 ConvSearch 仍是 3B/完整 QA/temperature .7/4096 输出预算，selector 是 7B/最近三轮/temperature 0/256；模型、窗口与采样成本不可冒充仅算法变化。五会话开发数据已多轮用于研究，本轮不作显著性、独立 test、质量准入或完整 P3 闭环结论。

下一步保持本轮全部 41 条固定输出，拆开**召回查询**与**重排查询**：以 ConvSearch 单路候选 50、current+ConvSearch 双路候选 50 为两种池，分别使用原 current 与确定性 user_topic 重排，形成预先固定的四臂。原始作者重排与 user_topic 基线保留。这样只新增必要分类分数，不重跑 CQR，不从真实相关性标签选择每条 query；所有新排名完成后再评分。该设计检验可复用的候选池是否能发挥作用，而不是预先承诺会改善。

source selector 的后续则应针对实际的对象/意图过选与 `By/Include/Sort` 等候选噪声，结合完整失败分母评估；enum 解决可输出 ID 的合法性，不能承担语义验证。是否进入生产模式，应由新的固定比较和 P3 自身历史闭环决定，当前默认不变。
