# A/C下一步机制：三个可证伪的工程扩展

2026-10-24。依据已保存的用户topic/双路P3、Gate0、epoch/host实验和评审负例，优先以下三项。它们改进实际材料或缓存行为；新增judge只负责辅助测量，不作为这些方法本身。当前均是实现建议，尚无新增效果结论。B的预算、等待和状态时效分支继续并行，不因A/C推进而收窄其目标。

## 1. A：把焦点行扩成带父级条件的来源表示

**当前直接依据。** `focus_refs`只在41轮中的附件纠正轮触发。已完成的开发正文扫描另发现两张无编号标题的明确限额表、一处带套餐条件的嵌套列表；现helper未覆盖这些结构。完整P3仍出现默认/套餐/对象范围混用。这里优先补足可见来源的结构，而不是再换一个通用判断提示。

实施更新：可选 `table_structure_mode: explicit` 的CPU入口已完成，[全41轮复核](table_structure_validation.md)新增两张表、六条关系。套餐列表在实际冻结正文中已失去缩进，不能把候选lead称为仍有明确嵌套结构；本轮未猜其父级关系。真正保留标题/缩进的父路径通过通用合同测试，但真实cohort未激活，生成效果尚待验证。

**复用与最小扩展。** 保留现有 `table_relations`原文span、CRAFT的MIT `build_row_texts`和当前词法focus；增加明确无编号表头、列表缩进/显式标题的父级路径。输出每项包含原对象、列/值、单位、父条件span、行内Evidence ID。只有原文结构支持的祖先作用域进入路径；无标记相邻注释仍标“scope unspecified”。不能从扁平缺失的colspan猜关系，也不把“同为MB”当作包含关系。TreeThinker的层次表头思想对应此缺口；其完整LLM树生成流程未复制，许可边界沿用既有核查。[TreeThinker论文§4](https://arxiv.org/html/2506.13405v1)、[已固定的CRAFT行表示](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/core/text.py)。

**可检验假设。** 在相同正文/QA/当前top5下，显式父条件路径比单层行表示更少产生“某套餐条件应用于全部对象”和“默认值覆盖例外”，同时不增加无源条件。新增贡献候选是有来源的作用域表示与生成使用效果，不是表头树概念本身。

**低成本反证。** 先重扫全部41个dev输入，报告新增覆盖、无变化轮和拒绝解析，不只列成功目标；添加对象/单位/标题层次的通用结构变体。随后在新覆盖轮冻结相同历史，比较现 `focus_refs`、仅新增结构解析、结构＋父条件三臂，保持原文、Q0和输出上限一致。检查条件遗漏、跨行包含、引用和新增错误及额外tokens/CPU。若解析覆盖增加却答案条件不改善，或引入错误作用域，假设被削弱，不能把多输出几个字段当收益。

BGE行评分可复用现有模型，但应在这项结构对照之后单独比较；CRAFT的`representative_row`和`mini_table`是不同算法，当前词法focus不是它们的复现。[固定dense入口](https://github.com/coral-lab-asu/CRAFT/blob/7c4be45bb292f4e050c4237b4141ccc9e91ebe30/src/craft_tabqa/retrieval/stage2_dense.py)。

## 2. C：按可恢复工作集和版本存活状态控制host写入

**当前直接依据。** 大池矩阵发生约6–9GB D2H但没有实际host恢复；16K已有恢复却host池仍很充足，Ours/no-retire未显示明显收益。当前C控制旧软引用，尚未控制新写入。最值得增加的是受控的写入准入分支，而不是把retire ACK改称physical free。

**复用与最小扩展。** EfficientAgent固定Apache-2.0源码在首次已解析prefix lookup时，按新增完整chunk数与host工作集/容量压力决定是否跳过该请求的新备份，并将决定贯穿重入队；复制时避免重复保存已有chunk。[固定`get_num_new_matched_tokens`与`_pressure`](https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/efficientagent/kvtier/connector.py)、[原论文§3–4](https://arxiv.org/html/2609.33762v1)。

可提取其小型准入谓词/计数思想，接到SGLang原生可选BACKUP_HOST写入决定处，保持lookup、load、锁、完成事件和原生重算路径；不搬入LMCache、1024-token chunk或其TP8字节常量。直接用本项目实际page64和57344 bytes/token。固定SGLang的FULL备份由主流程`cache_controller.write`处理，不能误以为改`FullComponent.build_hicache_transfers(BACKUP_HOST)`返回值即可控制写入。[固定FULL合同](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/mem_cache/unified_cache/components/full.py)。

本项目的必要扩展是将**业务会话活跃度与epoch分开**：同一会话频繁REBUILD生成多个engine session不能被当成多个独立活跃任务；已提交并退休的版本不增加未来工作集估计，但其尚未消失的实际占用必须仍计入压力。新增写入决定只使用当时可见的业务/版本状态与引擎事实，不使用未来回访或最终答案。空间足时走原生写入，不取消已启动DMA，不绕过native锁或恢复。

**可检验假设。** 在host容量接近实际可恢复工作集时，有条件的备份过滤减少“一次写入、回访前即被淘汰”的流量，保留更多有用prefix并改善完成时间；在大host时应与原生近似，否则过滤过度。版本感知增量要与不区分epoch的工作集规则单独消融。

**低成本反证。** 先用固定APPEND/REBUILD动作及已有真实input/output长度构建依赖保留的重放；在同一16K GPU下测试小host和现有大host两个点，确认实际压力与回访，不假设配置小就已饱和。比较选定原生写策略、固定过滤、条件过滤，再加有/无epoch去重；每次记录新增备份、跳过、实际host恢复、重算、completed DMA和完整时长。固定输出机制实验不作质量结论；有效后才进入真实闭环。若只减少D2H却增加重算或尾延迟，就没有达成目标。原生三写策略仍是共同强对照。

RR-Evict和近期replacement研究支持保留recency并关注每会话尾部损失，但当前记录还没显示“少数idle轨迹被整段清空”这一特定失效。故这三项中先选写入准入作为C扩展；不是认定轮转淘汰没有价值，也不以论文新近性直接更换底层策略。[RR-Evict](https://arxiv.org/html/2609.32278v1)、[replacement研究](https://arxiv.org/html/2609.28870v1)。

## 3. A：版本化API说明、签名与用例的证据包

**当前直接依据。** 最新P3中已取回正确SDK库链接，回答仍扩写出没有源支持的GET接口、参数名及JSON结构。仅提高top5相关性、识别GUID或换一个judge都不会自动提供缺失的API事实。值得增加的能力是把实际需要的接口证据补齐并版本化。

**复用与最小扩展。** DocPrompting以自然语言意图检索API说明再生成代码，官方实现Apache-2.0；这里复用该检索生成分工，继续用现有BM25/BGE，不迁移其FiD训练与旧Transformers环境。[作者固定仓库](https://github.com/shuyanzhou/docprompting/tree/b7e0e626d47a0649090465bb15c91192fee8777f)。2025年的API文档实证研究进一步将说明、参数、示例分别测试；其任务包含期望输出，一部分检索文档含目标示例，效果不能直接套到本项目。它还发现不匹配示例会诱导错误复制，因此不能只堆更多代码块。[原论文§3.2、§4.5](https://arxiv.org/html/2503.15231v1)。

实现候选是一个离线原语料adapter：从**固定版本的官方API/SDK来源**提取路径/符号、参数签名、示例及原位置，作为普通EvidenceBlock接现有CorpusIndex；按明确的库/产品/版本与用户操作意图检索一个小证据包。代码AST/OpenAPI只验证显式符号与参数形状，不把语法通过当运行正确，也不执行未知答案代码。版本无法确认时保留来源版本未知，不用今天网页替换旧快照、不把库最低版本号解释为唯一实现版本。完整QA与既有required继续保留，新材料/成本单列。

**可检验假设。** 对明确技术操作问题，版本匹配且有使用范围的签名＋原始用例，比只有产品说明或库链接更少生成不存在接口/错误关键字，同时不诱导不必要代码和错误类比。它是来源完整性扩展，不是静态分析本身的新算法。

**低成本反证。** 先离线扫描已保存代码回答，区分“证据里没有接口”“有接口但用错参数”“本题只需链接却额外造代码”；再冻结一小组原用户问题/历史，比较原材料、同版本说明包、说明＋签名/用例包。新增材料预算一致，纯缓存比较则把最终证据包共享给所有策略。报告接口/版本匹配、无源扩写、可执行性与额外检索/生成成本；负对照包括不同版本、不同产品同名参数和不适用示例。若只是答案更长或照抄更多例子，不构成改进。

## 执行顺序与验收边界

建议先做1的CPU结构覆盖与来源合同，同时做2的冻结轨迹/host压力测量准备；3先形成小型固定源码证据包和离线缺口分类。每项都有不依赖付费judge或新底座训练的第一步，必要的GPU调用随后由统一资源安排。新增模型判别结果可以帮助找误差，不能代替正反例的原文核验，也不能替这些扩展宣布质量通过。所有方法最终都回到可用答案、完整响应成本、资源影响和强对照，不按“模块实现了”宣称贡献成立。
