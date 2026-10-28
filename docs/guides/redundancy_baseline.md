# 旧证据正文冗余比例基线

`redundancy_threshold`是按旧证据正文占比选择动作的简单基线，与长度阈值、固定周期及成本计划器共用候选和执行链。

对APPEND候选计算

`ratio = 非本轮required的可见证据正文tokens / 全部可见证据正文tokens`。

计数逐个使用已有`EvidenceSpan`定位完整原文，并用生成模型的tokenizer独立编码每段，`add_special_tokens=False`。分子包含旧的非必需正文；当前检索、明确回访和依赖闭包中的required都从分子排除。分母为0时ratio=0。每个物理正文span计一次，相同key长度只在本次候选构造中复用。system、QA、来源标题/元数据、引用提醒与表格注释都不计入；不使用没有tokenizer绑定的`EvidenceBlock.token_ids`。

比例**严格大于**阈值时选择合法REBUILD，否则APPEND。版本失效、恢复、完整QA同步、输出预算和候选可行性仍优先；不能为达到阈值删required。独立编码正文的token数是统一计量代理，既不是两个完整prompt的长度差，也不是立即能释放的KV量。

使用[示例配置](../../configs/redundancy.yaml)，或在已有真实硬件配置中仅设置：

```yaml
planner:
  strategy: redundancy_threshold
  redundancy_threshold: 0.4
```

也可显式设为`fallback_strategy`。只有策略或回退策略用到该指标时才执行正文分词，其他策略默认没有这项额外工作。它不调用engine quote、不需要成本标定；完整构造与分词开销仍计入控制器墙时。`candidate_built`事件和保存的candidate记录分子/分母，未测时为null，不能解释为0。

`profiling.decision_pairs`按源配置冻结这些计数，`profiling.control`同样继承该计量开关，并只测策略实际需要的quote。旧冻结文件默认没有正文计数，不能直接切换到本策略重评分；需使用固定tokenizer从原始提交前数据重新freeze并核对input IDs。代码遇到未测计数明确报错，不以prompt长度差补值。

工程验证覆盖：完整QA很长但正文比例不变、中文token口径、required依赖、严格阈值边界、零正文、不可重建时保持可行候选、正文只计一次、不改变实际prompt IDs、显式回退与生命周期、无需quote的完整控制器路径，以及固定Qwen tokenizer。真实dev性能/质量比较尚未执行；0.2/0.4/0.6是可选开发点，共用A配置、资源和输出预算，在开发集按完整响应和答案质量选择，选定后用于测试。
