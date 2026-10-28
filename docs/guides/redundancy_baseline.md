# 旧证据正文冗余比例基线

`redundancy_threshold`是按旧证据正文占比选择动作的简单基线，与长度阈值、固定周期及成本计划器共用候选和执行链。

对APPEND候选计算

`ratio = 非本轮required的可见证据正文tokens / 全部可见证据正文tokens`。

计数逐个使用已有`EvidenceSpan`定位完整原文，并用生成模型的tokenizer独立编码每段，`add_special_tokens=False`。分子包含旧的非必需正文；当前检索、明确回访和依赖闭包中的required都从分子排除。分母为0时ratio=0。每个物理正文span计一次，相同key长度只在本次候选构造中复用。system、QA、来源标题/元数据、引用提醒与表格注释都不计入；不使用没有tokenizer绑定的`EvidenceBlock.token_ids`。

比例**严格大于**阈值时选择合法REBUILD，否则APPEND。候选保留完整required集合，并满足版本、恢复、QA同步和输出预算要求。该比例衡量独立编码的正文tokens；完整prompt长度和物理KV占用分别计量。

使用[示例配置](../../configs/redundancy.yaml)，或在已有真实硬件配置中仅设置：

```yaml
planner:
  strategy: redundancy_threshold
  redundancy_threshold: 0.4
```

也可设为`fallback_strategy`。策略或回退策略使用该指标时执行正文分词，其余情况跳过。该策略直接使用候选正文计数，构造与分词开销计入控制器墙时；已选candidate保存分子/分母，未测时为null。

`profiling.decision_pairs`按源配置冻结这些计数，`profiling.control`继承该计量开关并测量策略所需的quote。使用缺少正文计数的旧文件时，先用固定tokenizer从原始提交前数据重新freeze并核对input IDs；缺少所需计数会报错。

工程验证覆盖：完整QA很长但正文比例不变、中文token口径、required依赖、严格阈值边界、零正文、不可重建时保持可行候选、正文只计一次、不改变实际prompt IDs、显式回退与生命周期、无需quote的完整控制器路径，以及固定Qwen tokenizer。真实dev性能/质量比较尚未执行；0.2/0.4/0.6是可选开发点，共用A配置、资源和输出预算，在开发集按完整响应和答案质量选择，选定后用于测试。
