# 声明式 B6 矩阵、重复与冻结

2026-10-24清理后，唯一owned生成入口为 `python -m bench.b6_joint_sweep --matrix ... --output ...`，模型必须来自共享显式Qwen3 scope。默认只打印计划，`--execute`才启动自有服务器。固定7B生成、13格自动构造器及两套旧sweep CLI均已删除，不再进行7B实验。

[通用owned入口](b6_owned_matrix.md)说明资源、服务器验证及cleanup；[native入口](native_model_scope.md)说明B0/B1。原native_policy_sweep仅保留进程/测量/清单helpers，controller_capacity_sweep仅保留成本来源与实际runtime验证。历史13格及其判断见[b6_joint_validation.md](b6_joint_validation.md)，原产物不重写、不通过当前源码重跑。

## 生成与质量边界

每job保存实际server/runtime/pool、run/audit/summary和失败；模型、采样、材料、预算及源码保持预声明绑定。新生成不能借用其它模型或cache配置的成本标定。默认plan、CPU合同和一次小功能验收都不授予B6或质量资格。

开发矩阵可显式配置 `tuning.require_quality_evidence: true`，完成后只生成引用实际job文件的空评分模板。离线评分须满足[quality_acceptance](quality_acceptance.md)的完整cohort、答案绑定、固定evaluator及缺分规则。不能填quality_eligible布尔值代替评分。freeze不齐时只保存readiness，不能删失败job或只选最快子集。新的Qwen3生成、评分与冻结必须自足绑定，历史结果不能改标复用为同源性能/质量证据。

## 后续矩阵的显式重复与分组质量文件

后续 `bench.experiments` 支持显式 `repetitions: [1, 2]`，与 `seeds: [42]` 分开：每个系统重复仍使用声明的 seed，job ID 增加 `-r1` / `-r2`，结果及实际 raw `experiment_protocol` 保存 repetition；before-job hook 还接收 `CONTEXTKV_JOB_REPETITION`。重复按列表顺序外层展开，每次内部保持原方法/候选顺序，本接点不自动轮换或声称顺序无偏。未配置 repetition 时，旧 job ID、结果字段和执行顺序保持不变。时延配对只匹配相同 protocol / concurrency / seed / repetition 的参照。

固定7B的phase1构造器已删除。[Qwen3 owned matrix入口](b6_owned_matrix.md)支持显式重复及冻结test/dev_ablation；工程入口完成不代表heldout验收。多候选或多 workload/repetition 的质量冻结使用下列新索引文件；原单格 quality acceptance 配置继续可用。

```json
{
  "schema_version": 2,
  "blocks": [
    {"protocol": "P2", "concurrency": 4, "seed": 42, "repetition": 1,
     "quality_config": "/absolute/path/quality-P2-c4-s42-r1.json"},
    {"protocol": "P2", "concurrency": 4, "seed": 42, "repetition": 2,
     "quality_config": "/absolute/path/quality-P2-c4-s42-r2.json"}
  ]
}
```

矩阵仍用 `tuning.quality_evidence` 指向该索引。每块 `quality_config` 是原 `bench.quality_acceptance` 格式，但 `methods` 的键必须为该块全部真实 **job_id**，例如 `B5-c0-P2-n4-s42-r1`、`B5-c1-P2-n4-s42-r1`、`B0-c0-P2-n4-s42-r1`；`reference` 必须是该块预声明 reference method 的唯一实际 job ID。参照自身若有多个 candidate，不猜选第一个或最快者，直接拒绝。没有显式 repetitions 的矩阵使用 block `repetition=1`，对应 job ID 仍无 `-r1` 后缀。

每块分别保存实际 run/audit/scores/controller manifest/owned server record 引用及固定 evaluator protocol，按原格式评分和核绑定。评分合并文件如使用 `score_method`，应以实际 job ID 区分各次重复；不能按相同 task_id 覆盖多次生成。所有块必须使用相同的完整 evaluator 声明、五维定义和 epsilon，不能某次重复换宽松 rubric。每块允许不同的 methods/文件路径，因为它们对应不同的真实执行。

索引 blocks 与预声明矩阵的 `(protocol, concurrency, seed, repetition)` 集合必须完全相等，每块 methods 与真实 job 集合也完全相等；缺块、重复块、漏 job、错配重复的参照/文件均拒绝。生成失败或 unfinished 的 job 仍在 block 报告中，该块标记 `generation_incomplete`，不拿其余完整块做交集。全部评分文件和协议在最终 freeze 边界进入现有 provenance 绑定，不添加请求路径 hash。

资格按 `(method, candidate_index)` 在**全部预声明块上取 AND**。报告保留每块原有 session-cluster 区间，不把多次重复当作新增独立会话，也不输出合并重复的总体 CI。证据不齐与某候选完整评分后未通过是两种不同状态；前者阻止整个冻结，后者可使该候选落选。延续原行为：某个需要 tuning 的方法若没有任何合格完整候选，整体开发选择仍拒绝，不悄悄删除这个方法。所有块 AND 是预声明的开发准入规则，不是新的统计保证。

格式兼容不代表跨实现结果兼容。当前 phase1 远端 `0893134` 的结果若以后评分/冻结，仍须在该源码快照下处理；本地新增 repetition/block 实现改变了 implementation manifest，不能给旧 13 格补身份或改标成同源新实验。下一轮必须保留新矩阵、新实现及其完整原始执行记录。

## B6 是赢家别名，不再重复生成

冻结后的执行保留赢家原来的 `method`、`source_method`、candidate、job ID 和真实产物，增加 `method_aliases: ["B6"]`；不再另建一份相同配置的 `B6-*` 生成任务。`method-aliases.json` 将 B6 映射到每个实际 workload/repetition 的 canonical job ID。这些别名不是新样本或额外重复，赢家失败时 B6 也指向同一失败，不重试、不复制成功结果。

`reference_method: B6` 在冻结执行的时延比较中解析到该原赢家，comparison 仍保存真实 job ID。质量文件继续使用真实 method/job ID 和原 run/audit/scores，不能把 B6 别名再添加为一份独立评分。匹配消融只继承父配置，不继承 B6 名称；显式 repetitions 仍各自真实执行。此变更不重写历史报告；旧源码曾实际运行的 B6 副本继续作为历史原始执行保留，不能事后合并成只运行一次。
