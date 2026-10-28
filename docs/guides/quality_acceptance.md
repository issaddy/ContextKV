# 按任务适用性进行离线质量比较

当前真实结果统一见[数据与评价](../data_and_evaluation.md)，主指标与应用范围见[项目范围](../project_scope.md)。`bench.quality_acceptance` 读取已保存的生成、审计和评分，不调用模型，不修改原始分数。

## 声明评价范围

比较前在配置中声明 `dimensions`；可从 `correctness`、`groundedness`、`answerability`、`conditions_versions`、`citations` 中选择。默认保留原五维协议，已有结果含义不变。

- 正确性依据当前任务必要回答点与覆盖范围，不奖励额外内容或固定措辞。
- 来源支持检验实质结论；历史回答只供理解对话，不自动成为事实证据。
- 可答性关注有依据部分是否得到回答、缺失部分是否正确说明。
- 条件/版本用于需要保持或切换条件的任务；引用用于有引用要求的任务。

`task_scope` 可进一步声明某维适用的 task ID；未列出的维度默认覆盖全部任务。例如，一组任务都要求正确且有依据，其中只有部分轮次涉及条件变化：

```json
{
  "scope": "dev",
  "reference": "Native",
  "epsilon": 0.01,
  "dimensions": ["correctness", "groundedness", "conditions_versions"],
  "task_scope": {
    "conditions_versions": ["session-a:3", "session-b:4"]
  },
  "methods": {
    "Native": {
      "run": "runs/native/benchmark.json",
      "audit": "runs/native/audit.jsonl",
      "scores": "runs/native/scores.jsonl",
      "generator": {"model": "Qwen3-8B", "revision": "actual-fixed-revision"}
    },
    "ContextKV": {
      "run": "runs/contextkv/benchmark.json",
      "audit": "runs/contextkv/audit.jsonl",
      "scores": "runs/contextkv/scores.jsonl",
      "generator": {"model": "Qwen3-8B", "revision": "actual-fixed-revision"}
    }
  }
}
```

路径、task ID、revision与epsilon按实际任务填写。双方使用共同适用集合，报告`task_scope`、每维分母、缺分和`all_score_coverage`；未选维度及不适用轮次不参与该项比较。

模型评审可通过 `bench.judge --task-protocol protocol.json` 读取同样的 `dimensions/task_scope`，并读取 `task_requirements` 中各 task ID 对应的任务描述。每次请求只投影当前任务的描述和适用维度，完整实际历史与当前可见证据照常提供；整组任务目录、未来要求和其他会话要求不进入该请求。配对比较同时核对双方当前任务要求一致。

评审响应支持嵌套`scores`或直接按维度列值的JSON。原始响应随评分保存，格式问题可离线重解析。评分理由误读来源或评价了任务之外的内容时，记录为评价问题并复核。

评分行可包含所需维度或沿用五维文件，复用时对应原rubric和实际覆盖的任务。

```bash
python -m bench.quality_acceptance \
  --config /path/to/acceptance.json \
  --output /path/to/comparison.json
```

## 复用已有评分与审计

`bench.judge`、官方evaluator、规则检查、MiniCheck与`bench.blind_review`可按评价需要选择。规则检查提供引用范围和完成状态，MiniCheck评价论断与来源支持；完整回答质量使用任务评分。报告记录评审模型与生成模型的关系。

人工盲审包包含实际历史、可见证据、代码和表格。方法映射单独保存，由评审者填写后导入聚合。已有填写文件保留原样，案例审查与人工评分分别记录。

```bash
python -m bench.blind_review import \
  --key /path/to/blind/private-key.json \
  --ratings /path/to/reviewer-a.csv --output /path/to/human-a.jsonl
python -m bench.blind_review aggregate \
  --input /path/to/human-a.jsonl --input /path/to/human-b.jsonl \
  --output /path/to/human-aggregate.jsonl
```

盲审模板使用五维rubric，不适用项填写NA。多评审者按等权均值聚合，并保留缺失、分歧和评语。

## 比较、冻结与工程边界

导入复用现有 task/conversation/query、实际答案与提示身份，核对评分属于本次回答。双方任务、当前问题和协议一致，允许各自产生不同真实历史与证据。完整 audit 覆盖全部成功回答。生成失败、截断和状态一致性作为工程指标单独报告。

各维在共同适用任务上计算配对会话bootstrap，保留缺分及统计不确定性。`dev_quality_eligible`记录该开发集、任务范围和evaluator下的结果。

正式 `scope=heldout` 沿用原固定 `split_manifest` 和原运行的划分记录，并提供预先保存的 `frozen_protocol`：

- `scope`、`reference`、`epsilon`、`dimensions` 与 `task_scope`；省略 `task_scope` 表示各维覆盖全部任务。
- 精确 `cohort`、`split_manifest` 和 `methods`；方法声明去除 `scores` 与 `score_method`。
- 相同 evaluator/rubric。人工声明 `kind=human_blind`、实际 `rubric_version` 和 `aggregation`；模型声明 `kind=offline_judge`、固定 judge 配置及 `reference_targets_supplied`。

冻结文件保存指标、适用集合和方法声明。独立模型评价与人工盲审分别输出，历史批次按原协议读取。
