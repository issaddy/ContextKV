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

路径、task ID、revision 与 epsilon 需按实际任务预先填写，上例不是实测配置。双方使用同一适用集合；不能根据得分删除困难轮次，也不能把空集合当作通过。选中维度的缺分保留为缺分；未选维度或不适用轮次不进入非劣判定。原有全部维度覆盖情况仍在 `all_score_coverage` 中，报告同时列出精确 `task_scope` 与每维分母。结论仅适用于声明的维度和任务。

模型评审可通过 `bench.judge --task-protocol protocol.json` 读取同样的 `dimensions/task_scope`，并读取 `task_requirements` 中各 task ID 对应的任务描述。每次请求只投影当前任务的描述和适用维度，完整实际历史与当前可见证据照常提供；整组任务目录、未来要求和其他会话要求不进入该请求。配对比较同时核对双方当前任务要求一致。

评审响应支持嵌套 `scores` 或直接按维度列值的 JSON；需要的值仍必须明确且合法。保存原始模型响应，格式问题可离线重解析，无需再次调用模型。模型分数与原文审查分别保留，评分理由要求了未来任务或误读来源时，先标明评价无效，不把它算成系统的任务失败。

评分行可只含需要的维度，也可沿用已有五维评分文件。选择比较维度不会改变评分 rubric；旧评分若没有评价目标任务需要的内容，不能仅凭新配置宣称完成了新协议评价。

```bash
python -m bench.quality_acceptance \
  --config /path/to/acceptance.json \
  --output /path/to/comparison.json
```

## 复用已有评分与审计

`bench.judge`、官方 evaluator 适配、规则检查、MiniCheck 与 `bench.blind_review` 均为现有入口。规则检查提供引用范围和完成状态等工程证据，MiniCheck 提供已覆盖论断的来源支持；二者不能补造整份回答的正确性或任务完成率。自动评分按实际 evaluator 解释，自己的生成模型自评不能被称为独立验收。

人工盲审包保留完整实际历史、可见证据、代码和表格。方法私钥独立保管，先由真实评审者填表，再导入和聚合。导出不覆盖已填写文件；不把 assistant 案例审查填成人工评分。

```bash
python -m bench.blind_review import \
  --key /path/to/blind/private-key.json \
  --ratings /path/to/reviewer-a.csv --output /path/to/human-a.jsonl
python -m bench.blind_review aggregate \
  --input /path/to/human-a.jsonl --input /path/to/human-b.jsonl \
  --output /path/to/human-aggregate.jsonl
```

已有盲审模板仍为五维 rubric：不适用或不能评价的项按原说明填写 NA，比较时以预先声明的适用性决定是否要求该项。多评审者按等权均值聚合，同一适用维度有人填 NA 时保留缺失，分歧与评语不被删除。

## 比较、冻结与工程边界

导入复用现有 task/conversation/query、实际答案与提示身份，核对评分属于本次回答；不新增线上 hash。双方任务、当前问题和协议一致，允许各自产生不同真实历史与证据。完整 audit 仍覆盖全部成功回答。生成失败、截断和状态一致性另作工程检查，不与质量分数任意加权。

每个选中维度只在其共同适用任务上计算配对会话 bootstrap。缺分不取交集删除；只有一个会话时保留数值差和统计不确定性。`dev_quality_eligible` 表示声明协议下的开发集 evaluator 结论，不是整个系统或所有应用领域的认证。

正式 `scope=heldout` 沿用原固定 `split_manifest` 和原运行的划分记录，并提供预先保存的 `frozen_protocol`：

- `scope`、`reference`、`epsilon`、`dimensions` 与 `task_scope`；省略 `task_scope` 表示各维覆盖全部任务。
- 精确 `cohort`、`split_manifest` 和 `methods`；方法声明去除 `scores` 与 `score_method`。
- 相同 evaluator/rubric。人工声明 `kind=human_blind`、实际 `rubric_version` 和 `aggregation`；模型声明 `kind=offline_judge`、固定 judge 配置及 `reference_targets_supplied`。

冻结同时约束指标与适用集合，不能事后替换低分子集。配置记录来自研究操作者，不声称软件认证了冻结时间。独立模型结论与人工盲审结论分别输出；未完成独立验收不否定已有工程正确性或已观察到的响应收益。所有旧批次按原协议保留，不通过本次接口修改回填历史“通过”结论。
