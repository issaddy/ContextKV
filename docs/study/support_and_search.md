# 支持评分与预算集合搜索

当前主实验采用固定模型推理，不新增训练。任务状态、原文条件与版本进入 `adapters/support_model.py` 的结构化查询，`TeacherSupportBackend` 使用与 READ/ANSWER 相同的固定 Qwen3-32B 端点评分。单段、pair、triple 的适用性、支持程度、桥接价值、支持类别和原文跨度均来自真实模型输出。真实历史用于确定问题指代与条件，事实依据保留原始来源跨度。

每个 Need 默认先评分 24 个单段；有适用的完整单段时结束该需求的组合评分。其余需求从单段评分保留 6 个桥接候选，批量评估 15 个 pair；前 2 个可扩展 pair 各取 3 个后继，最多形成 6 个 triple。`single_candidates_per_need`、`bridge_candidates_per_need`、`triple_seed_pairs` 与 `triple_successors_per_pair` 控制这一调度。教师请求在单个 batch 中共享查询和原文目录，每个组合只能引用自身及依赖对应的原始跨度。`support_schedule` 保存每级实际请求与成功评分数量。

主配置 `selection.utility: support_coverage` 对每个 Need 取集合内已评分单段或组合的最大 `applicability × joint_support`，再对 Need 求均值。该显式集合效用复用真实模型评分；beam 扩展及动作、布局候选不再调用模型评分。B2 只允许单段，B3 与 B5/B6/B7/Lmap 均可复用已评分组合；B5/B6/B7/Lmap 的其他机制由各自计划决定。

```python
scores = support_model.score_support(task_state, evidence_pool)
planner = BudgetedEvidencePlanner(
    tokenizer, SupportCoverageUtility(),
    measure_context=compiled_prompt_token_count)
sets = planner.search_sets(task_state, evidence_pool, scores, context_budget)
```

`controller/evidence_planner.py` 把单段及已评分组合当作扩展动作，每次计算依赖闭包。正文并集中的相同文本只渲染一次，同时保留各来源和版本身份；正文、来源标识与完整 prompt 均使用真实 tokenizer 计量。原 passage 数约束同时生效。默认束宽 4、最多 12 次扩展，最终保留不超过 3 个集合。选中集合保存原始 `source_blocks`，供来源档案和最终引用关联。

READ 后的 OpenNeed 将绑定实体、缺失关系与父 claim ID 追加到支持查询，原始问题及历史跨度保持原值。定向召回的旧证据与新证据共同参加组合支持评分与预算搜索；旧 passage 从新增 passage 配额中排除。正文预算按新 READ 包实际提交的材料和依赖计量：已读的独立前提只参与逻辑支持，新材料所需的表头或定义即使此前读过也随包提交并计 token。每轮缺口扩展单独保存评分、集合、配额与辅助推理成本。

`selection.rerank_retained_sets: true` 可选择对最终不超过 3 个集合再做固定模型重排，默认关闭。该调用只发生在最终集合上，原文批量支持评分与重排成本分别计入 `support_ms` 和 `utility_ms`。显式 `selection.utility: learned` 仅用于加载已有训练资产的独立实验；训练、实际回答偏好及 bootstrap 资产不参与当前主矩阵的学习结论。

教师的每次真实请求保存原始输出、原生 metadata、输入和输出 token、请求耗时与跨度目录。`backend.usage` 累积调用数、评分例数、实际 token 与支持/集合推理耗时；provider 将这部分作为在线辅助成本并入全过程总账。CPU 集合搜索耗时单列，离线训练与评分费用独立。主路径的固定模型调用数量随评分组数增长，与 beam proposal 或动作、布局候选数量无关。
