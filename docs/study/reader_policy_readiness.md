# Reader 与策略接续状态

截至基线 `702ebe9`，R1/R2 已实际训练，P1 prototype22 已拟合并进入自身历史实验。完整数值统一保存在 [final_report.md](final_report.md)，此页只维护资产依赖与下一次训练接口。

## 已有资产

| 资产 | 已有规模与用途 |
|---|---|
| R0 | Qwen3-32B 非思考基座与当前机制对照 |
| R1 | 1773 个训练样本、1544 个实际训练 turn，56 optimizer steps |
| R2 | 在 R1 上继续训练，2264 个采样样本、71 optimizer steps；19 个验证样本沿用 |
| P0 | actual-prefix 支持资产 |
| P1 prototype22 | 22 个实际候选答案、5 个训练状态；奖励绑定 R1 |
| B v3 | train8/dev4 状态、176 次实际模型调用形成的收益资产 |
| C | 2 个 train、2 个 dev 固定窗口的经验保留资产 |

R2 的 2264 个样本包含课程重采样。样本条数、实际不同 turn 数和监督 token 数分别计量。P1 资产不能仅因同属 32B 系列就当作已适配 R2 的奖励模型。

## 本轮目标生成协议

`training.reader_data` 保留原 `traces → reviews → prepare` 文件流程：

1. 从精确运行 messages 生成目标；仅附阶段和输出预算。
2. 对 ANSWER 再执行独立核验：读取完整原文、实际历史和已执行的 READ/REDUCE 结论，评价固定目标的来源正确性、可见输入支持和任务绑定。
3. 核验发现上游错误时写入 `upstream_errors`，不为同一个 ANSWER 输入改写成隐藏来源中的正确答案。
4. 两次实际调用分别记录在 `review_calls.jsonl`。prepare 使用接受的固定目标；`visible_prompt_support=false` 的目标不进入新训练。
5. 已有 reviews 保持原样。重审使用新输出目录，保留原训练资产及对照。

当前 protocol 标识为 `visible_target_independent_review_v1`。READ 的原子事实、CPU REDUCE 的量值和最终 ANSWER 对全范围结论的使用分别形成课程，不用同一份最终答案错误同时监督三个阶段。

## 下一次训练与比较顺序

先离线重算既有公开分数，再用当前 R0 重放修正后的执行机制。随后以训练分区的实际错误生成新目标，开展 reader 对照，最后采集绑定新 reader 的 P1 奖励。具体目标、采样规模和 A/B/C 阶段依赖见 [review_20261103.md](review_20261103.md)。

固定历史四格用于归因；最终比较使用各自真实生成历史。原 13 或 8 个共同状态的分数保留为诊断，完整声明 cohort 另报完成率与正式质量。运行配置、评分环境和资产清单见 [final_runbook.md](final_runbook.md)。
