# P0/P1 × R0/R1 与各自 P3

`bench.evidence_policy_comparison` 使用两份现有 support/utility 资产和两个真实 reader 服务。
P1 的训练绑定保持原 R1 identity；P1_R0 明确登记为跨 reader 的评估格。
P0 是 actual-prefix 初始化资产，不由这份运行入口宣称已选出最强 Incumbent。

先训练当前小批 P1 原型；已有成功回答可先 prepare，不必等整批采集结束：

```bash
python -m training.train_evidence_policy prepare --config configs/evidence_policy_r1.yaml
python -m training.train_evidence_policy train --config configs/evidence_policy_r1.yaml
```

prepare 从原批次连续物理采样给已保存回答补全成本。失败候选保存在独立 failed_cases.jsonl，
不参与质量偏好。原型样本数、题数和参数散列写入资产；训练完成不代表已取得开发收益。

冻结历史先跑一个状态的四格，再使用同输出目录去掉 limit 接续完整批次：

```bash
python -m bench.evidence_policy_comparison frozen \
  --matrix PATH/diagnostic_matrix.yaml --dataset dynamic_goal_dev \
  --records PATH/E0/results.unscored.jsonl --limit-states 1 --output runs/study/policy-comparison
```

冻结内容是原 actual E0 TaskState、actual QA 与 candidate_pool，不使用 verified TaskState 或 E1/E2 oracle。
AuditedTaskInterpreter 验证当前问题、轮次和历史绑定。初始检索/任务解释是共享原准备费用，单列其原始usage；
每格重新执行支持评分、效用选择和 READ/ANSWER。后继 gap 只能在同一冻结池中选材。
物理范围是GPU1与对应reader GPU，固定native前缀和关闭滚动gain策略。

各自 P3 使用同一问题目标图，沿本格已提交的真实回答推进：

```bash
python -m bench.evidence_policy_comparison p3 \
  --matrix PATH/diagnostic_matrix.yaml --dataset dynamic_goal_dev \
  --goal-policy PATH/goal_graphs.json --limit-conversations 2 --output runs/study/policy-comparison
```

P3复用bench.study.run_dataset，逐会话独立执行；明确READ输出截断保留该会话已完成结果、失败Attempt/StepRecord/原始输出/receipt，
不生成该失败轮的后继问题，继续其他会话。其他基础设施异常继续抛出。
辅助教师通过generation.helper_run_config明确指向R0配置，工厂实际核对helper端点的checkpoint与tokenizer/template。
GPU1做query/support，GPU2做在线R0 helper，reader为R0 GPU5或R1 GPU4；GPU6只在head训练时使用。

frozen每格目录、P3每会话目录均保存official/audit/references与unscored结果，评分独立执行：

```bash
python -m bench.study_scoring --matrix PATH/diagnostic_matrix.yaml --dataset dynamic_goal_dev \
  --generation-dir runs/study/policy-comparison/frozen/P1_R1
```

当前证据是本地入口、必要smoke与编译检查；实际四格和P3结果待指定容器运行。
