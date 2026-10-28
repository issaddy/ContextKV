# 离线成对决策验证

`profiling.decision_pairs` 把已完成开发集运行的某轮输入冻结下来，在**独立、隔离的 SGLang profiler 服务**上分别执行 APPEND 与 REBUILD，比较预测差、实际差与选错代价。它补充阶段标定误差验证，不改变在线服务的单次生成行为，也不提供质量结论。

## 1. 从开发集账本冻结输入

先完成一个真实模型的 P2 或 P3 开发集运行。保留它的 `sessions.sqlite3`、`manifest-<run_id>.json` 和实际使用的开发/测试划分文件。在同版本控制器、同模型/tokenizer 下运行：

```bash
python -m profiling.decision_pairs freeze \
  --database runs/cloud-dev/sessions.sqlite3 \
  --run-manifest runs/cloud-dev/manifest-<run_id>.json \
  --split-manifest configs/mtrag_split_manifest.json \
  --model-path "$CONTEXTKV_MODEL_PATH" \
  --selection stratified --seed 42 --limit 15 \
  --output runs/decision-dev/cases.jsonl
```

替换 `<run_id>` 为真实运行 ID。程序以只读 SQLite 事务读取该运行的 COMMITTED attempt，并通过 `expected_version` 获取**该轮执行之前**的账本。SQL 排除当前 attempt 的 `result`：当前答案、当前真实输出长度和未来轮次不会用于候选构造或输出先验。B4 使用已保存的 `retrieval_priority`，条件闭包不会变成检索排名。

约束和输出：

- 源 manifest 中全部会话必须属于传入划分的 `dev`，且 split 文件 hash 必须和源运行一致；拒绝 `test`、P1 参考历史及未决运行。
- 重新构造的原选中候选必须与保存的 `input_hash` 一致。旧代码/旧模板或缺失排名导致差异时记为 `source_candidate_changed`，不悄悄改输入。
- 初始轮、强制恢复和不足两个可行候选的轮次记录跳过原因。
- `stratified` 在完整可行池中按会话及距上次 REBUILD 的轮数（0、1、2+）分层，以固定 seed 打乱后轮转抽取；有足够名额时覆盖各层，以比较不同上下文积累阶段。选样只读 pre-turn 状态，不按当前输出、实测收益或 oracle 动作挑案例。manifest 保留每个选中案例的历史轮次、版本年龄和两候选长度；当 limit 小于层数时不保证覆盖所有层。旧默认 `database_order` 保持可复算性。
- `cases.jsonl` 保存两个完整候选、实际 token、pre-turn 快照及预热 token；旁边 `.manifest.json` 保存输入来源、划分、冻结文件 hash 与跳过记录。
- 可冻结模拟数据以验证文件/协议流程，但后面的硬件收集命令拒绝将模拟 token 来源用于真实实验。

冻结文件必须独立保存，不覆盖已有输出。它包含原问题、证据和历史回答，应按源实验数据相同方式管理。

## 2. 在隔离 profiler 执行两个候选

profiler 必须启用 `CONTEXTKV_ENABLE_PROFILING=1`，并且不承担业务请求。`cold`/`host_only` 准备会修改或清空该服务的缓存；不要指向正在跑主实验的共享 endpoint。源 SQLite 不会被修改。

```bash
python -m profiling.decision_pairs run \
  --cases runs/decision-dev/cases.jsonl \
  --config configs/qwen3_8b_scope.example.yaml \
  --base-url http://127.0.0.1:31000 \
  --cost-artifact runs/calibration/cost-model.json \
  --cache-states cold,gpu_ready,host_only \
  --repetitions 3 --seed 42 \
  --output runs/decision-dev/pairs.jsonl
```

GPU-only服务使用`--cache-states cold,gpu_ready`。模型/tokenizer/template指纹与冻结输入匹配。省略成本文件时，按配置使用Measurement文件或未标定模型，并记录实际采用的固定策略。

启动及每个pair开始时检查artifact的engine fingerprint/cache configuration。不匹配时保留诊断预测，标为未标定并执行固定策略；manifest和pair记录`calibration_scope`。旧artifact缺少绑定时标为`legacy_unbound`。GPU、驱动和拓扑按实际运行环境记录。

每个 pair 的两个 arm 都执行以下独立流程：

1. `cold` 清空缓存。两个候选的执行先后按固定 seed 随机化。
2. 对 `gpu_ready`/`host_only`，以**旧 execution 的实际 input IDs + 实际 output IDs**作为预热输入，末尾加同一个非目标续接 sentinel，再生成 1 token 使旧路径完整物化。不会重新采样历史答案。预热临时 session 结束后退役，再等原生传输完成或迁到 host。
3. 同时只读报价两个冻结候选；两臂准备后的 GPU/host/pending/native-load 特征必须一致。记录报价、预热 receipt 与成本预测。
4. 仅执行当前 arm 的一个候选。记录完整新输出/实际长度、receipt、服务端完整生成耗时和客户端完整响应耗时；清理临时 session。

两臂之间重新执行上述流程，因此先执行的候选不会为另一候选留下 KV。发现其他请求在执行/排队、状态不符或两臂 cache shape 不一致时记录失败。若第二臂失败，第一臂已经得到的输出和报价仍保留；不把失败 pair 从失败统计中抹去。

`cold/gpu_ready`逐臂核对receipt的`device/host/prefill`与quote的`ready_gpu/0/input-ready_gpu`，quote已包含原生`n-1`及page处理。执行中出现host加载、命中变化或额外重算时，保存完整result并将pair标为失败。`host_only`记录原生准入实际选择的恢复或重算。

该fixture以并发1恢复一条已提交历史路径，隔离比较两种布局的执行成本。源运行中的其他会话、分支和LRU状态属于另外的负载条件。`host_only`的恢复量以实际quote和receipt为准。

## 3. 比较差值与错误动作代价

执行结束自动生成 `pairs.summary.json`。也可分别使用服务端和客户端范围分析：

```bash
python -m profiling.decision_pairs analyze runs/decision-dev/pairs.jsonl \
  --time-field server_elapsed_ms --tie-ms 1 \
  --output runs/decision-dev/server-summary.json

python -m profiling.decision_pairs analyze runs/decision-dev/pairs.jsonl \
  --time-field client_complete_ms --tie-ms 1 \
  --output runs/decision-dev/client-summary.json
```

`Δ = C(APPEND) − C(REBUILD)`；正数表示 REBUILD 更便宜。统计包括：

- 两候选都受标定支持的 pair 覆盖率、计划原因及回退数。
- 标定 pair 上的 Δ 符号准确率；`tie-ms` 为事先选择的平局容差，不能看测试结果后调参。
- 实际计划选中动作的 regret：`C(selected) − min(C(APPEND), C(REBUILD))`。按实际选中动作计算。
- 各会话聚合及等会话权重的平均 regret；每个 pair 的预测差、实际差和双方输出长度均保留。
- 失败/缺失计时数量，不把缺失补成零。

默认服务端时间范围与标定联合首 token＋后续 Decode 更一致。客户端 `client_complete_ms` 从 generate 提交到完整响应，包含 HTTP/返回开销；不包含状态重置、预热、prepare、retire 或业务账本提交。所有候选共同控制成本在此诊断中设为 0，冻结模型不做在线残差更新，配置/模型文件及这一范围均写入 manifest。

该 regret 衡量受控 fixture 中的时延选择差距。两候选的生成内容和长度可能不同，方法收益结合同任务的回答质量评价。此入口用于开发集调试，测试配置预先冻结。

## 4. 在同一批真实输出上比较冻结成本模型

`rescore` 复用一次已经完成的成对采集，离线比较例如 lookup 与 positive-basis artifact。它不创建 engine client，不发送请求，也不重新生成候选或答案：

```bash
python -m profiling.decision_pairs rescore \
  --cases runs/decision-dev/cases.jsonl \
  --input runs/decision-dev/pairs.jsonl \
  --config configs/cost.yaml \
  --cost-artifact runs/calibration/cost-positive.json \
  --output runs/decision-dev/pairs-positive-rescore.jsonl
```

`--input` 旁必须保留原 `.manifest.json`；其 frozen bundle 必须与 `--cases` 相同，源采集必须完成。`--cost-artifact` 也可由配置的 `cost_model.artifact` 指定，但必须是已冻结 artifact。`--config` 只提供成本项消融和 planner 参数；候选文本、生成上限、历史输出先验的已提交数据来自 frozen cases，不能通过新配置改变已经采集的生成条件。为了单独比较成本模型，应保持两次 planner 与消融设置相同。

重评分流程：

1. 匹配每行 `case_id` / conversation / request 及每臂保存的输入 hash、长度、receipt；校验两臂报价描述冻结候选，且缓存准备 shape 相同。
2. 用源 manifest 的 engine 配置检查新 artifact 的标定范围。已知不匹配时保留诊断预测，但标为未标定并回退；不把来源不明的配置称为硬件已验证。
3. 按原 `execution_order`，使用**先执行那一臂的原报价**重放 `choose_plan`。每臂自身的 prediction 使用该臂原报价，与采集时一致；并发仍为 1、共同控制成本仍为 0。
4. `choose_plan` 只接收冻结的 pre-turn snapshot、两候选、原 quote、新模型和 planner 参数。当前生成文本、实际输出长度和耗时不作为输入，不调用 `observe`。当前输出/耗时只有在事后的 `analyze_pairs` 中参与差值、regret 和长度展示。

输出新的 JSONL、manifest 和 summary，默认拒绝覆盖已有文件。每臂的实测时间、输出、receipt、warmup、报价和 hardware/simulation 来源保持原值；原选择、原因和预测保存在 `rescoring` 中。源失败行保留为失败并标记 `source_failure_preserved`，不会静默丢弃或变成成功。结果仍可用现有 `analyze` 选择 server/client 时间范围，summary 的逐 pair 明确记录 repetition。

新 manifest 保存源 pairs/manifest 的路径与文件 hash、源采集完整清单、frozen 来源、替换 artifact、配置使用范围、标定 scope 和 `new_generations: 0`，用于文件级实验来源检查。原始采集文件与答案保持原样。

重评分在同一批实测输出上比较决策模型，复用已有硬件测量和答案。模型使用独立阶段标定数据拟合；依据本批 dev regret 调整的结果记录为开发调试，泛化效果使用独立测试评价。

## 本地合同验证

```bash
python -m pytest -q tests/test_decision_pairs.py
```

CPU测试覆盖冻结输入、数据划分、B4排名、候选变化、两臂独立准备、Host状态、部分失败留存及差值/regret计算。重评分检查预测仅使用提交前信息、case/hash/shape匹配、源记录保留及纯离线执行；测试时间由double提供。
