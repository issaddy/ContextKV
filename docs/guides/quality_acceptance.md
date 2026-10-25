# 可执行的离线质量验收

当前真实结果统一见[数据与评价](../data_and_evaluation.md)。本页说明现有离线评分、来源绑定和独立冻结验收接口；自动评分、人工评审与统计资格分别记录。

建包时，真实生成、完整 evidence audit、规则审核、MiniCheck、官方 evaluator 适配、HTTP judge 和人工盲审入口均已存在。旧 Q0/Q1/Q2 的 P1/P2 各 123 份盲审表及本批新增P3的123份表仍为空；该批当时没有付费API或远端judge调用。后续独立模型的实际运行不回填这些人工表，assistant案例检查也不作为人工评分。

本轮补齐三个实质问题：重新导出不再覆盖已填写的盲审包；多评审者有显式聚合入口；双侧分数必须绑定各自真实运行的答案和提示，不能用同一 task 的另一份回答或同一回答的另一份提示代替。离线入口复用已有 task/conversation/query/input IDs 与 `receipt.input_hash`，只在导入时用既有 `token_hash` 核对，不新增在线 hash。

## 当前可用的非付费路线

| 路线 | 能完成的工作 | 不能据此声称的结论 |
|---|---|---|
| 显式配置的本地 Qwen3 OpenAI-compatible 服务 + `bench.judge` | 全可见证据、实际自身历史、五维 JSON 诊断；冻结 prompt/model/revision/temperature | 同生成权重自评不是独立质量验收；更换模型名/本地路径不能改变这个事实 |
| 已运行的 MiniCheck | 独立判别器检查指定 prose claim 与 source 的支持关系 | 未覆盖的代码、表格、整体正确性、回答充分性不能由 grounding 分数补齐；逐 claim 也不等于完整回答五维分数 |
| 固定 IBM 官方 evaluator | 原 algorithmic、underspecified、IDK、RAGAS、RADBench 流程及官方输出 | “官方脚本运行成功”不等于正确、非劣或独立；使用同生成器作官方 judge 仍属自评 |
| 独立人工盲审 | 全文、代码、条件/版本、引用、历史不一致与不可回答情况 | 当前空表和 assistant audit 不是已完成人工评分；5 个 dev 会话不是最终 heldout 证据 |

固定官方源码 `53fef6f...` 的 `load_chat_openai_llm` 连接 `http://localhost:8001/v1`；`--provider vllm` 实际走该 OpenAI-compatible 接口。`hf` 分支并非完全无服务依赖：underspecified 阶段仍调用同一 local API。因此复用现有 SGLang 时应在独立评测时段准备该地址及独立 evaluator 环境，而非假设换一个 CLI provider 就能运行全链路。原流程先执行算法指标，再执行各 judge；官方依赖和模型文件须已准备，不能把此文档当成安装完成的记录。[固定 judge wrapper](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/scripts/evaluation/judge_wrapper.py)、[固定 generation 入口](https://github.com/IBM/mt-rag-benchmark/blob/53fef6f5917c7e10c436f697e69a955dbc795d81/scripts/evaluation/run_generation_eval.py)。

已有 `bench.official prepare` 只在生成后 join targets/enrichments，保持方法自己的 input/contexts；judge 和盲审的在线信息边界不变。人工导出白名单不含 targets/enrichments。独立自动 judge 能支持开发集方法选择，但它仍是特定 evaluator 的测量；正式结论应保留独立人工检查及模型评分的不一致情况。

## 真实 P3 包与操作顺序

`runs/quality-acceptance-dev-v2/package-manifest.json` 指向当前盲审包。原 `runs/quality-acceptance-dev` 的 v1 包未修改，但其 private key 缺少评分输入绑定，已由 v2 替代；不能将 v1 回填结果当成已绑定验收分数。三组为 BM25/last3、hybrid/current、hybrid/correction_aware，每组 41 turn、同 5 个 Cloud dev 会话。123 轮 query/历史/检索证据/提示 token 及原始划分摘要已实际核对通过。rule audit 的未知引用及可见正文缺失均为 0；截断分别为 0/1/1。这些 flags 不包含语义正确率。

只交付 `blind-p3/review.jsonl`、`ratings.csv` 和 `REVIEWER_GUIDE.md` 给评审者，独立保管 `private-key.json`。每位评审者复制一份空表，独立填写 reviewer ID、五项分数、history mismatch 和依据。正文中的引用标记、完整证据以及方法自身历史都保留；先前 assistant 的断言不能当作 source。评分 rubric 有版本标记。

```bash
# 两位实际评审者完成各自 CSV 后执行；不要自动生成这些评分。
python -m bench.blind_review import \
  --key runs/quality-acceptance-dev-v2/blind-p3/private-key.json \
  --ratings /path/to/reviewer-a.csv --output runs/quality-acceptance-dev-v2/human-a.jsonl
python -m bench.blind_review import \
  --key runs/quality-acceptance-dev-v2/blind-p3/private-key.json \
  --ratings /path/to/reviewer-b.csv --output runs/quality-acceptance-dev-v2/human-b.jsonl
python -m bench.blind_review aggregate \
  --input runs/quality-acceptance-dev-v2/human-a.jsonl \
  --input runs/quality-acceptance-dev-v2/human-b.jsonl \
  --output runs/quality-acceptance-dev-v2/human-aggregate.jsonl
```

聚合为每份答案的等评审者算术均值，任一评审者将某维填 NA 则该维仍缺失。保留每人的分数、comment、min/max 分歧和 reviewer IDs，不自动仲裁。重复 reviewer、不同答案、不同 rubric 均拒绝合并。比较时每个对应 task 两侧必须使用相同评审者集合与聚合规则，避免把评审者宽严差别当成方法差别。

将 `acceptance.json` 中各方法的 `scores` 改成聚合文件路径，保留 `score_method` 以选择对应方法，然后：

```bash
python -m bench.quality_acceptance \
  --config runs/quality-acceptance-dev-v2/acceptance.json \
  --output runs/quality-acceptance-dev-v2/human-acceptance.json
```

这个尚未回填的123份包已实际产生 `readiness.json`：每方法五维各缺41个task，`dev_quality_eligible=false`，没有伪造CI。它不是后续2665维运行的状态；后者分数完整，但12候选五维联合非劣均未成立。分数不齐时仅完整维度计算配对会话bootstrap，不删除困难task、NA或截断答案来凑交集。

若要先跑零付费自评诊断，已有 `local-qwen-self-judge.json` 固定现有 7B revision 与 prompt。服务地址及 served model 名需对应实际服务器；本轮未发起此请求，性能实验期间不要并行运行 judge。

```bash
python -m bench.judge --audit PATH_TO_SAVED_AUDIT \
  --config runs/quality-acceptance-dev-v2/local-qwen-self-judge.json \
  --output runs/quality-acceptance-dev-v2/METHOD-self-judge.jsonl
```

另存一份 acceptance 配置，为各方法设置各自 judge 文件路径并删去 `score_method`（只有合并人工文件需要方法过滤）。输入过长或 API/解析失败应保留失败记录，不静默截断可见证据。相同模型名或相同权重 revision 会标记 `self_judge`，数值比较可显示为诊断，但不能通过质量准入。不同模型需在 generator/judge 的声明中保留准确身份；这些声明不是远程权重证明。

## 准入和最终冻结边界

`bench.quality_acceptance` 读取每方法 `run`、`audit`、可选 `scores` 与 generator model/revision。它复用 `bench.report.bind_quality_rows` 将双方评分绑定真实成功回答；audit 必须覆盖全部成功答案。audit 的 query/protocol/自身历史/检索证据核对原 `online_turn`；原生提示 IDs 直接比较，controller 则复用原 receipt 的 input_hash。新 judge 与盲审评分保留 query/input_hash，缺少绑定的旧分数仅作诊断。双方 query/conversation/protocol 必须相同，允许方法自身历史不同。payload、turn.result、audit 或 score 任一位置记录 simulation/fake_http_correctness 都阻止验收，较宽泛的硬件标签不会掩盖它。generation 失败、未完成依赖、缺分、未知/不匹配评分协议或自评也不能准入。`bench.report --reference-quality` 同样要求 `--reference-run` 并核对双方答案，单独报告不宣称正式非劣。

dev 的 `dev_quality_eligible` 只表示在本次开发 evaluator 上五维CI均通过既定epsilon。矩阵显式启用 `require_quality_evidence` 后会在freeze边界读取并核对实际评分/生成文件、完整任务及各workload/repetition块；本工具本身不修改矩阵或自动认证语义真值。模型/规则探索与已看过的会话不能被改名为正式验证。

最终 `scope=heldout` 要求提供原固定 `split_manifest`，audit 的全部会话确实属于 test 且不属于 dev；还核对原运行已有 `workload.split` 及 `split_manifest_sha256`。controller 的同 UUID `manifest-*.json` 自动读取，也可用方法声明中的 `manifest` 指定保存的 sidecar；使用其现有 dataset 摘要。划分文件只在离线入口读/hash 一次。已存在的矛盾拒绝导入；没有原始摘要的旧记录仍可诊断，但不能 established。还须提供预先保存的 `frozen_protocol` JSON，其字段必须与实际配置一致：

- `scope`、`reference`、`epsilon`、完整五维 `dimensions`；
- `split_manifest` 路径、排序后的精确 `cohort` 会话列表；
- `methods`：acceptance 配置的完整方法声明，但排除 `scores` 和 `score_method`；
- `evaluator`：人工为 `kind=human_blind`、`rubric_version=contextkv-five-dim-v1`、`aggregation=equal_rater_mean_any_NA_missing`（单评审直接导入为 `single_reviewer`）；自动为 `kind=offline_judge`、完整固定 `judge` 的 provider/model/revision/prompt/temperature/max_tokens 及 `reference_targets_supplied`。

冻结文件来源仍是操作者保留的研究记录，不声称本工具认证冻结时间。独立自动 evaluator 通过后只设置 `heldout_evaluator_noninferiority_established`；`quality_noninferiority_established` 还要求人工盲审来源。5 个开发会话的显著性和同模型自评均不能通过改配置变成这一字段。

针对测试覆盖答案错配、缺分、NaN、双侧 reviewer/rubric 不匹配、模拟、路径别名自评、dev 冒充 heldout、冻结不匹配、旧表不覆盖、标签不导出及显式多评审聚合。所有实现均在离线评测模块，未更改主控制器、生成模板或线上证据边界。
