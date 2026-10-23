# Granite Guardian 4.1原生诊断入口

`bench.granite_judge` 是独立离线groundedness风险适配，复用固定IBM模板文字与已有 `NativeSGLangClient`，没有新HTTP框架。默认no-think、64输出tokens，先服务已有29项开发probe；不修改在线答案、不进入五维quality gate。模型/作者固定版本、格式差异与依据见[独立核查](goal_acceptance_audit.md#五granite-guardian-41原生接入核查)，Apache归属见[NOTICE](third_party/GRANITE_GUARDIAN_NOTICE.md)。

输入从完整audit读取：全部visible正文通过官方 `documents` 槽位，按稳定reference ID关联；即使不是本轮top5，只要仍可见也会保留。仅归档而不可见的材料不进入documents，缺正文则失败。实际history保持user/assistant角色，当前问题、本轮待评答案和末尾guardian块单列。过去assistant断言是对话上下文，不是source。没有targets/labels输入参数。

固定本地tokenizer的 `apply_chat_template(..., documents=..., tokenize=True, return_dict=False, add_generation_prompt=True)` 只编码一次；完整IDs直接送原生 `/generate`，不传session ID、不在服务端再套chat模板。请求返回prompt IDs并与提交值核对。部署窗口必须显式配置，区别于模型config的131072上限；输入加输出预算超限会完整保留为失败，不静默截断。

原生 `<score>yes</score>` 表示满足“不受来源支持/与来源冲突”的风险标准，**yes=风险，no=未检测到风险**，不代表确定的整答正确性或五维分数。只接受单个yes/no，可带think标签；缺失、重复/非法score、HTTP错误、abort和输出截断均保持未评分。即使截断前出现完整score，也不当有效结果。ledger保留实际IDs、原输出、finish reason、HTTP wire、模型/协议、生成来源及原答案绑定；失败后继续剩余样本，最后退出非零，无自动重试。

配置示例（端口和窗口须对应实际已启动服务）：

```json
{
  "base_url": "http://127.0.0.1:31117",
  "context_window": 32768,
  "max_new_tokens": 64,
  "think": false
}
```

```bash
python -m bench.granite_judge \
  --audit runs/atomic-development-v3/audit.jsonl \
  --config runs/granite-config.json \
  --tokenizer-path models/granite-guardian-4.1-8b \
  --output runs/granite-development-v1/ledger.jsonl
```

同29项完整来源/运行记录的driver为 `runs/run-granite-development.py`。不加 `--execute` 只计划；执行时复制audit和标签sidecar原字节，标签不解析或传给模型，保存config、来源hash、runtime、ledger、summary与execution。未使用test，不按标签选择要执行的样本。

```bash
python runs/run-granite-development.py \
  --inputs runs/atomic-development-v3 --output runs/granite-development-v1 \
  --base-url http://127.0.0.1:31117 --context-window 32768 \
  --tokenizer-path models/granite-guardian-4.1-8b \
  --resource-note 'operator-recorded co-resident model/server state' --execute
```

18项专项测试验证完整材料、角色、精确wire IDs、无双模板、风险极性及失败/截断；与native/quality验收联合65项、另6子测试通过。它们使用可识别的模拟response，不是Granite准确率。真实源集的supported/unsupported正反例、单位换算、脚注、历史污染和代码需分别复核；no-think与显式think属于不同协议与成本，若比较应另存结果，不能把真实失败后自动增加预算当作原配置成功。

实际H200完整29项已执行并保留两版结果：v1在发送前因返回类型失败，v2的29项全部scored/stop且输入与独立固定tokenizer逐token一致。22个事实控制吻合，但拒答、引用绑定和真实对象范围仍有不足，详见[真实验证与BYOC候选](granite_judge_validation.md)。测试通过不替代该语义核查，no也不表示整答已被验收。

## 可选BYOC预设标准

配置新增 `criterion_profile`，只允许下面三个固定值；旧配置省略该字段仍用 `ungroundedness`。默认criterion、guardian完整文字和原生协议名保持不变。两个新增标准是ContextKV的BYOC适配，复用官方模板与原生transport，不是作者原始ungroundedness的复现结果；没有任意在线prompt入口、阈值训练或新hash体系。

| profile | 判定目标 | 覆盖口径 |
|---|---|---|
| `ungroundedness`（默认） | 原始整答source-grounding风险 | 原行为不变 |
| `relation_scope_v1` | 完整对象—量值—单位—比较符—逐项/总量—包含关系—条件/例外/版本/角色，以及当前问题的证据可答性 | 使用全部visible原文；显式关系允许正确推导/换算，不能由分列值推断包含、相加或性能保证；恰当的证据有限拒答不应当作风险 |
| `citation_binding_v1` | 当前答案实际引用的来源是否支持对应断言 | 复用现有单ID/逗号组解析；仅引用之外的来源支持不能补救错引；组内联合证据须支持所述关系 |

所有profile保持**yes=该标准检测到风险，no=未检测到风险**；ledger的judge元数据保存 `criterion_profile`、`criterion_profile_version=1`、完整criterion、协议名和极性，不生成五维分数。模型输入仍是完整有角色的history、当前问题/答案和documents；历史assistant不能作为事实来源。64输出预算/no-think仍为默认，原始wire、IDs、finish reason和失败保留行为共用。

引用模式只有当前答案存在已支持语法的引用才调用模型。未知但语法可解析的ID（例如组内多出一个不存在的ID）**仍然调用模型**，并保留 `rule_audit.unknown_citations`；规则不会直接代替模型输出yes。没有引用或只有普通数字、裸ID、链接等未支持形式时，保存完整消息/来源并返回 `status=not_covered`、`native_score=null`、`risk_detected=null`，不执行tokenizer或generate。它表示该引用协议未覆盖，不是正确、不正确或模型运行失败。summary分别保存 `scored`、`failed_task_ids`、`not_covered_task_ids`；所有原输入均保留一行。可见来源缺失等真实输入错误仍失败，不会伪装成未覆盖。

```json
{
  "base_url": "http://127.0.0.1:30001",
  "context_window": 32768,
  "max_new_tokens": 64,
  "think": false,
  "criterion_profile": "relation_scope_v1"
}
```

两段新标准及16个通用结构变体固定在 `tests/fixtures/granite_byoc_variants.json`；原默认guardian另有逐字fixture。变体覆盖分列表格对象、合计、条件脚注、来源给定换算、历史污染、两类拒答、支持仅在别处、未知ID组和联合来源。预设期望是合成世界定义，不是人工标签；CPU测试用统一mock响应而非按期望返回正确值，因此不能从测试通过推得模型准确率。这批变体没有使用产品/任务专名或正式test答案，也不是独立meta-test。

已导出 `runs/granite-byoc-variants-v1/{audit,labels}.jsonl`，标签与audit分开。`bench.granite_judge`可直接读取该16条audit，标签文件不是其参数。较早同29项的完整回归driver支持多profile，以下命令**默认只输出计划，0次HTTP**：

```bash
python runs/run-granite-development.py \
  --inputs runs/atomic-development-v3 \
  --output runs/granite-byoc-development-v1 \
  --base-url http://127.0.0.1:30001 --context-window 32768 \
  --tokenizer-path models/granite-guardian-4.1-8b \
  --profiles ungroundedness relation_scope_v1 citation_binding_v1
```

显式加 `--execute` 后按所列顺序执行，每profile单独保存同29个输入、原标签sidecar、config、manifest、runtime、ledger与execution。无引用行仍在对应ledger，不选取改善的样本；一组失败后保存失败并继续其余预设组，最终任一真实失败使进程非零退出。`not_covered`单列而非失败。顶层 `profiles.json`保存顺序与结果，不作共享cache环境下的速度因果比较。新BYOC标准和16变体尚未真实模型验证；原29是已知开发回归，后续需冻结独立meta-test才可判断泛化。

本地52项Granite合同测试、native/quality联合99项及6子测试通过。固定官方tokenizer将原29个默认输入重建为与既有v2逐token相同的IDs；新relation/citation预检最大输入分别4,229/3,999 tokens，均满足32,768窗口与64输出预留。同29引用profile覆盖23条、未覆盖6条，派生预算记录在 `runs/granite-development-review/byoc-prepared-budget.json`。这些均是执行前工程检查，尚无新增模型准确率结论。
