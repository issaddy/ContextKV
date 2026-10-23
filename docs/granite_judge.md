# Granite Guardian 4.1原生诊断入口

`bench.granite_judge` 是独立离线groundedness风险适配，复用固定IBM模板文字与已有 `NativeSGLangClient`，没有新HTTP框架。默认no-think、64输出tokens，先服务已有29项开发probe；不修改在线答案、不进入五维quality gate。模型/作者固定版本、格式差异与依据见[独立核查](goal_acceptance_audit.md#五granite-guardian-41原生接入核查)，Apache归属见[NOTICE](third_party/GRANITE_GUARDIAN_NOTICE.md)。

输入从完整audit读取：全部visible正文通过官方 `documents` 槽位，按稳定reference ID关联；即使不是本轮top5，只要仍可见也会保留。仅归档而不可见的材料不进入documents，缺正文则失败。实际history保持user/assistant角色，当前问题、本轮待评答案和末尾guardian块单列。过去assistant断言是对话上下文，不是source。没有targets/labels输入参数。

固定本地tokenizer的 `apply_chat_template(..., documents=..., tokenize=True, add_generation_prompt=True)` 只编码一次；完整IDs直接送原生 `/generate`，不传session ID、不在服务端再套chat模板。请求返回prompt IDs并与提交值核对。部署窗口必须显式配置，区别于模型config的131072上限；输入加输出预算超限会完整保留为失败，不静默截断。

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
