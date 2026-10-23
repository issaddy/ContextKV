# GroUSE 元评测入口

`bench/grouse_meta.py` 用现有 httpx 调用本地 OpenAI-compatible endpoint，执行固定 GroUSE 四模板 JSON 协议。不引入 LiteLLM、datasets 或运行时自动安装；模板渲染需要 `evaluation` extra：

```bash
uv sync --extra evaluation
# 或在已有环境中：pip install '.[evaluation]'
```

这项元评测检查 evaluator 是否满足 GroUSE 的预期条件，不是 Cloud 回答质量评分，也不是 Selene 原生 `**Result:**` rubric 协议。后者由独立的 native-judge 适配器负责。

## 固定来源与复用

官方仓库固定在 [`e4c57ba9eb14548c7642daca3ed4ff24d760c7fa`](https://github.com/illuin-tech/grouse/tree/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa)。四个模板原样保存在 `bench/assets/grouse/`，同目录保留完整 [MIT 许可及作者归属](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/LICENSE.txt)、`source_manifest.json` 中的文件 hash 和改编逻辑来源。实现复用官方条件调用、answer_2 读取、nullable 推导和六指标条件通过率，没有替换上游提示正文。

数据集固定为 `illuin/grouse@caaaae0dacf5126279ae59609d26926e124b8032`。每次只下载指定的 `train.jsonl` 或 `test.jsonl`，不先加载 DatasetDict。也可传精确的本地 split 文件；入口用该 revision 的 Git blob ID 校验一次，因此不能将另一个 split 或修改后的内容冒充固定数据。当前已下载 train 16 条至 `runs/open-judge-research/grouse-train.jsonl`；实现期间只取得 test 文件元数据，未读取 test 样本正文。

## 调用与判定合同

先依次调用 answer relevancy 和 completeness。Relevancy 非 null 时，usefulness 记为不适用并调用 faithfulness；relevancy 为 null 时先调用 usefulness，只有其值非 null 才继续 faithfulness。正常路径三次或四次调用。Relevancy 失败仍保留 completeness 的独立调用；依赖失败沿原流程记 Failed。

- 模板中的 `answer_1` 是参考回答，`answer_2` 是被评价回答；最终读取 answer_2。Relevancy/completeness 为 1–5 或 null；usefulness 为 0/1 或 null。Faithfulness 模板明确请求 boolean/null，按官方 DTO 的行为将 true/false 规范为 1/0，也兼容数值 0/1。
- positive acceptance 与 negative rejection 只由 relevancy/completeness 的 nullable 状态推导，不增加模型调用。Expected conditions 的 nullable 状态同样只用于离线推导目标。
- Failed 与 null 分开保存。六项条件通过率都以整个选中 split 为分母；Failed 贡献 0，不删行。总体为六项通过率的算术平均。
- Prompt 只渲染 `input`、`actual_output`、`expected_output` 和 `references`，各模板按官方需要使用相应字段。`conditions`、`metadata.goal/test_type` 仅保存在离线记录，不进入模型请求。
- 保留完整引用正文，不按 top-k、字符或 token 数截断。该入口不自行估算 tokenizer budget；服务端超限等 HTTP 错误记 Failed。若外部使用 Transformers 5.19 做预算检查，需显式 `apply_chat_template(..., return_dict=False)`，不能把返回字典长度当 token 数。

解析适配明确命名为 `bounded-json-scores-v1`：保留官方首个 fenced JSON 提取和缺失可选 score 默认 null 的语义；另外检查模板规定的数值范围，拒绝越界及数值字符串，避免把诸如 relevancy=99 误计为通过 `>=3`。Faithfulness 的布尔转换如上。这比原始 Pydantic DTO 的通用整数转换更严格，报告不能省略该差异。真实 HTTP 响应只有 `finish_reason="stop"` 才参与评分；length、缺失终态或其它终态均记 Failed，且保存 raw response、usage、原因和调用时间。不自动修复 JSON 或重试选择性失败。

## Train 开发、冻结、Test

先保存配置，例如 `runs/grouse-selene-config.json`。`model` 应与实际 endpoint 的 served model 名称一致；revision 是操作者声明，单凭 HTTP 响应不能证明权重 revision：

```json
{
  "provider": "local_sglang",
  "model": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
  "revision": "427792f1c3e2073cb7da216924fd884b1ba496e0",
  "base_url": "http://127.0.0.1:30001/v1",
  "temperature": 0.01,
  "max_tokens": 2048,
  "timeout_seconds": 120,
  "concurrency": 1
}
```

该采样默认值来自官方普通模型调用；若开发时改为 temperature=0 或其它预算，新值会进入冻结配置。API key 只从命名环境变量读取，不写入报告。不自动启动或下载模型。

```bash
python -m bench.grouse_meta evaluate \
  --config runs/grouse-selene-config.json --split train \
  --dataset-file runs/open-judge-research/grouse-train.jsonl \
  --output runs/grouse-selene-train-v1

python -m bench.grouse_meta freeze \
  --train-run runs/grouse-selene-train-v1 \
  --output runs/grouse-selene-frozen-v1.json

# 仅在完成开发并决定冻结后执行；这是首次读取该 split 正文的入口。
python -m bench.grouse_meta evaluate \
  --frozen-evaluator runs/grouse-selene-frozen-v1.json --split test \
  --output runs/grouse-selene-test-v1
```

Test 必须提供由完成的 train run 产生的 frozen evaluator。配置、当前实现源码、parser 标识、模板内容和 dataset revision 的一致性在读取 test 数据前检查。冻结表示固定评测协议，**不是根据 train 分数自动认定 evaluator 合格**；同样不能证明操作者从未看过 test。若后续根据 test 调参，应把它标为已使用的开发证据，不能继续声称未见测试集。

每次输出必须是新目录，包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | evaluator/config/source/template 身份、单 split 来源、train 或 frozen-test 模式、完成状态 |
| `results.jsonl` | 每条样本、离线条件/元数据、完整六指标状态、逐调用 raw response/usage/耗时；完成一条即 flush |
| `summary.json` | 六项通过率、总均值、Failed 数、实际调用解析率、已观测 token usage 与调用耗时 |

中途异常保留已写行并标 `incomplete`，不能拿部分样本冻结。逐模型响应失败仍完成该样本的规定流程并进入分母。Usage 缺失时为未知，并记录观测调用数，不能当作零 token 消耗。

## Python 接口与本地验证

`GroUSEEvaluator(call)` 接受异步 `call(messages) -> response`。真实 transport 返回 OpenAI response dict，以保留终态和 usage；测试或已显式负责完整性的 adapter 可返回字符串。`HTTPCall(GrouseConfig(...), transport=...)` 支持 httpx MockTransport，便于不启动模型的合同测试。

`evaluate_sample(sample)` 返回六指标状态与调用记录；`evaluate_conditions(evaluation, conditions)` 只做离线比较，支持官方数值比较和 `==None`，不执行任意表达式。完整运行用 `evaluate_run`，不需要把主项目质量评分的五维 JSON 套到 GroUSE 六项上。

39 项 fake 边界测试覆盖三/四调用分支、nullable 推导、Failed 传播与分母、faithfulness boolean、越界/缺字段/截断、条件和 goal 不泄漏、完整 references、单 split 来源、先冻结后读 test、独占输出及缺少可选依赖。独立审查另核对了 nullable acceptance 组合。这里只报告工程验证，没有产生真实 judge 元评测成绩。
