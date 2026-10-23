# 本地独立开源 judge 的实施方案

本轮建议先用 **Selene Mini 8B** 做本地元评测，以 **Prometheus 2 7B** 为原协议对照，再为通过实际检查的 evaluator 接入已保存的 Cloud dev 回答。两者分别来自 Llama、Mistral 家族，与当前 Qwen2.5-7B 生成器不同；无需付费 API 或等用户填写人工分数即可推进。跨家族只能减少一种共同偏差，不能证明评分可靠，也不能把模型评分称为人审。

截至 2026-10-24，本轮完成官方论文、作者仓库、固定模型元数据和本地质量代码审查；未下载大权重、未启动 GPU 服务、未产生任何新质量分数。小文件与访问记录在 ignored `runs/open-judge-research/`，请求准备文件在 `runs/open-judge-preparation/`。本方案独立于正在运行的 B 阶段计时实验。

## 现有链路与最小缺口

仓库当前没有 `bench/local_judge.py`。现有入口是 `bench/judge.py`：它已有 OpenAI-compatible HTTP、固定 model/revision/prompt/temperature、完整可见证据与实际历史、规则 flags、query/input_hash 绑定和失败记录。但它强制一次返回五维 JSON，不能直接保持专用 judge 的原生 rubric 格式。新工作应是一个薄的本地模型 prompt/parser 适配器，复用这些输入与输出边界，而非再建一套验收框架。

`bench/quality_acceptance.py` 已核对真实 run、audit、答案、提示身份和划分，并区分自评、开发集准入、heldout 模型 evaluator 非劣与人工质量结论。后续仍保留 `scores` 五维完整键、NA 和失败；新增原生分数/推理文本作为诊断。同底座不同微调模型目前可能被识别为 `independent_model_declared`，因此新 manifest 还应记录 base model family；首选跨家族模型，不能靠改路径或 served model 名“变成独立”。详见 [现有质量验收](quality_acceptance.md)。

## 已核实的候选与版本

| 模型 | 固定模型 revision | 可访问与许可记录 | 本次用途 |
|---|---|---|---|
| `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `427792f1c3e2073cb7da216924fd884b1ba496e0` | API/config/tokenizer 200、gated=false，首个 safetensors 分片未鉴权 HEAD 200；模型卡与作者代码声明 Apache-2.0，保留 Llama 3.1 基座身份 | 首选：官方支持绝对分数、二分类、配对、有/无参考答案 |
| `prometheus-eval/prometheus-7b-v2.0` | `66ffb1fc20beebfb60a3964a957d9011723116c5` | 同上访问成功；模型卡与代码 Apache-2.0；Mistral-7B-Instruct-v0.2 基座 | 对照：rubric 1–5 与 A/B 专用判分 |
| `Unbabel/M-Prometheus-7B` | `030fb74806e4228c466a98706a297d43b31ce5df` | 同上访问成功；模型卡 license=`other`，此次未发现单独许可正文 | 较新多语言备选；Qwen2.5-7B 底座，不作为主要跨家族证据 |
| `Skywork/Skywork-Critic-Llama-3.1-8B` | `825f34599593c0145be91644be233d5c634b2380` | API/config 200；官方仓库为 33 个 `.bin` 权重分片；卡片标 Skywork Community License | 主要面向配对偏好，暂不优先用于五维绝对评分 |

访问成功表示本次元数据/一个权重分片的可访问性，不等于已下载完整模型或实测可加载。固定来源：[Selene 模型](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B/tree/427792f1c3e2073cb7da216924fd884b1ba496e0)、[Prometheus 2 模型](https://huggingface.co/prometheus-eval/prometheus-7b-v2.0/tree/66ffb1fc20beebfb60a3964a957d9011723116c5)、[M-Prometheus 模型](https://huggingface.co/Unbabel/M-Prometheus-7B/tree/030fb74806e4228c466a98706a297d43b31ce5df)、[Skywork 模型](https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-8B/tree/825f34599593c0145be91644be233d5c634b2380)。许可栏忠实记录发布者声明，不因基座或仓库代码许可自动补全权重授权条件。

Selene 的作者研究评估通用 rubric 判分；Prometheus 2 研究同时支持直接与相对评分；M-Prometheus 扩展多语言 judge。论文上的总体相关性或通用任务成绩不直接证明本项目的表格条件、代码、历史错位和必要拒答可靠。[Selene 论文](https://arxiv.org/abs/2501.17195)、[Prometheus 2 论文](https://arxiv.org/abs/2405.01535)、[M-Prometheus 论文](https://arxiv.org/html/2504.04953v2)。

GroUSE 官方 README 链接的 `illuin/llama-3-grouse` 本次未鉴权 API 返回 401；不能据此区分私有、受限或不可用，也不能把它列成已就绪依赖。[官方链接所在仓库](https://github.com/illuin-tech/grouse/tree/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa)。

## 显存与已有推理接口

Prometheus 2 为 7.242B BF16 参数，官方权重文件约 14.483 GB；Selene 为 8.030B BF16 参数，约 16.061 GB。权重本身约 13.49 / 14.96 GiB。两者配置均为 32 层、8 个 KV heads、head dimension 128；TP=1、BF16 KV 时，每 token KV 约 128 KiB，32,768 slots 约 4 GiB。**24–32 GiB 是单并发 32K 上下文的初步资源预算，非实测最低显存或峰值**，还需包含激活、图和 allocator。Selene 配置支持 128K，但不能把 32K 预算直接用于 128K。

已有固定 SGLang `3f36fc08331f64cd63c4fbbdf39cc402f38bb38b` 源码包含 MistralForCausalLM 与 LlamaForCausalLM；因此可复用当前隔离 GPU 环境，不必为此再安装 vLLM、FastChat 或完整 Prometheus Python 包。这里只验证了源码支持，实际加载仍需首次运行确认。[固定 Mistral 实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/models/mistral.py)、[固定 Llama 实现](https://github.com/sgl-project/sglang/blob/3f36fc08331f64cd63c4fbbdf39cc402f38bb38b/python/sglang/srt/models/llama.py)。

后续在性能采样结束后的独立时段，用固定 snapshot 启动 stock SGLang，建议独立端口 `30001`、初始 context/total tokens 32768、并发 1。不要直接复用固定 Qwen revision/fingerprint 的 `scripts/launch_sglang.sh`。原生 chat endpoint 按新模型自身 tokenizer chat template 渲染；不要手工再包一层 Mistral `[INST]` 或 Qwen 模板。保留实际 server info、模型 snapshot、启动参数与显存观测。

现存完整 judge JSON 的实测字符长度：correction P3 的 41 份为 22,216–48,885，16K Ours 的 41 份为 19,620–54,510；这不是 token 数。正式评分前用 judge 的固定 tokenizer 做一次入口预算核对。超限时记录失败或显式换更长上下文运行配置，不能静默丢掉旧可见证据。

## 原生判分请求与薄适配

Selene 官方绝对评分模板输出 `**Reasoning:**` 与 `**Result:**`，分数为 1–5；另有 reference/no-reference、Yes/No 和 A/B 模板。原生协议并非五维 JSON。[固定 reference 模板](https://github.com/atla-ai/selene-mini/blob/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de/prompt-templates/absolute-scoring-with-reference.yaml)、[固定无参考模板](https://github.com/atla-ai/selene-mini/blob/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de/prompt-templates/absolute-scoring.yaml)。

`runs/open-judge-preparation/prepare.py` 已从本轮取得的固定官方模板渲染两个合成样例，生成 `selene-absolute-scoring.request.json` 和 `selene-absolute-scoring-with-reference.request.json`。脚本没有网络操作，样例没有模型分数。请求结构为：

```python
request = {
    "model": "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
    "temperature": 0.0,
    "max_tokens": 1536,
    "stream": False,
    "messages": [{"role": "user", "content": official_template.format(**fields)}],
}
# 服务准备好之后，才发送：
# POST http://127.0.0.1:30001/v1/chat/completions
# 不加 response_format=json_object；保留该模型的原生 Result 格式。
```

实际 HTTP 示例（本轮未执行）：

```bash
curl --fail-with-body http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @runs/open-judge-preparation/selene-absolute-scoring.request.json
```

Prometheus 则使用 `[RESULT]` 后的 1–5 或 A/B。其固定官方 wrapper 默认 `ABSOLUTE_PROMPT_WO_REF`，仅传 `reference_answers` 不会自动切换 reference 模板；RAG 模板还有 `{relevant_context}`，但通用 `absolute_grade` 没有提供该格式变量。应显式渲染对应官方模板，不应误称“传了参考答案就已被模型使用”。[固定 prompts](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/prompts.py)、[固定 wrapper](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/judge.py)、[固定 parser](https://github.com/prometheus-eval/prometheus-eval/blob/dcfb44272d5d0414832f5dbb8c2a05ebc2614234/libs/prometheus-eval/prometheus_eval/parser.py)。

最小代码合同如下，后续由 root 实现：

1. 复用 `judge_payload`、`evaluation_binding`、`rule_audit`、已有 HTTP transport 和独占输出创建。每维独立 rubric，冻结完整模板、rubric、解析、参考答案策略和聚合版本；输出仍是原 `offline_judge` 绑定格式。
2. Selene/Prometheus 各自解析原生结尾字段，保留 raw response、native score、reason、finish reason 和 usage。有效 1–5 才映射 `(score - 1) / 4`。解析失败或输出被截断不补分；NA 不自动当 0。原生 1–5 模板没有 NA，因此若某维确实无法判断，需要显式适用性步骤或协议变体，经元评测后冻结；不能声称原模板天然支持 null。
3. 主评测先用完整可见证据和自身历史、无黄金参考答案。需要 reference 的独立轨道才离线 join targets，并记录 `reference_targets_supplied`；黄金答案不是模型当时可见的 source。脚本历史与实际自身历史不一致仍单列，不把参考答案适用于不同历史当作既定事实。
4. 可复用一次加载的模型身份、模板与预算检查；不新增逐请求多层 hash。冻结 evaluator metadata 应完整传到 acceptance，不能更换模板而沿用同一个配置身份。

## GroUSE：先验证 evaluator，再用于本项目

GroUSE 是具有预期行为条件的 GQA 元评测，不是另一套 Cloud 生成质量标签。官方固定 dataset `illuin/grouse` revision 为 `caaaae0dacf5126279ae59609d26926e124b8032`，train 16 个供 prompt 开发、test 144 个供冻结后的检查。本轮没有读取 test 样本正文。[固定数据集](https://huggingface.co/datasets/illuin/grouse/tree/caaaae0dacf5126279ae59609d26926e124b8032)。

论文报告 Prometheus 2 7B 的总体条件通过率为 **52.78%**，并讨论其 1–5 训练格式对 boolean/null 指标的局限。该数字来自原论文，不是本项目复测，也不证明 Selene 必然更好。因此本方案要先实际测，而不是凭通用 judge 名声确定正式 evaluator。[GroUSE 原论文](https://aclanthology.org/2025.coling-main.304/)。

源码并非固定“三个 prompt”，而是 **四个模板、条件执行**：先 relevancy 与 completeness；relevancy 非 null 时再 faithfulness；为 null 时先 usefulness，若 usefulness 仍适用再 faithfulness。通常三次，最多四次调用。positive acceptance 与 negative rejection 是 nullable 状态的确定性推导，不是再问两次模型。relevancy/completeness 为 1–5 或 null，faithfulness/usefulness 为 0/1 或 null；请求失败 `Failed` 与“不适用” null 必须区分。模板将参考答案与候选组成 answer_1/answer_2，读取 answer_2。[固定执行路径](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/grounded_qa_evaluator.py)、[固定 DTO](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/dtos.py)。

官方 CLI 的真实参数为以下形式，不存在单独的 `--base-url` 参数：

```bash
# 官方独立环境，MODEL_NAME 为该环境 LiteLLM 已注册的模型路由。
grouse meta-evaluate MODEL_NAME OUTPUT_DIR --train_set --prompts_path PROMPTS_DIR
# evaluator 冻结后，去掉 --train_set 才运行 144 个 meta-test。
```

原 `call_llm` 使用 LiteLLM、默认 temperature=0.01/max_tokens=2048。`load_unit_tests` 当前未固定 dataset revision，且先加载整个 DatasetDict 再取 split。因此不能直接跑未经约束的默认 CLI 并称版本/边界已冻结。[固定 CLI](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/main.py)、[固定 loader 与推导规则](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/utils.py)。

最少新依赖的实现是复用固定 MIT prompts、适用性与条件比较规则，使用已有 httpx/Jinja，单独读取固定 revision 的 train 文件；保存许可归属。也可以在隔离官方环境中仅替换 transport 和固定 split loader，保留 `GroundedQAEvaluator` 与 `MetaEvaluator`。不要为 16 个测试把官方 datasets/LiteLLM/diskcache/matplotlib 等整套依赖加入主工程。条件比较只接受源码已有的数值比较与 `==None`，六项通过率及均值保持原口径，Failed 计失败而不是删行。[固定元评测](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/grouse/meta_evaluator.py)、[固定依赖](https://github.com/illuin-tech/grouse/blob/e4c57ba9eb14548c7642daca3ed4ff24d760c7fa/pyproject.toml)。

judge 输入只含 `input`、`actual_output`、`expected_output`、`references`；**conditions、metadata.goal/test_type 不进入模型请求**。原始 GroUSE JSON 协议与 Selene 原生 rubric 协议分开命名：可先在 Selene 上运行原始四模板测指令遵循，也可开发 rubric/适用性适配版，但后者必须注明改动，不能把项目五维分数当作原版 GroUSE 六项。Prometheus 的格式局限同理，不强塞 JSON 后声称保留了作者协议。

## 下一步的有界执行顺序

1. 准备薄 adapter 与 fake HTTP 测试：正确/缺失/越界 Result、NA 适用性、finish_reason=length、引用正文与实际历史完整保留、评分绑定/失败不丢行。以本轮两个请求样例验证协议，不产生质量分数。
2. 性能采样结束后下载固定 Selene snapshot 并独立启动，先做 tokenizer 预算和极少量合成接口检查，再跑 GroUSE train 16。Prometheus 作为同预算候选对照；报告每项通过率、解析率、失败类型、token/时间成本，不把 16 例结果当可信区间保证。
3. 针对项目已见错误增加明确标为开发诊断的反例：附件 10 MB 与 50 MB、错误概括“1 MB 含附件”、把历史 assistant 断言当来源、无效 curl JSON、合理简短拒答与无依据长答。这些是规则/assistant 构造探针，不伪称新增人工标签。
4. 依据 train 元评测冻结 model/revision、四模板或适配协议、抽样参数、适用性与归一化，再运行 GroUSE test 144 一次。失败如实报告；若后续据此修改，明确它已转为开发证据，不能继续称未见元评测。
5. 对现有三组 P3 共 123 份实际回答完成五维模型评分，完整覆盖失败、NA、截断和困难案例，按现有会话配对验收。与 MiniCheck、规则 flags 的分歧用于定位缺口，不强行融合成“真值”。dev 可以用于继续优化方法；最终冻结的项目 test 保持独立。

此路线推进的是**独立开源模型评分证据**。通过固定 heldout 模型 evaluator 时使用现有 `heldout_evaluator_noninferiority_established`；不能设置仅限实际人工来源的 `quality_noninferiority_established`。它不要求用户先完成 123 份人工表，但也不把模型分数替代人类质量真值。

## 近期方法可复用的部分

2026 年 Autorubric 提供 rubric、分维聚合、agreement 等工程实践；可借鉴原子条件与显式聚合，但当前主工程已有 HTTP/绑定/配对统计，不必立即引入其 LiteLLM、statsmodels、sklearn 等完整依赖。[论文](https://arxiv.org/html/2603.00077v1)、[作者仓库](https://github.com/delip/autorubric/tree/3a85dc182a3b4a5ecf3a8eb4765b936d7645fd3e)。PReMISE 强调 rubric 的结构、偏好适配、可靠性和对抗稳健性不能只由高 agreement 代替；其论文链接的作者仓库本次返回 404，故只作研究启示，不列现成可运行依赖。[PReMISE 论文](https://arxiv.org/abs/2605.30803)。本轮到此收敛，不继续扩模型清单或启动训练。
