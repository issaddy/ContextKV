# 证据支持重评分：开源复用与最小工程扩展

2026-10-24 的优先扩展是**本地、逐论断的引用—证据支持重评分**。它直接针对已观察到的错绑与资料不足，先形成离线可审查信号，再据此试验选择性纠偏。现已实现 `bench/grounding.py` 的最小适配；本轮没有运行真实权重，模型分数有效性仍待目标设备与开发数据验证。没有修改默认提示词、线上答案或核心调度器。

## 为什么选支持评分，而不只继续排序或改提示词

保存的 Q0/P1/P2 与 Q1/Q2 结果已经提供三类可区分问题：

- Cloudant 附件 10MB 的核心事实有依据，但回答引用了 Security and Compliance Center 的 Attachment 50/account。需要检查 **claim 与特定 ID 的绑定**，仅问“检索材料是否相关”不够。
- Cloudant SDK 的完整可见集合实际是 Analytics Engine、Monitoring 等资料。没有排序算法能凭空补出缺失来源；需要识别资料不足或跨产品用途错配。
- CocoaPods 的修复文本在相邻正确片段，但回答引用前段，或把同一检索块中的 Android Gradle 步骤搬到 iOS。块级词面相似度可能很高，仍需要保留具体来源、章节和条件范围。

这些是已分析 dev，可用来训练探索、校准和构造回归案例；不能用原答案中的 ID 是否存在自动生成支持真值。独立 test 保持冻结。具体原始证据和 24 条定向阅读判断见[共享提示词验证](shared_prompt_validation.md)。

## 比较的可复用方法

| 方法 | 核查到的能力与工程入口 | 对本次缺口的利用价值 |
|---|---|---|
| MiniCheck | 文档—句子二分类，直接检查 grounding；作者发布非生成式 DeBERTa/RoBERTa 与 Flan-T5 等权重和 `MiniCheck.score(docs, claims)` API。[作者论文](https://arxiv.org/html/2404.10774v1)、[官方代码](https://github.com/Liyan06/MiniCheck) | 当前优先：与错引数据的输入/输出最吻合，可在已保存完整 audit 上运行；不能把它的模型输出当独立人工真值。复合句、多文档联合支持、指代与代码仍需要检查 |
| CRAG | 查询—检索材料 evaluator 决定不同处理动作；论文使用 T5-large，作者实现含内部片段筛选与外部搜索。[论文 §4](https://arxiv.org/html/2401.15884v3#S4)、[固定内部处理入口](https://github.com/HuskyInSalt/CRAG/blob/de7c2961ae624a1483a138c5798e1f6d0c4fb0e0/scripts/internal_knowledge_preparation.py) | 复用“置信信号触发不同动作”的原则；其 relevance 不等同当前 claim 的支持。官方外部搜索分支需要 OpenAI/Serper；本次不调用。该仓库快照未确认明确许可，不复制其实现 |
| Self-RAG | 联合相关性、支持和效用反思；官方训练 critic/generator 并引入特殊反思 token。[官方训练/推理说明](https://github.com/AkariAsai/self-rag/tree/1fcdc420e48f50a7d7ab1ece5494221b93252e99) | 支持与相关性分开的设计值得复用；完整训练与 segment search 是更大的可选路线，不能把其 token 打分直接当成普通 Qwen 已有能力 |
| 开源 cross-encoder reranker | `CrossEncoder.predict(query, passage)` 是成熟的局部相关性重排入口；MS MARCO MiniLM 等模型可直接部署。[官方文档](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) | 适合后续检索排序对照，可能抑制无关产品；它不验证“SDK 属于哪个 API”这样的生成论断。与 MiniCheck 可比较，不把 sigmoid relevance 当已校准支持概率 |
| 官方 ContextPilot 索引 | 官方实现提供在线/无状态重排和索引，排名注释保留检索重要性。[官方 API](https://efficientcontext.github.io/contextpilot-docs/reference/api)、[原论文](https://proceedings.mlsys.org/paper_files/paper/2026/hash/b0131b6ee02a00b03fc3320176fec8f5-Abstract-Conference.html) | 可以作为真实索引复用分支推进，检查跨会话复用与查询尾部注释；这是缓存/布局能力。它与支持评分解决不同缺口，不需要为了集成其中一个禁止另一个探索 |

这次选择只决定先实现哪个小模块，不宣布其他路线无用。首先获得可用的局部支持信号，再与 query-only relevance、现有稳定排序及官方索引复用比较质量与成本。

## 固定代码、模型与实际适配

- 官方仓库：`https://github.com/Liyan06/MiniCheck`，固定 `b58b9fa69acbd1015ec970fa65dd752413a053d2`；[许可证](https://github.com/Liyan06/MiniCheck/blob/b58b9fa69acbd1015ec970fa65dd752413a053d2/LICENSE)为 Apache-2.0。
- 首选 checkpoint：[`lytang/MiniCheck-DeBERTa-v3-Large`](https://huggingface.co/lytang/MiniCheck-DeBERTa-v3-Large/tree/2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb)，revision `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb`，官方 model card 标 MIT。
- 可比较 checkpoint：[`lytang/MiniCheck-Flan-T5-Large`](https://huggingface.co/lytang/MiniCheck-Flan-T5-Large/tree/96eafd01cee2d16cf81aaa2fb226b14f422a37b3)，revision `96eafd01cee2d16cf81aaa2fb226b14f422a37b3`，官方 model card 标 MIT。

适配直接复用官方 `MiniCheck.score` 和 `Inferencer.fact_check/inference` 的分块及支持概率计算。官方 initializer 使用 `revision='main'` / `device_map='auto'`；本地小型子类只替换加载部分，接受明确的本地 snapshot 和 device。官方代码的 AlignScore 来源说明保留在原仓库。本项目记录固定预期模型 revision，权重来源标记为 `operator_supplied_local_snapshot`，不伪称重哈希验证了所有权重。

模型分数是 `supported / unsupported` 信号，并不区分事实矛盾、来源不提和模型不确定性；默认 0.5 是上游诊断阈值，不是 Cloud 场景校准结果。当前先使用非 LLM 分支，无需额外 vLLM 服务、API key 或付费 judge。

## 已实现的离线入口

```bash
# 默认只导出原文配对，不加载 torch、MiniCheck 或模型。
python -m bench.grounding \
  --audit runs/example/audit-<run>.jsonl \
  --output runs/grounding/pairs-dry.jsonl

# 显式使用事先准备好的本地权重；不会在运行中下载模型。
python -m bench.grounding \
  --audit runs/example/audit-<run>.jsonl \
  --output runs/grounding/deberta.jsonl \
  --model deberta-v3-large \
  --model-path models/MiniCheck-DeBERTa-v3-Large \
  --minicheck-root vendor/minicheck \
  --device cuda:0 --max-length 2048 --chunk-size 400 --batch-size 8
```

源 audit 必须含全部 `visible_evidence_manifest` 与 `audit_evidence`。程序不从 official top-k 文件猜测完整输入。先对实际引用块评分，再对其他可见块评分，得到引用支持矩阵；例如 cited block 无支持、旧可见 block 有支持时输出 `supported_elsewhere_citation_mismatch`。多个引用中仅一部分有支持、完全缺引用、未检测到支持都分别保留。

分句使用可查看的保守 prose span：原回答 start/end 偏移不变；fenced code、表格、标题和单独引用行不当作已验证论断。自然语言指代、复合句、未标记的命令和 inline code 仍可能影响模型，输出注明评估范围，需要阅读复核，不自动原子化改写。完整正文不修改；上游分块供评分使用，记录每块输入 token 数、原块文本和超长截断。截断块的高概率不能变成 `support_signal=True`，相关论断标记覆盖不完整。

输出为可审查 JSONL 与 manifest，记录 device/dtype、代码及模型版本、长度参数、分块、分数和状态。已有输出不覆盖，出错时保留已写记录及失败状态；没有总体质量分，不自动换引用、删句或修改原回答。合同测试使用 mock scorer 验证映射和状态，其分数不代表模型效果。

DeBERTa 与 Flan-T5 默认 `max_length=2048`，直接遵循固定官方 initializer 与 API 文档。DeBERTa checkpoint 虽有 `max_position_embeddings=512`，同时为 `position_biased_input=False`、`relative_attention=True`，不能把普通绝对位置模型的 512 硬限制套用到这里。先以 >512、≤2048 的实际输入做 runtime smoke；512 作为显式长度敏感性对照，而非无依据降低默认。运行成功与长输入评分质量分别记录。官方 DeBERTa 分块近似按 tokenizer token 数，Flan-T5 分块按词数；两者的 `chunk_size=400` 不能解释成相同输入长度。单个很长句可能超过分块目标，实际 tokenizer 截断仍逐块记录。

依赖隔离：dry-run 仅用已有项目依赖；显式模型需要固定 MiniCheck 源码、`torch`、`transformers`、`numpy`、`nltk`、`tqdm`，具体 tokenizer 可能需要 `sentencepiece`。NLTK `punkt_tab`/`punkt` 在准备阶段提供；adapter 只检查，不运行下载。没有修改 `pyproject.toml`、`uv.lock` 或现有 torch 环境，不安装整个上游 requirements。真实分类器运行需记录与 SGLang 并存的显存/时间影响，结束后释放分类器再跑性能基线。

## 下一步实验：积极探索与公平对照

1. **先验证可用信号。** 用现有完整 dev audit 与已知错绑/正确引用案例跑本地分类器；为覆盖常规句、版本条件、否定、跨产品、代码及资料不足补阅读标注。Q0/Q1/Q2、P1/P2 和同会话所有轮次同组，做会话 OOF 的阈值/模型选择；已观察数据用于开发，正式 test 不参与。
2. **保留可证伪对照。** 比较 ID 存在规则、词面匹配、query-document reranker 与 MiniCheck claim-document 支持。指标包括错绑检出、误报、未覆盖、引用实际支持及成本；不只报模型平均概率。正确答案含错误额外 ID 与答案本身无根据要分开。
3. **再做有限纠偏实验。** 从同一冻结初稿和同一可见材料开始，比较原稿、所有稿统一一次修订、仅被支持模型标记的稿修订。修订只作为独立离线实验产物，最多一次本地 Qwen 调用；保留前后答案与额外延迟。实际来源缺失时不能把另一产品的来源补成“正确引用”。线上 draft/最终提交边界属于后续工程扩展，不在当前模块偷偷改历史。
4. **与缓存主线协同。** 对最终选定的检索/质量设置，再共享它比较 APPEND/REBUILD 和生命周期策略；如果改变证据选择，则单列资料选择实验，并共享语料、问题、模型、输出预算与资源条件。纠偏调用、分类器开销、输出变化和缓存影响全部计入完整成本，不能只把更短回答算收益。

预期增量是：利用可复用 grounding 模型和现有精确可见性账本，定位错误绑定、区分可修复引用与证据不足，再以选择性纠偏减少不必要的重复生成。支持矩阵本身是工程能力；能否成为 A 的方法增量，取决于上述对照中实际获得的质量/成本改善。若效果不足，分析误报与未覆盖区间，继续比较模型、源片段组织或纠偏动作，不把一次负面结果等同于禁止后续探索。

## 已实现的引用纠正与真实整合检查

`bench.citation_repair` 提供可选的离线引用纠正：只处理完整评分、引用已知、无 inline code、旧引用全部缺乏支持而其他可见来源提供支持的 prose span。默认新来源概率至少 0.9、旧来源最大概率不超过 0.2；这是一组尚未校准的实验参数。只替换引用标记，保留所有回答措辞、原回答、前后位置及概率；不增加材料，不自动重写当前会话。

```bash
python -m bench.citation_repair \
  --grounding runs/grounding/deberta.jsonl \
  --output runs/grounding/repaired.jsonl
```

以相同 MiniCheck 重新评分这些结果不能作为独立质量验证，因为选择动作本身使用了该模型。需要对纠正项与未纠正项另做来源阅读核查、替代模型或人工盲审，并把误报一并报告。

H200 上第一轮 Q1/P2 41 轮、718 个 prose spans、3590 个 claim/source pairs 已执行成功，但发现引用标记进入 entailment 输入会产生系统性伪负例。对前 10 个含引用 spans 的 50 个来源配对，仅移除引用标记，超过 0.5 支持阈值的配对从 0 变为 6；一条原文直接引文对其正确来源的概率从 0.0133 变为 0.8851。这是输入整合诊断，不是分类器精度评测。

已修复为保留原文/offset/绑定，仅对 `semantic_claim` 移除标准引用标记，再送入官方评分；合同测试检查标记不会进入模型。旧结果完整保留，Q1/Q2 v2 正在重新评分。第一轮最大实际分块输入为 490 tokens，不能由它推断 >512 输入的评分质量或运行已获验证。
