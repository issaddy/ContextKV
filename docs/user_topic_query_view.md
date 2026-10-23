# 用户历史主题补全：可选检索视图

`query_mode: user_topic` 已接共享 P3 retriever factory、controller/native CLI 和离线检索评测。它保留当前用户问题原文，在少量显式后续问题后附加历史用户原文中的命名短语；不生成新事实，也不修改生成阶段的 query、完整 QA 或证据。默认仍为 `current`，原 BM25 与其他视图不变。[五视图真实检索对照](user_topic_retrieval_validation.md)已完成820条结果核验：SDK来源改善，同时保留WebSocket定义候选丢出预算的反例；回答质量尚需完整P3验证。

## 为什么增加这一对照

[三臂完整 P3 回归](table_focus_p3_validation.md)中，三组最后都将其他产品的 SDK 资料当成 Cloudant。第 6 轮**用户**明确提到 IBM Cloudant，第 9 轮却只问 Python SDK；当前 `correction_aware` 在没有纠正线索时使用 current，因此丢失产品锚。`last3` 只拼第 7、8、9 轮，同样漏掉第 6 轮；`all_user` 可以保留它，但也带入更早的附件、文件大小等问题。这正好形成 current、last3、all_user、correction_aware 与选择性补全的可检验对照，不预设复杂方法必胜。

该错误还涉及生成器忽略产品适用性；把检索 query 补完整只是一个干预入口，不能独自证明后续 SDK 示例可靠。

## 机制和边界

纯函数 [`user_topic_view`](../adapters/query_views.py)只接收截至当前轮的用户问题序列：

1. 首轮、明确换话题、当前已有命名文字时保留 current。词法命名线索包括连续大写开头词、混合大小写或带数字的标识符；普通问句/话语引导词排除。单个首字母大写词只用于当前问题的回退判断，不作为历史主题带入。没有产品或 task ID 字典。
2. 显式纠正且没有上述当前命名文字时，使用现有最近三个 user questions 的视图，保留原否定文字；不把否定的对象重新断言为正向主题。
3. 其他含代词、明确后续话语或比较线索的问题，寻找最近一个有命名线索的历史用户轮，将该轮的原文短语附在当前问题后。遇明确历史换题/否定边界停止；多个并列短语保留，不假装已经选出了唯一对象。
4. 没有线索、无法解析的否定或没有历史命名文字时保持 current。输出附 `User topic context:`，其含义是用户话题上下文，不是外部证据。

每个返回值保存 `route`、`reason`、`source_user_turns`、`selected_history_spans`（原 user turn、start/end、text）、当前命名线索和必要边界轮。字符跨度可以在对应用户原文直接回查。`HybridRetriever` 的最终 manifest 记录实际 query 和每轮 trace；离线 `retrieval.jsonl` 同样记录 trace。版本使用普通配置说明 `literal-user-topic-v1`，不增加逐请求 hash 体系。

这是一组**可解释的英文词法线索，不是可靠实体识别或完整意图消解**：

- `Python SDK` 这样的多词工具名也会触发“当前命名文字”，从而不补历史产品；同一句的大小写会改变路由。
- 全小写产品名、简称、单词拼写错误可能漏识别或保留错误拼写。函数不自动纠错、不扩写缩写。
- 没有明确标记的换题可能仍带入旧主题；历史中有多个对象时，无法判断究竟指哪一个。
- 否定词并不总是否定旧对象。例如开发集的 “I do not know how to set up namespaces” 当前回退，不解决这类隐含关联。回退是可观察的局限，不是已证明问题不需要历史。
- 单个命名短语可能省略版本、条件或操作目的，因此 all_user/last3 仍是必要的强简单对照。source span 只证明来自用户，不能证明该用户陈述是真实世界事实。

当前规则只改变检索视图，继续复用同一 BGE-M3、BM25/RRF 和 BGE reranker。没有额外模型下载、LLM 调用或答案缓存；online 的完整 retrieval 时间包含视图构造与锁等待。离线额外保存 `query_view_ms`，各阶段 `elapsed_ms` 包含一次该视图构造，重用阶段的耗时不能再相加成独立并发基准。

## 官方相关方法核查与复用选择

| 方法与证据 | 可用原理与本项目取舍 |
|---|---|
| [Dawn，SemEval 2026 §3 / Appendix A](https://aclanthology.org/2026.semeval-1.363.pdf) | 将从会话恢复请求意图，与从参考段落抽取可用事实分开。这里适配其信息来源分工：用户历史提供检索主题，检索正文提供回答依据。没有复制其多次 LLM 分类/生成流程；官方补充材料仅为论文源码，不宣称已直接复用完整 Dawn 系统 |
| [Sifei，SemEval 2026 §3.1/3.5](https://aclanthology.org/2026.semeval-1.32.pdf) | 当前问题与独立改写拼接，结合混合检索和 BGE reranker 1024 budget，为“保留当前措辞、再补上下文”提供同场景依据。原系统用 GPT-5.2 改写、不同候选数及语料清洗；清洗还保护 qrels 对应段落，这不进入本项目在线路径。论文给出的作者仓库本次返回 404，没有复制代码或移用其收益数字 |
| [JCT，SemEval 2026 §3 / §4.2 / Appendix A](https://aclanthology.org/2026.semeval-1.173.pdf)；[官方源码固定版](https://github.com/TalFarhan/mt-rag-task-a/tree/1eb63b3439e6b4379895321c73c89cc00a7c9dca) | 1.5B Qwen 改写与完整历史双路径检索，论文同时分析旧实体强加、关键限定词丢失及未明确问题被过度补全。[`scripts/rewriter.py`](https://github.com/TalFarhan/mt-rag-task-a/blob/1eb63b3439e6b4379895321c73c89cc00a7c9dca/scripts/rewriter.py)使用 4-bit 模型、最多 60 个生成 tokens，并接收完整角色历史；固定仓库未见独立 LICENSE。它可作为后续本地生成式改写对照的 API 参考，但尚未迁移：需要限制用户历史输入、固定模型/许可、保留原 query，并计入额外模型加载、prefill/decode 和失败。当前没有复制其 prompt/code |

新增代码是独立的词法适配；直接复用的是工程已有的用户历史边界、纠正检测、检索工厂和 BGE 排序路径。上述论文不能为这组新词法规则提供未经测量的效果保证；也不因新增视图较轻便便排除后续本地改写或语义历史选择。

## 全部 41 轮开发视图

`runs/user-topic-dev-review/prepare.py` 只读取已保存 off 开发 audit 的 query 与 user history，未读取 test 示例、targets、qrels，也未以助手答案选择路由。输出 `queries.jsonl` 保留五种视图和原文字段，`summary.json` 保存覆盖与本地 CPU；这是视图构造诊断，不是检索指标。

| 新视图路径 | 轮数 | 实际情况 |
|---|---:|---|
| current | 37 | 5 首轮、19 当前命名文字、10 无后续线索、2 无历史命名线索、1 未解析否定 |
| user_topic | 3 | b461 第4轮带入第3轮 `WebSocket API`；d5 第7/9轮带入第6轮 `IBM Cloudant` |
| last3 | 1 | d5 第5轮附件纠正，与现有 last3 文本相同 |

平均字符数分别为 current 56.20、last3 149.88、all_user 261.56、correction_aware 66.05、user_topic 60.93。25 次本地重复的逐轮 CPU 中位数再平均约 0.008 ms，最大约 0.035 ms；不是 BGE tokenizer 长度，也不替代 GPU 服务延迟。全部选中 span 均与原用户文字相符。与 correction_aware 相比，两个已明确写出 CocoaPods 的纠正轮会改用 current，这也要进入回归，不能只看 SDK 目标。

## 可执行对照与工程验收

远端由运行方在相同语料、索引和权重上执行；以下命令没有在本地启动 GPU：

```sh
python -m bench.retrieval_eval evaluate \
  --input data/mtrag/RAG.jsonl --split-manifest data/mtrag/split_manifest.json --split dev \
  --sparse-index data/cloud-bm25s --dense-index data/cloud-bge-m3 \
  --encoder-path models/bge-m3 --reranker-path models/bge-reranker-v2-m3 \
  --device cuda:0 --dtype bfloat16 --batch-size 16 --cpu-threads 1 \
  --encoder-max-length 8192 --reranker-max-length 1024 \
  --candidate-k 50 --rerank-k 50 --rrf-k 60 \
  --query-modes current last3 all_user correction_aware user_topic \
  --qrels data/mtrag-official/mtrag-human/retrieval_tasks/cloud/qrels/dev.tsv \
  --output runs/cloud-dev-user-topic-retrieval-v1
```

各视图使用相同候选预算，但 query 不同会产生不同候选池；这是整条检索路径对照，不应声称固定了完全相同的候选文档。先完成所有排序再读取 qrels，保留缺失标注轮；比较全部开发轮、逐会话、CocoaPods 两个变更轮和 SDK 目标，报告 top5 与候选池指标、截断/到 cap 情况和完整检索成本。随后以同一视图单独执行 P3 才能判断答案是否改善；不以 retrieval recall 或源 ID 出现代替回答质量。

线上启用仅需在现有 retriever YAML 中设置 `query_mode: user_topic`，其余 sparse/dense/model 路径和预算保持对照值；controller 与 native 都通过 `--retriever-config` 使用该共享配置，生成问题和完整历史保持原样。

相关测试 **55 passed**（另一个会读取完整可选数据集的既有 cohort 测试未重复运行），Ruff 通过；覆盖原模式逐字兼容、用户 span、不同名称、并列对象、否定/换题、大小写/拼写局限、助手污染不进入 query、encoder/reranker 使用同一实际 query，以及 controller/native/离线 CLI 的 manifest 和后置 qrels 边界。mock 只验证工程合同，不提供模型收益分数。
