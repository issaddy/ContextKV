# 当前方法的复用依据

本文记录主线依赖及可执行的独立对照。任务边界、实现和运行结果分别见 [任务书](study/task_spec.md)、[架构](architecture.md) 和 [运行手册](study/final_runbook.md)。文献结果与本项目实测分开记录。

## A · 任务、来源与事实

混合检索复用 BM25 与 BGE-M3 dense 表示的 RRF，发布版交叉编码器为 query–passage 相关性重排。当前 ranked 入口按该顺序和来源依赖选材。Need 的主体、条件、版本和支持关系由项目任务与证据接口表达。支持模型及集合效用是可选对照，候选回答偏好监督复用固定 reader 的实际输出。[BGE-M3](https://huggingface.co/BAAI/bge-m3) · [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) · [REPLUG](https://arxiv.org/abs/2301.12652)

任务学生入口借鉴双线性图边评分，用于关联 Need、主体、条件与前驱。线上主线使用任务解释和来源合同，跨轮复用核对适用条件后导入原子事实。事实的原文坐标和推导父链分别用于来源定位与证明。[双线性边评分](https://nlp.stanford.edu/pubs/dozat2017deep.pdf) · [数据溯源](https://www.research.ed.ac.uk/en/publications/why-and-where-a-characterization-of-data-provenance/)

## B · 计算价值与有界后续

B 以当前直接 ANSWER 为基准，比较候选动作接固定后续的最终质量与完整成本。线上只执行选中动作，随后用实际状态重新选择。缺口检索沿未解关系和父事实触发，优先阅读已有就绪来源。此处借鉴计算价值、rollout 和检索与推理交替，实际收益由配对监督与完整队列衡量。[计算价值](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1988/5755.html) · [Rollout](https://web.mit.edu/dimitrib/www/dp_rollout_book.html) · [IRCoT](https://aclanthology.org/2023.acl-long.557/)

质量头使用岭回归，绝对成本头使用非负最小二乘与岭惩罚，直接拟合动作和原生报价所表达的非负工作量。已付成本用于约束预算；同一报价下的绝对分支预测用于计算相对成本。证据充分性用于定位检索、READ、最终窗口与回答各阶段的错误，最终质量使用实际答案及来源评价。[SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html) · [Sufficient Context](https://arxiv.org/abs/2411.06037) · [ALCE](https://aclanthology.org/2023.emnlp-main.398/)

## C · 请求编译与原生缓存

SGLang RadixAttention 和 HiCache 提供精确因果前缀匹配、GPU/Host 驻留与迁移，项目读取实际 Native 报价和完成回执。主四格中的 ContextPilot 因素复用作者 CPU 重叠索引与排列，候选顺序由项目编译成真实请求后报价。各排列的命中、回载、重算和完整响应以共同负载记录。[RadixAttention](https://arxiv.org/abs/2312.07104) · [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design) · [ContextPilot](https://arxiv.org/abs/2511.03475) · [复用范围](third_party/CONTEXTPILOT_NOTICE.md)

可选 TTL/value 在原生容量管理之上分配额外保护。保留目标借鉴有预算的覆盖和分层替换，在共同观察窗以 Native 为参照，核算共享页、增量驻留和控制成本。主矩阵使用 Native retention。[预算最大覆盖](https://doi.org/10.1016/S0020-0190(99)00031-9) · [RAGCache](https://arxiv.org/abs/2404.12457)

## 外部参照与公开评价

`LlamaIndex_native` 调用上游 Memory、历史问题凝练与 compact/refine，检索使用 BM25＋dense 和发布版 reranker，生成交给统一账本下的固定 Qwen。外部比较保留各自实际历史，框架调用与检索成本进入完整账本。[上游问题凝练](https://github.com/run-llama/llama_index/blob/v0.14.25/llama-index-core/llama_index/core/chat_engine/condense_question.py) · [上游 compact/refine](https://github.com/run-llama/llama_index/blob/v0.14.25/llama-index-core/llama_index/core/response_synthesizers/compact_and_refine.py)

公开任务及作者评价与构造机制轨分别报告。评分基于实际答案、历史、可见原文和结论来源闭包，保留声明分母与失败记录。[MT-RAG](https://github.com/IBM/mt-rag-benchmark) · [MultiDoc2Dial](https://github.com/IBM/multidoc2dial) · [Loong](https://github.com/MozerWang/Loong) · [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) · [VersionQA](https://github.com/danielhuwiler/versionrag) · [数据与评价](data_and_evaluation.md)

## 可选对照入口

- `selection.mode: support`、`whole_plan`、`E0_native` 用于独立选材、整计划与历史参照；主复用消融为 `S3_contextpilot_no_fact_reuse`。[运行手册](study/final_runbook.md)
- `B6_value` 与 `S3_value` 用于比较额外保留。Host 准入是另一个默认关闭的原生对照，启用后按真实业务会话身份记录。[矩阵](../configs/study_matrix.yaml) · [EfficientAgent 适配](third_party/EFFICIENTAGENT_NOTICE.md)
- 表格注释可按配置使用 CRAFT 行文本函数，复用范围及表格关系由现有适配说明记录。[CRAFT 说明](third_party/CRAFT_NOTICE.md) · [表格关系](research/table_relations.md)
