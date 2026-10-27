# HTTP问题检索与多轮提交

`POST /sessions/{session_id}/query`把当前问题交给已配置的检索器，再沿原Controller的会话锁、attempt、生成、持久提交与恢复流程执行。客户端只提交`query`和`request_id`；实际历史来自服务自己的已提交QA。原`/turns`供证据接口继续保留，不自动改变为检索模式。

本入口复用BM25S/BGE检索factory、CorpusIndex和单次原生生成，没有上传/索引管理服务，不加载数据集参考答案或未来问题。当前CPU合同使用真实BM25与模拟引擎验证工程流程，不证明模型回答质量、GPU速度或成本策略优势。

## 零GPU可运行示例

先安装本项目锁定的检索依赖，再将仓库内的合成材料建为固定索引。索引目录已存在时build会拒绝覆盖；重复演示可复用已建索引，并使用新的会话ID或新的state目录。

```bash
uv sync --extra retrieval --locked
uv run python -m adapters.retrieval build \
  --corpus bench/examples/http-corpus.jsonl \
  --index runs/http-demo-index \
  --snapshot synthetic-http-retrieval-v1

uv run contextkv --config configs/demo.yaml --state-dir runs/http-demo-state serve \
  --port 18080 \
  --retriever-index runs/http-demo-index \
  --evidence-corpus bench/examples/http-evidence.jsonl \
  --corpus-id synthetic-http
```

另一个终端按顺序发送：

```bash
curl -sS http://127.0.0.1:18080/health
curl -sS -X POST http://127.0.0.1:18080/sessions/demo/query \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"r1","query":"What is the ORCHID request timeout?"}'
curl -sS -X POST http://127.0.0.1:18080/sessions/demo/query \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"r2","query":"What about retry exclusions?"}'
curl -sS http://127.0.0.1:18080/sessions/demo
```

两条请求也保存在`bench/examples/http-requests.json`。材料含明确的条件依赖，CorpusIndex会补入原文；模拟引擎只返回其协议测试文本，**不能把该文本当真实问答效果**。重复发送r1可验证幂等重放；最后用以下接口关闭会话：

```bash
curl -sS -X POST http://127.0.0.1:18080/sessions/demo/close
```

### 使用已有Qwen3与真实索引

先按[部署说明](../deployment.md)启动与显式scope一致的研究SGLang服务，再使用同一入口：

```bash
uv run --extra tokenizer --extra retrieval contextkv \
  --config configs/qwen3_8b_scope.example.yaml \
  --state-dir runs/http-qwen3-state serve \
  --retriever-index data/cloud-bm25s --corpus-id mtrag
```

`--retriever-config configs/retriever_hybrid.example.yaml`可替换`--retriever-index`使用已有BGE dense/RRF/rerank配置；两者互斥。BGE权重和其PyTorch运行环境必须预先可用，不在请求中下载。模型scope、系统提示、采样与输出预算均来自共享RunConfig；HTTP入口不会套用旧模型成本artifact或把工作配置当质量赢家。

`--evidence-corpus`读取原文、来源、版本、条件/表头依赖的JSONL；`--corpus-id`是证据命名空间，默认`mtrag`，自有文档应明确填写。公开来源字段按`source_uri → source → url`读取，标题也支持metadata；不会复制答案标注。MTRAG已有索引可配合`--evidence-corpus data/mtrag-official/cloud.jsonl`恢复原始文档地址，正文和检索排名不变，无需重建向量。服务的corpus snapshot只取已加载retriever的manifest；请求不能任意指定snapshot。元数据必须能精确匹配原检索正文，条件闭包不会改写正文，也不给追加依赖编造检索排名。

索引构建同时接受BEIR `_id`和元数据语料的`passage_id`；同一`document_id`下的不同片段保持独立，`source_uri`可来自顶层或metadata。缺少来源地址时，证据URI使用显式`corpus_id`命名空间。旧版本可能把自有语料的缺省来源错误写成`mtrag://…`；既有archive不会自动改写，同key的来源变更仍会被拒绝。此类旧状态应显式使用新会话/新state并固定正确source，不能静默迁移后宣称来源一直相同；默认MTRAG来源格式保持不变。

## 请求、结果与来源

新query请求只接受两个非空字段，拒绝`history`、`evidence`、`targets`、`reference`、`corpus_snapshot_id`等额外输入。没有配置检索器或应用未就绪时返回503与`RETRIEVAL_NOT_READY`，不会静默以空材料生成。

响应沿用`TurnResponse`，包括原生`assistant_text/output_ids`、request/attempt/ledger版本、动作、真实receipt和`truncated`，另增加：

- `retrieval_ms`：原attempt这次检索及元数据适配的实际耗时。
- `visible_evidence_manifest`：原attempt所选candidate的EvidenceSpan，含稳定ref、EvidenceKey、source_uri、provenance和该次输入消息内的正文位置。
- `citation_scope`：本轮必需证据对应的允许引用ID；APPEND与REBUILD共用同一范围。APPEND中仍可见的旧材料不自动获得本轮引用资格，明确回访时正文与资格一起恢复。

visible说明该次实际输入包含哪些正文，**不是答案实际引用清单，也不证明支持答案**。完整原文可从`GET /sessions/{id}`的archive按EvidenceKey回读；消息内位置属于这次attempt的输入，不能套到后续execution上。retrieval rank/score、返回材料与明确依赖记录在现有events；初始化时的`retrieval_configured`记录检索配置/来源manifest，不每请求重复hash索引或权重。

r1完成、又完成r2以后重放r1，仍返回r1原答案/IDs/来源，不能取当前snapshot的最新来源。重放的`retrieval_ms`是原attempt的历史耗时，`complete_ms/timing`不伪造本次重放测量。它不会再次检索或生成。

## 幂等、失败与恢复

同一session的幂等检查、未决结果恢复、读取已提交历史、检索与提交处于同一Controller串行边界。检索采用现有worker线程和检索器资源约束，避免阻塞HTTP事件循环；不同业务会话仍可独立进行，不能把全局模型串行限制称为业务互斥。

| 情形 | HTTP / 状态语义 |
|---|---|
| 同ID同query已经COMMITTED | 返回原结果，`idempotent_replay=true`；检索/生成次数不增加 |
| 同ID不同query或供证据/检索模式冲突 | 409，原答案与历史不变 |
| 检索后端/元数据匹配失败 | 503 `RETRIEVAL_FAILED`，attempt为FAILED；不新增QA、不prepare/generate。显式新ID才重试 |
| 输入字段无效或全文超预算 | 422；正文/QA不被静默截掉 |
| 引擎结果未知/仍在途 | 保留attempt；按现有409 pending或503引擎错误返回，不重发生成 |
| 完整结果可恢复 | 同ID请求或`/recover`取得同一结果并只提交一次，不重新检索 |
| 原生length终止 | 保留实际文本/IDs并标`truncated=true`，不冒充完整正常回答 |
| 已close会话继续新请求 | 沿原StoreError返回409 |
| 后台退役程序或协议错误 | 写入`epoch_retire_worker_failed`事件，已提交及后续回答保持正常；drain/关闭报告维护错误 |

未完成prepare的显式abandon仍是管理恢复动作，必须先确认原执行者已停止；结果未知不能因为客户端超时就放弃并重采样。超出引擎结果保留窗口或引擎重启后的恢复沿已有合同，不承诺跨重启exactly-once生成。

```bash
curl -sS -X POST http://127.0.0.1:18080/sessions/demo/recover \
  -H 'Content-Type: application/json' -d '{"abandon":false}'
```

新请求的完整服务时序包含会话排队和检索，控制/规划成本中不再重复计这段检索时间。外层HTTP响应当前是缓冲返回；引擎客户端内部SSE不等于HTTP调用者收到逐token stream，不虚构外层客户端TTFT。

## 初始化与关闭

应用启动时加载一次检索器和元数据，保存`service_started`、`retrieval_configured`及实际配置。`/health`提供engine信息、pending retire计数和检索manifest。

请求取消会先等待其检索线程结束，再释放会话锁；关闭期间新请求返回503 `SERVICE_SHUTTING_DOWN`。ASGI计数/Event等待已接受请求收尾，随后依次关闭检索器、引擎客户端及SQLite。原生未决生成通过持久attempt恢复，退役维护使用同一drain路径。状态协议见[架构](../architecture.md#故障恢复)。

HTTP与SDK接线的已有验证集中在[工程接线验证](../data_and_evaluation.md#工程接线验证)，接口回归位于`tests/test_http_retrieval.py`、`tests/test_api.py`及`tests/test_api_shutdown.py`。
