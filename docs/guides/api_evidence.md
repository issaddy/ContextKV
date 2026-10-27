# 固定版本SDK证据包

`adapters.api_evidence` 是独立离线构建器，将本地Python SDK声明转为普通 `CorpusIndex` records。它不导入SDK，不执行任何源码或答案，不联网，不修改MTRAG索引或默认retriever。构建的相同原文与依赖应同时供给所有被比较方法。

## 输入与运行

入口为 `extract_sdk_records(source_path, source_spec) -> (records, manifest)`；批量CLI读取JSON规格，先完成提取和现有CorpusIndex依赖闭包验证，再写一个新的目录：

```sh
.venv/bin/python -m adapters.api_evidence \
  --spec runs/api-evidence-review/source_bundle.json \
  --output runs/api-evidence-local
```

现有 `source_bundle.json` 是已冻结的本地开发材料，含Cloudant0.0.27、Watson7.0.0两份官方源码及10个公开符号；不是要求从网络下载新版本。规格结构如下，路径相对规格文件所在目录：

```json
{
  "corpus_id": "official-sdk-example",
  "snapshot_id": "cloudant-0.0.27",
  "sources": [{
    "repository": "IBM/cloudant-python-sdk",
    "commit": "c98dd5b640538978790c777049f3c2e9df8e816a",
    "tag": "v0.0.27",
    "package": "ibmcloudant",
    "package_version": "0.0.27",
    "path": "ibmcloudant/cloudant_v1.py",
    "source_file": "official/IBM-cloudant-python-sdk/ibmcloudant/cloudant_v1.py",
    "license_file": "official/IBM-cloudant-python-sdk/LICENSE",
    "license_id": "Apache-2.0",
    "symbols": ["CloudantV1.post_document"],
    "dependencies": {"Document": ["Attachment", "Document.to_dict"]}
  }]
}
```

包名、包版本、Git commit、tag和仓库路径都必须显式提供；不从安装说明的最低版本推定唯一目标版本。commit记录声明的源码版本。构建器离线读取一次原文件SHA，manifest明确记录来源声明与本地字节；可提供 `expected_source_sha256` 与先前独立冻结的源文件核对。该检查核对本地字节，运行时直接复用构建结果。

输出包括 `records.jsonl`、`manifest.json` 和逐源原样复制的 `licenses/*-LICENSE.txt`。manifest保存原文件头、许可声明、输入规格、原文位置、未知项与来源身份状态。`license_id`来自规格声明，构建器不自动判定许可法律适用性。已有输出目录不覆盖。

## 提取与未知范围

- 函数/异步方法保留包括装饰器、签名、docstring与方法体的连续原文区间；类记录保留装饰器、类声明及完整类docstring，缺docstring时只保留声明。不存在的文档不补写，类片段不承诺是独立可执行程序。
- 可显式选择公开类的`Client.__init__`，或将它作为操作符号的同文件依赖；初始化函数保留原始完整源码、声明参数与`initializer_of`。该接口补齐操作参数之外的初始化依赖。仅选择`Client`不会自动展开初始化代码，也不开放其他私有/dunder符号；继承而未在该类定义的初始化仍为未知。它不自动补齐外部认证库或证明整个示例可执行。
- `span` 是按源文件声明encoding解码后的Unicode字符偏移，半开区间；行号从1开始。CRLF、Unicode及原缩进保留。原文件字节SHA与文本字符偏移用途不同。
- `api_source.signature`、`docstring`、HTTP原表达式均附原文span；签名按token定位，不重构格式。`required_declared_parameters`按原AST列出必需的声明参数，也包含声明中的 `self`；不是运行时绑定后的调用签名。
- 只识别函数体中的语法形式 `*.prepare_request(method=..., url=...)`。直接字符串或同一函数体唯一、前序、无条件字符串赋值可以列出原字面值；动态format、属性、其他请求库、条件改写等保留unknown。它描述源码中的请求配置，不保证实际transport行为；装饰器、默认值、参数/返回注解和嵌套函数/class/lambda中的调用不算该方法体的请求。
- 请求的继承方法、导入属性或多次同名声明不会选一个猜测。它们记入 `unknown_symbols`。存在此类未解析请求时仍保存其余有效片段与manifest，CLI返回1；完整构建返回0。明确依赖指向缺失/歧义目标时，已有CorpusIndex验证拒绝发布该包。

## 依赖与现有证据入口

依赖仅来自规格中的显式同文件公开符号关系，或原方法注解指向同文件实际声明。前向字符串类型可以解析，`Literal`值和`Annotated`说明文字不会变成类型依赖；外部类型/导入别名不自动展开。类docstring里的关系不靠自然语言猜测，可通过公开source-spec附原位置解释后显式配置。未被请求或闭包到达的owner不会应用其依赖；manifest保留全部 `declared_dependencies`，并将这些owner列为 `unapplied_dependency_owners`，不声称已验证其目标。

例如真实 `CloudantV1.post_document` 的原注解包含 `Document`，自动加入该类；示例规格再按源码中 `_attachments` 序列化关系明确加入 `Attachment`、`Document.to_dict`。现有 `CorpusIndex.resolve` 得到四块原文。Watson的 `list_language_models` docstring明确返回 `LanguageModels`，该类型的docstring再指向 `LanguageModel`；开发规格将两步作为显式依赖保存，不宣称由类型注解自动推得。

每个record用普通 `passage_id`、固定 `version`、`text`、`span`、`source_uri`、`anchors`、`dependencies`、`provenance`。同包同路径同符号在不同版本中仍有不同EvidenceKey；裸passage ID跨版本歧义由既有CorpusIndex拒绝，调用方须选择确切key。不同仓库却共享同包/路径/符号和版本时，现有依赖查找会报告歧义，不悄悄绑定另一个仓库；调用方需分开构建有明确来源的包。新增 `api_source` 诊断信息留在records/manifest，CorpusIndex仍使用现有EvidenceBlock，不扩展核心schema。

加载产物：

```python
from adapters.corpus import CorpusIndex

index = CorpusIndex.from_jsonl(
    "runs/api-evidence-review/production-bundle-v2/records.jsonl",
    corpus_id="official-sdk-preview",
    snapshot_id="cloudant-0.0.27-watson-7.0.0",
)
blocks = index.resolve(("ibmcloudant:ibmcloudant/cloudant_v1.py#CloudantV1.post_document",))
```

证据包可由同一BM25构建入口索引，检索key取`passage_id`，不会把同一源码文件中的多个符号误当重复document。固定`source_uri`保留为原GitHub出处；HTTP通过同一records作为metadata恢复真实document/version和依赖：

```bash
uv run python -m adapters.retrieval build \
  --corpus runs/api-evidence-review/production-bundle-v2/records.jsonl \
  --index runs/sdk-bm25 \
  --snapshot cloudant-0.0.27-watson-7.0.0
uv run contextkv --config configs/demo.yaml --state-dir runs/sdk-demo-state serve \
  --retriever-index runs/sdk-bm25 \
  --evidence-corpus runs/api-evidence-review/production-bundle-v2/records.jsonl \
  --corpus-id official-sdk-preview
```

以上路径指已冻结的本地研究产物，不随wheel发布；使用自己的包时替换路径和身份。示例生成器是CPU mock，实际Qwen3使用对应scope配置及已启动引擎。扩展索引和目标包/版本仍由调用方明确选择；新增SDK材料的效果按系统对照报告。

## 将关系传入回答

共享候选构造器将当前required材料的依赖和parent_section渲染为稳定E_ID关联注记，A/R与原生基线共用。APPEND追加当前关系并保留旧正文及span；无关系元数据时沿用普通输入。关系表达语料声明的连接，实际注记token计入预算。

构建与桥接结果见[工程接线验证](../data_and_evaluation.md#工程接线验证)，原件位于`runs/api-evidence-review/`及`runs/evidence-context-integration/sdk-v1/`。提取与来源合同由`tests/test_api_evidence.py`覆盖。
