# 固定版本SDK证据包

`adapters.api_evidence` 是独立离线构建器，将本地Python SDK声明转为普通 `CorpusIndex` records。它不导入SDK，不执行任何源码或答案，不联网，不修改MTRAG索引或默认retriever。研究动机、三个原始案例和新增材料的公平比较边界见[API证据方案](api_evidence_plan.md)。

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

包名、包版本、Git commit、tag和仓库路径都必须显式提供；不从安装说明的最低版本推定唯一目标版本。完整commit只约束声明的版本形式，**不会证明本地文件来自该commit**。构建器离线读取一次原文件SHA，manifest明确记录来源声明与本地字节；可提供 `expected_source_sha256` 与先前独立冻结的源文件核对。即使该检查通过，也不冒充远程Git验证。没有运行时重复hash、额外图存储或新增在线版本检查。

输出包括 `records.jsonl`、`manifest.json` 和逐源原样复制的 `licenses/*-LICENSE.txt`。manifest保存原文件头、许可声明、输入规格、原文位置、未知项与来源身份状态。`license_id`来自规格声明，构建器不自动判定许可法律适用性。已有输出目录不覆盖。

## 提取与未知范围

- 函数/异步方法保留包括装饰器、签名、docstring与方法体的连续原文区间；类记录保留装饰器、类声明及完整类docstring，缺docstring时只保留声明。不存在的文档不补写，类片段不承诺是独立可执行程序。
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

它不会自动接入检索：扩展索引、目标包/版本选择及生成对照须另行明确配置。未来缓存策略比较必须共享同一冻结扩展包；不能把新API材料带来的变化混入同材料B0–B6缓存比较。

## 已完成验证

`tests/test_api_evidence.py`覆盖无执行、装饰器/async、Unicode/CRLF原文与偏移、复杂签名/缺文档、显式和注解依赖、Literal/Annotated非类型内容、动态HTTP/嵌套作用域、继承/重载未知、版本/仓库歧义、未到达的依赖声明、离线源身份核对、许可复制与CLI退出码。与既有CorpusIndex测试合计26项通过。

真实离线构建产物为 `runs/api-evidence-review/production-bundle-v2`；`production-contract-check-v2.json`证明两份固定官方源的10条正文、span、行号与此前保存的10个preview完全一致，并列出各闭包。`PYTHONPATH=. .venv/bin/python runs/api-evidence-review/validate_production.py --bundle runs/api-evidence-review/production-bundle-v2 --output runs/api-evidence-review/rechecked.json` 可重新核验。旧回答、旧audit、旧preview与构建v1均保留。此验证只证明工程提取与来源绑定，没有新增生成、运行时API正确性或总体答案质量结论。
