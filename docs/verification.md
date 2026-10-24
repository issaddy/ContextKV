# 工程验证记录

日期：2026-10-24（Asia/Shanghai）。本地为 macOS arm64 / Python 3.12.13；真实引擎部署于用户提供的 Linux 容器 /home/relay/songjiayun/ContextKV。执行成功与研究收益分开记录。

## 本地工程证据

| 验证 | 实际结果 |
|---|---|
| 最近完整 CPU 回归 | `af9e139`：1328 passed，6 subtests passed，已显式开启真实 tokenizer 项，无跳过；runs/pytest-owned-local-scope-complete.log |
| 固定 Qwen tokenizer 专项 | 原生完整编码、批量编码、边界回退与冻结候选 IDs 等价通过，包含在最新完整回归中；Linux 30 个真实候选也逐 token 相同 |
| Ruff / wheel | Ruff、diff检查通过；固定`af9e139`源码构建wheel/sdist，逐字核对85模块、29包内资产、345个sdist tracked文件、106测试/配置及13许可，未包含运行数据、模型或环境；runs/dist-owned-local-scope/asset-check.json |
| 开发集模拟矩阵 | 33 jobs / 99 轮成功 |
| 冻结模拟测试矩阵 | 39 jobs / 117 轮成功 |
| 继承主方法参数的开发集消融 | 15 jobs / 45 轮成功，不参与主方法选择 |
| 官方 MTRAG 格式检查器 | 6 轮 fixture，errors / warnings 为空；只证明格式兼容 |
| 固定 MTRAG 全量复算 | 842 任务 / 110 会话；Cloud 205 / 26；1053 处参考标记 |

`af9e139`继续将显式本地scope与共同采样接到owned `--matrix`，以声明的显存比例/hostGB替代新模型误用旧7B固定值，按实际模型、池、cost scope校验后生成native清单；质量模板使用真实local-snapshot身份。旧13格自动构造器、greedy默认、质量gate与调度保持。18项新合同包含在全回归，独立38项审查通过；该owned切片尚未部署或真机验收，不能借上轮静态B0 probe追认。

另完成[指定来源支持地址](source_local_support_design.md)离线原型：直接复用exact matcher/SpanRef，6项通用反例和43条已有来源坐标重放通过。固定作者代码6组见证进一步明确AFTG宽松定位与ReClaim隔离/训练边界。没有新增GPU、模型或在线生成协议，不构成语义质量或prefill节省。26文件研究快照独立归档于`runs/source-local-support-artifact-v1.tar.xz`，与核心wheel/sdist分开。

`cae9149` 的全局审查切片完成[正式原生模型/采样入口](native_model_scope.md)、B6赢家别名去重复，以及已有Plan收益/margin/稳定期的事件投影。25项新模型合同、5项别名合同包含在完整回归；独立复核还重放四份旧真实B0/B1 manifest，全部兼容。旧冻结模拟矩阵的39 jobs是其实际历史执行数，不因新别名逻辑重写成更少job。

该CPU切片验收时尚未部署或运行新GPU实验。随后`04b92f8`已在远端核验343个tracked文件，完成[固定12题D/N/P真实诊断](source_plan_diagnostic_validation.md)：3次stock接口检查通过，45次诊断HTTP均有终态，保留36个回答/60个stage位置。D12个final；N12次准备均384/length、未执行final；P9个final、3次准备length。独立实际输入/采样/原生输出/停止原因核验无问题；N未完成并非全局EOS失效。当前v2不扩159，不提高cap补跑。原研究服务已恢复、health200且池为空；研究Controller新采样、本地新B1完整路径未由这次stock检查验证。

矩阵证据在 runs/matched-dev、runs/matched-test 和 runs/matched-ablation-dev。此前 48-job 开发矩阵属于未匹配消融参数的旧版流程，不再作为最终方法对照。runs/official-checker/verification.json 明确使用 fake fixture 答案，不是质量结果。上游 Starlette TestClient 有 HTTPX 弃用提示，不影响通过。

`3b852aa` 同轮 Ruff 全仓通过；`runs/build-native-budget-judges.log` 的 wheel/sdist 构建通过，wheel 包含两种新增 judge、Lettuce 原生模板/协议/MIT 许可及 Granite Apache 归属，不包含运行数据或模型权重。真实 Granite 首轮另发现 Transformers 5 默认返回 BatchEncoding；`856a753` 显式请求 token ID 列表并增加相应合同测试，首轮失败记录与修复后结果分别保存。

复核使用 .venv/bin/python -m pytest -q 和 .venv/bin/ruff check .；真实 tokenizer 专项设置 CONTEXTKV_TEST_HF_TOKENIZER=1。新增模块的定向测试记录应在下一次完整回归后统一更新总数。

`58a0a08` 的完整回归包含业务会话host准入、准备恢复ACK、来源主题解析的两条CLI、离线来源范围及ConvSearch-R1原生格式。Ruff通过；`runs/dist-host-source-topic-http` 的wheel/sdist已核对六个相关模块、ConvSearch prompt/protocol/完整许可/NOTICE与EfficientAgent归属，未打入运行数据或权重。新增HTTP与本地tokenizer共享thinking参数，避免换模型时两种模板设置漂移。

后续真实验证记录分别为[同源164轮标定比较](native_budget_targeted_validation.md)和[host准入机制](host_admission_validation.md)。前者降低回退但未证明稳定加速；后者的过滤/补齐4次、压力与身份控制15次、中等容量双方向重访16次均已独立复核，保留业务间前缀保留收益及自身重算代价。历史主题v1/v2各82条记录、72次HTTP与语义审查已完成，enum减少结构失败但没有纠正话题选择，见[选择器验证](source_topic_selector_validation.md)。

`494e33a`完整回归又覆盖[SDK原文与显式初始化合同](api_evidence.md)、[正文冗余基线](redundancy_baseline.md)、查询解析失败留痕、召回/重排查询分离及矩阵启动源码绑定。Ruff与diff检查通过。初次`python -m build`因本地缺该构建入口失败；改用项目已有uv成功构建，但逐文件核对发现sdist遗漏`tests/fixtures/*.txt`，已补MANIFEST收录并重新打包，旧日志和旧包保留。最终包验证结果另存`runs/dist-rerank-redundancy-complete/asset-check.json`，核对包内Python模块、运行资产/许可和sdist测试/配置，不包含运行数据、权重或环境文件。新基线CLI四轮模拟演示依次REBUILD/APPEND/REBUILD/REBUILD，事件含正文分子/分母；不作硬件性能证据。

独立[ConvSearch-R1 41次改写](convsearch_validation.md)、[八查询方法328条检索](frozen_query_retrieval_validation.md)及[SDK证据12次生成](api_evidence_validation.md)均已真实执行，原始输入、失败/截断分母和具体退化保留。这些硬件记录的项目源码仍为各自manifest所记的`20fef31`等版本，不能追认为包含后续新代码。后续[四臂冻结重排](frozen_rerank_validation.md)已完成164条记录：分离查询挽回了长改写重排的退化，但尚未稳定胜过简单用户主题视图。[完整独立开发评分](dev_quality_validation.md)已完成615次尝试，609维有效、6维null；完整配对维度仍不确定，两个候选均未通过开发质量门。下一全cohort可选原生格式约束已完成CPU合同，但不修正评审器语义错误。最终B6与冻结test仍未完成。

`e544ea7` 新增[13格B6联合执行器](b6_joint_sweep.md)与真实答案绑定的开发质量冻结；审查发现并修复了通用入口忽略服务器profile的问题。全仓首次回归发现8个表格测试的native模拟payload缺少已有的engine_manifest_sha256字段，`d4473ae`补齐测试合同，未放松真实清单要求；复跑1091项和6个子测试全部通过。Ruff、shell语法和diff检查通过。`179f844`的可选原生Selene格式约束默认兼容旧协议，已完成独立118项相关测试和真实tokenizer/XGrammar CPU检查，尚不代表约束后模型评分已完成。最终包记录在`runs/dist-b6-native-regex-complete/asset-check.json`；后续真实13格使用下述独立0893134来源，不能把fake-owned测试当作硬件结果。

`cac25e6` 增加显式系统repetition与逐job/block质量绑定；`5d07915` 增加[通用owned矩阵入口](b6_owned_matrix.md)，在启动前验证冻结选择和实际成本适用范围，支持dev/test/dev_ablation及父profile继承。全仓1156项和6个子测试通过，Ruff/diff检查通过；运行文档明确完整cohort、显式GPU上限和固定host支持范围。新增入口尚未在真实heldout使用，不能给旧运行补写新的源码身份。

真实[13格phase1](b6_joint_sweep.md)在项目`0893134`的310文件核验部署下完成 **533/533轮、0生成失败、13个length答案保留**。远端独立CPU审计13格/533轮issues=0；prepare再次核对533份完整自身QA与2725个可见正文span。实际为GPU16384、非B0 host279040、P2/c4/Q0/cap1024、host admission off、单次有序运行。前两格完成后曾在远端压缩/传输公共快照，可能与后续格host CPU/IO重叠；本次只能作开发筛查，正式计时重复应延后归档。原服务已由supervisor恢复，随后另行进入离线评价时段。

本轮完整2665维Selene输入预检通过，最大input14381，加1536输出预算不超过32768，未裁证据；tokenizer元数据4096的提示保留，实际模型config位置上限131072及Llama3 RoPE已另存核对。新原生regex协议已完成[533答案/2665维评价与独立审计](b6_quality_validation.md)，全部有效、issues=0；60维比较7项noninferior、53项inconclusive，12候选均未获得联合资格。原089入口保存readiness而未产出B6，未改门槛。原始生成位于远端`runs/b6-joint-dev-phase1-v1`，CPU审计位于`runs/b6-joint-review/h200-v1`，评价准备位于`runs/b6-quality-prep/frozen-v1`；这些记录的生成/评分环境保持0893134；之后的新版本部署不追改本批来源。

`c005319`新增默认关闭的原生准入门控诊断；独立整合审查发现新info字段尚未进入typed客户端，`4e689eb`补齐默认False兼容字段和真实MockTransport合同。修复后完整1179项及6子测试、全仓Ruff通过；固定SGLang原文件校验、补丁正反应用与14文件语法检查通过。新补丁为`730ead04291bb23e5ab09ee6acb15df9225c0d1942a14dd3516f20c5a103c474`，不改变原生预算、调度或锁。已在远端核验部署321个tracked文件并完成真实完整checkout的源检查；受控并发硬件执行另存记录，源检查本身不是GPU验证。

本轮包检查进一步发现sdist漏收三个`bench/examples`文件，`6d63af9`补齐package data；旧构建保留。最终`runs/dist-admission-client-complete/asset-check.json`按固定4e689eb Git树逐字核对83模块、29个wheel包内非Python资产、320个sdist tracked文件（含97测试/配置）及13个许可文件，均通过，不含运行数据、权重或环境文件。29是包内资产的完整枚举，不能与旧报告的19个限定运行资产直接比较新增数量。

[369轮计时分解](b6_timing_diagnosis.md)和[联合问题模型](joint_optimization_model.md)进一步明确了下一项机制的证据要求。旧等待估计器不把候选服务插入peer事件，不能据此生成externality分数。[两peer受控fixture](peer_interference_validation.md)现已完成54个正式episode、18个暖机，216个target加48个历史seed，共264/264次模型调用；39文件公共归档经独立审计issues=0，来源为部署4e689eb，研究服务恢复且健康检查通过。27个正式配对的peer项非零，但自身与总flow-time排序全部同向；大APPEND的准入门槛不能被GPU命中消除，不据此新增planner收益结论。

fixture本地19项合同包含6项真实typed客户端到Runtime.quote的快照测试；远端默认plan、只读原生quote预检与全部部署来源核验通过。容器没有pytest，远端CPU测试命令未成功，未为重复CPU测试安装额外包。v1/v2零生成工程失败、v3的72次MIXED_SNAPSHOT失败及48次已完成seed分别保存，不拼入v4成功分母。v4修复仅将A/R与两个peer分为2/1/1报价，不改原生快照协议。固定输出512的实验不作为自然EOS问答质量证据。

`16e4ef9`在既有只读prefix遍历中增加匹配GPU前缀与FULL硬锁状态的交集，只修正可选native_budget、无waiting/host/pending情况下的即时fit。完整1189项及6子测试、Ruff、固定补丁正反应用与14文件语法检查通过；`runs/dist-shared-pin-complete/asset-check.json`逐字核对83个Python模块、29个包内资产、322个sdist tracked文件、97个测试/配置和13份许可。已在远端部署并验证323个Git blobs，新patch为`7a67fc7fea1eec99c3c578f061bc59dd442d0304101d2a90bda2730218b1333f`。9项独立fixture/恢复合同通过，默认plan不发生成请求。

[共享前缀H200验证](shared_prefix_admission_validation.md)随后完成8个episode/32次调用；30文件完整公共导出保留全部预热与正式结果。正式3个共享状态纠正了旧等待分类，3个无关状态仍需等待；实际forward顺序支持这一作用路径。未运行新旧planner的自然问答策略对照，不宣布系统加速。supervisor恢复命令返回0后，首个health探针仍处加载期；随后同PID70413返回HTTP200，没有重启或重跑实验。

`870a250`为原生客户端增加可选temperature/top_p/top_k/min_p设置，仅在构造边界复制/校验一次，默认温度0的请求保持原样，输出预算仍由每次generate唯一指定；实际stream/nonstream请求审计保留采样参数。完整1207项及6子测试、独立相关审查与Ruff通过。固定源码构建的wheel/sdist核对83模块、29资产、324 tracked文件、97测试/配置、13许可，见`runs/dist-qwen3-sampling-complete/asset-check.json`；远端共享前缀实验仍绑定16e，未追改其身份。

用户现有`/media/ssd1/qwen3-8b`与`/media/ssd1/qwen3-32b`已只读检查，5/17分片齐全、两个真实tokenizer可加载、非思考模板渲染一致，记录于`runs/qwen3-local-model-inspection.json`。这不是权重逐字校验，未确认上游revision；未启动新权重下载。已在325文件核验的`a6b5711`部署上完成[全159轮8B P2及32B同输入重放](qwen3_reference_validation.md)，使用独立stock引擎和本地模型身份。318轮均成功；独立重渲染、重分词、输出解码、完整QA和源码绑定全部通过，length17/13保留。研究服务已恢复且health/协议核验通过；回答质量另行分析，不继承旧7B标定或B0/B6身份。

模型迁移审查发现主候选只验证证据消息，未检查原生模板是否改写历史助手正文。现有渲染边界已扩展为按序检查全部消息正文，保持原生模板和既有函数入口，不新增render或hash。CPU反例覆盖reasoning-prefix变换、重复历史、APPEND/REBUILD和失败时快照不变；不是对所有模板语义或角色边界的形式证明。真实固定7B tokenizer启用的完整回归为 **1210项及6子测试通过**，Ruff通过，记录在`runs/pytest-full-qa-render.log`。当前Qwen3运行器已有同类完整历史检查，本地修复没有改动正在运行的远端源码。

显式[Qwen3本地模型配置](model_scope.md)已贯穿controller、launcher、native guard和profiling/lifecycle探针；local identity与HF加载revision分离，初始化检查实际目录/metadata/tokenizer/模板，旧无scope默认保持7B。新scope不能采用未绑定或旧模型的成本artifact。`a87638a`最终 **1236项及6子测试通过**，另25项原测试与4项独立scope合同通过，Ruff/shell/diff检查通过。新patch为`b1b58e096c309eef1d7c10b1da3d29bec86503030e72f0f8e3badba9cf9c8a05`，固定原源码校验、应用/逆向检查及15文件编译通过。其固定Git树构建核对84模块、29包内资产、332个sdist tracked文件、102测试/配置及13许可，见`runs/dist-qwen3-scope-complete/asset-check.json`。随后333个tracked源码已部署，两模型完成[有限H200协议/完整QA/GPU与Host生命周期验收](qwen3_scope_gpu_validation.md)，并保留并发观测的partial覆盖。此前能力参考始终绑定`a6b5711`stock引擎，两个实验的源码、配置与结论独立记录。

`b3d3c2f`将[显式采样](sampling.md)贯通研究client/controller/profiler与离线重评分，默认旧wire保持。旧执行时间点保留；跨配方不冒充已校准，经验长度和learned模型分别绑定来源与实际残差基准。初轮完整回归1273通过、1项旧list形态合同失败，日志v1保留；恢复原只读容器形态并修正训练/服务先验中心后，最终1277项与6子测试通过。新增41项、另4项独立合同通过，独审实际运行了四会话的length-fit CLI。CLI demo同时修正真实引擎被恒标simulation的问题，mock四轮与实际8B记录的CPU回放分支检查通过；后者没有新模型调用。最终固定Git源码包的核验见上表。本批采样实现尚未部署；容器仍是已完成GPU验收的`a87638a`及原生兼容补丁，不将先前硬件探针追记为新采样验证。

## 已完成的 H200 集成验证

单张 H200、TP=1、Python TreeCore、BF16 权重/KV、page size 64、context 32768、最多 8 个运行请求。实际环境：Python 3.12.3、torch 2.14.1+cu130、CUDA 编译目标 13.0、transformers 5.19.0、tokenizers 0.23.2、驱动 595.58.03。GPU KV pool 为 121920 tokens；host 为 279040 tokens，即 16001269760 字节（约 16 GB / 14.90 GiB）。固定源码的 hicache-size 采用十进制 GB，并按 page 对齐。完整环境保存在 engine-environment.txt。

SGLang 固定 3f36fc08331f64cd63c4fbbdf39cc402f38bb38b，Qwen2.5-7B-Instruct 固定 a09a35458c702b33eeacc393d103063234e8bc28。以下初次 GPU 证据使用补丁 49355b997267b7c88ca7182d53ab4846109d13bf7064d1e179b729fc444362a5；后续补丁须分别验证，不能自动继承结论。

| 验证 | 实际结果 |
|---|---|
| GPU-only 模型协议冒烟 | 完整生成/结果恢复、重复报价、退役后拒绝请求、ACK 不虚报物理释放，均通过 |
| kernel + layer-first host 冒烟 | 通过相同验收 |
| 冷/GPU-ready/host-only 小标定 | 128 输入、8 输出、prefix 0/64、并发 1、每单元 3 次；18 条请求阶段与 12 条完成传输观测 |
| 小样本 P2，GPU-only | 6 成功 / 0 失败 / 0 未完成 |
| 小样本 P1、P3，host 启用 | 各 6 成功 / 0 失败 / 0 未完成；P3 使用真实 BM25S |
| Cloud 开发集 P2，host、并发 4 | 41 成功 / 0 失败 / 0 未完成；512 输出上限下 21 轮触顶截断 |
| 完整 Cloud 语料 P3 开发集，1024 上限 | 固定 72442 个原始 passage，BM25S 冻结索引；41 成功 / 0 失败 / 0 未完成 |
| 未补丁原生 B0 / B1 | 分别独立固定源码启动；各自 fixture 6 轮、Cloud P2 开发集 41 轮均成功 |

本地镜像证据在 runs/remote-h200-initial，包含原始报告、manifest、观测和失败/成功日志。容器导出使用明确文件白名单并核对归档 SHA256，没有导出代理配置或权重。

后续补丁 69bf539f582ad7065d1b16c6fa6efe348c3e69d9d4aec8a296f3206996b76912 已在相同 H200 配置通过 runtime_manifest、协议冒烟、暖机小标定及生命周期配对探针。受控 GPU/host 两层实验均显示：保留旧 epoch 时旧/新/共享路径为 256/128/128 tokens；退役旧 epoch 后在相同原生 128-token 淘汰压力下变成 128/256/128。退役瞬间两个 pool 的占用变化均为 0。该 fixture 故意在新路径后再次访问旧路径以隔离软引用作用，commit 边界由工具模拟；它证明限定配置中的机制，不证明一般工作负载吞吐收益。完整证据在容器 runs/lifecycle-paired.json，解释见 [引擎验证](../engine_patches/README.md)。

2026-10-24 从 5 个开发会话按版本年龄 0/1/2+ 分层冻结 15 个 pre-turn 案例，观察到 APPEND 输入最大 11217 tokens，超出此前 8192 的采样上限。因此在相同引擎配置追加 16384 长度标定：prefix 比例 0/.5/.9/.99、并发 1/4、输出 128/512，28 个兼容 case，每 case 1 次 warmup 和 3 次测量，共 70 个预热请求、210 个测量请求、778 条阶段/已完成传输记录。原 1500 条记录独立保留，补充数据在 runs/remote-h200-expanded；扩大覆盖本身不是方法收益。两个 estimator 的支持域已共同加入实测总长度边界，phase 别名按同一预测特征分组，模拟拟合来源不能被加载为硬件校准。

随后 15 cases × 3 缓存状态 × 2 次重复共 90 对全部完成，82/90 双臂处于标定支持域。Lookup 6 次成本切换，平均单步 regret 366.55 ms；同一观测上的 positive 离线重评分未切换，regret 437.05 ms。固定 REBUILD 为 129.14 ms，8192 阈值为 177.11 ms；当前结果不支持 B 优于简单策略。84/90 对的 Decode 差主要来自输出长度，180 个臂中有 18 个截断；这是并发 1、冻结历史下的诊断，尚无质量调整或闭环基线结论。详见[分层决策验证](stratified_decision_validation.md)。

**运行限制。** 原生 kernel + page_first 在首次 D2H 出现段错误，栈为 staged HiCache → cudaMemcpyBatchAsync / cuMemcpyBatchAsync_v2。保持固定源码和 host 语义，改用已有 layer_first 并设置 SGLANG_HICACHE_TMA_TRANSFER=0 后通过上述验收。尚未隔离具体驱动/运行时原因，不声称修复上游 kernel。正式比较必须统一兼容配置并重新标定。

**评测限制。** Cloud 512-token 配置有 21/41 轮截断，成功 receipt 不等于完整优质答案。按技术方案单列 1024-token 敏感性后，41/41 成功、0 截断、平均输出 536 tokens；记录于容器 runs/cloud-dev-p2-output1024。该运行没有控制与前一次相同的缓存初态，只比较截断现象，不能宣称时延改进。正式对照应给所有方法相同输出上限，512 原始结果保留。首次小标定含初始化影响，仅证明管线可运行；正式采集需独立 warmup、充分范围与留出误差报告。

## 已验证的控制开销改进

新增共享提示词开发验证共六组，每组固定 41 轮：Q0/Q1/Q2 分别运行 P1 与 P2。Q1/Q2 减少截断或缩短回答，但完整原文复核仍发现跨产品错答和错引，未选为默认，也未将输出变短算为质量保持下的加速。组合 ID 引用的实际漏计已修正；所有已知 ID 均存在仍不代表支持论断。完整自动统计、24 条定向论断审查及复算入口见[共享提示词验证](shared_prompt_validation.md)。

证据身份只在入口计算一次：100 个已归档证据＋1 个新证据的合成 REBUILD 中，正文 hash 次数从 304 降至 1。实际 token 绑定、请求幂等和 CAS/epoch 约束保留。

252 个静态 Decode 标定点、输出均值 128/P90 256 的本机 CPU 微基准，单候选中位预测耗时由 38.22 ms 降到 0.58 ms；区间案例由 15.55 ms 降到 0.46 ms。原插值、误差与支持范围语义通过随机稀疏/密集、重复点、区间、边界及完整预测等价性测试。证据为 runs/cost-profile/comparison.json；这是 CPU 开销改善，不是 GPU 端到端加速比。

15 个真实长会话快照的局部 CPU profile 显示重复类型注解解析和标量通用遍历可避免；只缓存静态 schema 并增加标量快路径后，临时 WAL/FULL 部分存储链的等案例均值从 47.22 ms 降为 13.46 ms，JSON 字节和 typed contract 保持一致。候选构建的哈希本身占比很低，没有因此删除证据校验。该局部结果不包含 RPC、并发等待和全部事务成本，不能换算为远端控制 P95；详见[控制路径诊断](controller_control_profile.md)。

远端 P1/always_rebuild/c4 的一次 before→after 对照均 41/41 成功，全部输入 IDs 相同，control P95 为 456.11→168.68 ms；16 个同输入/输出样本的 control P95 为 378.55→157.07 ms。P2 对照虽然也观察到控制耗时下降，但只有 6/41 输入相同，历史与动作变化存在混杂。两组均非随机交叉重复实验，P1 输入仅 2460–4231 tokens；不外推长 APPEND、质量一致或一般吞吐增益。原始数据在 runs/remote-h200-control，逐项校验与统计见同一控制路径文档。

候选联合分词进一步复用本次调用中相同的特殊 token 边界片段，保持完整文本及 IDs 等价；本地完整构建 13.31→4.63 ms。Linux 四线程分词局部 wall 22.61→7.58 ms、CPU 22.49→17.44 ms；默认大线程池则 CPU 增加，不能泛称减少算量。源码依据、逐 token 验证和线程预算见[候选分词测量](candidate_tokenization_profile.md)。

## 新增方法与真实开发验证

- 候选长度、条件误差、会话留一训练及离线 planner 重评分已实现。15 cases / 5 sessions / 90 pairs 的联合探索中，conditional＋candidate OOF regret 为 270.30 ms，低于原 global 437.05 ms，但仍高于固定 REBUILD 的 129.14 ms。完整 GPU P2/c4 集成 41/41 成功；该运行使用 full-fit dev artifact，不是 OOF 验证，且没有触发成本节省切换。见[长度与决策验证](length_decision_validation.md)。
- 原硬件控制器矩阵 11 jobs 和匹配消融 5 jobs，共 656/656 成功。所有实际 host 恢复为零，退役 ACK 不虚报物理释放；该结果不能证明 host 项或容量竞争收益。逐轮机制诊断见[硬件矩阵分析](hardware_controller_matrix.md)。
- 固定 MiniCheck 在 H200 上完成 Q1/Q2 各 41 个完整 audit 的评分。已修复把引用 ID 当事实输入产生伪负例的问题；两个高置信引用替换候选仍经原文审查发现部分支持及段尾绑定歧义，因而默认改为只增补来源。分句/未覆盖/评分与修正局限见[assistant 审查](grounding_v2_assistant_audit.md)，不计为人工盲审或正式质量通过。
- 原始 72442 passage 的 BGE-M3 索引与三种 query 视角、四阶段检索均已在 GPU 执行。39/41 个有 qrels 的开发问题上，last3/BM25 Recall@5=0.2094、nDCG@5=0.1671；current/hybrid_rerank=0.5231、0.4833。原文、候选池、模型版本、时长和全部阶段结果保存在 runs/remote-h200-retrieval。该提升属于已观察开发集的检索相关性，不等于回答质量或缓存收益。
- 嵌套会话留一的增益模型将候选长度分支 regret 270.30→236.14 ms，但决策与简单的稳定期内支持域门控完全一致，尚无细粒度学习增量。见[增益校准验证](gain_calibration_validation.md)。
- P3 BM25 与 BGE 增强各 41/41 成功，检索 ID/次序与离线结果一致；增强分支检索均值 4.94→699.26 ms，完整轮均值 2.546→3.624 s，且回答仍有事实/代码错误。纠错查询又完成 41/41，取回 10MB 附件限制表，但回答仍错误合并文档与附件限制。详见[P3 集成](p3_retrieval_integration.md)、[纠错查询](correction_query_view.md)和[代码审计](code_audit.md)。
- 固定 Qwen3-Reranker-0.6B 在同一 41 问题、2050 个候选对上真实评分；无截断或接口对接错误，Recall@5/nDCG@5 却降至 0.4085/0.3907。因此保留 BGE 为增强分支选择，负面结果及未标注替代相关段落一并保留。见[较新重排器验证](qwen3_rerank_validation.md)。
- 原生三种 HiCache 写策略分别完成 32K 和 16K GPU 池各 123 轮。32K 无 host 恢复；16K 恢复 10752/896/13120 tokens，各组全部成功、无 retraction。单次运行、输出和等待差异不足以选 B6，原始证据见[容量与写策略](native_policy_validation.md)。
- 16K 研究池独立完成 76 个 case 的标定：552 个正式请求、184 个独立 warmup，共 2268 条阶段/完成传输观测；四份 manifest 与合并文件来源一致。c4 首 token 留出仅 50% 受支持，长输入的输出预算变化主要反映在首次 forward 前的联合等待/准备区间；配置匹配不等于预测准确。详见[有限池标定分析](capacity_calibration_validation.md)。
- 同一 16K/279040-host 配置的六方法矩阵共 246/246 成功。Ours 与 no-retire 的 41 轮输入/输出完全相同，但仍有 34 次固定策略回退、零成本切换；完成均值 5.325/5.313 s，未观察到退役的时延收益。Rebuild 为 4.122 s，但与其 no-retire 只有 6 轮输入一致，不能将差值直接归因 C。完整状态、真实恢复/传输、回退因子与初态局限见[有限容量联合验证](controller_capacity_validation.md)。
- 质量验收新增双侧原始问题/历史/检索材料/输入身份绑定，核对原运行的 split 声明与既有摘要，并汇总各层模拟标记。独立反例审查修正了开发数据冒充 heldout、模拟来源被 audit 遮蔽、相同答案挪用到不同提示三个漏检。真实 P3 的 123 轮原始绑定均通过；v2 盲审包评分仍全部为空，不建立质量结论。见[质量验收](quality_acceptance.md)。
- 字面表格注释已完成 82 轮 P3 与 8 次冻结历史实验；随后直接复用 CRAFT 的 MIT 行文本函数，扩展当前 query 的词法焦点与行内来源。两固定历史、三臂正反顺序共 12 次 GPU 运行全部成功，新样式在两历史中均保留正确单附件限制和支持它的原引用，未重现已知范围错误。已可选接入共享候选、原生基线和矩阵；完整正文/QA/预算保持，默认关闭。仅是开发探针上的局部改善，见[焦点行验证](table_focus_validation.md)。
- Selene Mini 8B 与 Prometheus 2 7B 的原生五维协议各完成 10 次真实评分，均正常解析，但关键范围扩张与缺引用仍有漏判。GroUSE 固定 train16 分别完成 45/32 次调用；Selene 条件通过48/96，6次字符串 null 失败；Prometheus 30次输出截断、2次JSON失败，零分代表该协议运行失败。官方解析/DTO与判分重放一致。显式 normalized_v1 已实现并完整重跑 train：51次调用全解析、64/96条件通过，完整性仍仅5/16；不回填旧调用，不按格式完成率宣布语义可靠。见[独立评审验证](open_judge_validation.md)。
- 同一优化源码的 gate2/gate0/Fixed4096/Rebuild 两次对照完成328/328轮；源码、8份冷态和全部原始记录通过独立复核。Gate0等会话完成均值下降19.88%，但比Fixed4096慢15.16%，P2输入输出分叉且仍有截断，不建立质量相等或强B6胜出结论。见[稳定期消融](stage_gate_validation.md)。
- `focus_refs` 完整P3三臂共123/123成功、每臂2轮截断，完整输入/QA/来源重建通过。附件目标有原始引用的10MB回答改善了已知范围错误，但后续套餐条件、SDK跨产品混引仍存在，非触发轮也因闭环生成变化而分叉，见[P3焦点验证](table_focus_p3_validation.md)。
- 新增[原子声明诊断](atomic_judge_proposal.md)复用既有HTTP、完整输入绑定与来源合同，适配RefChecker/Claimify等抽取和来源核验原则。实际代码支持抽取、原文位置、限定槽位、可见/被引来源分判及未知/失败记录；新增可选JSON Schema协议并通过真实XGrammar 0.2.7编译；v1/v2/v3分别在同一29输入完成42/58/67次调用。v3消除了两条顶层JSON失败，但24条prose仍为8 matched、5 mismatched、11 unknown，与v2相同。详见[原子实测](atomic_judge_validation.md)，不进入正式五维质量gate，也未在线修答案。

- 新预算/匹配观测补丁 `3aac1df…` 已完成[H200功能验收](native_observation_validation.md)：三份保留报告共474个报价预算逐项复算一致，GPU-v2与host各三请求完成/退役/拒绝旧epoch通过；首次probe的HTTP代码预期错误与客户端消息丢失均保留并修正。真实fixture全部ignoreEOS，尚未覆盖ratio缩放或clip饱和；normalRAG的82轮事件没有quote，不作补充覆盖。
- [用户主题五视图检索](user_topic_retrieval_validation.md)完成820条结果，R@5提升至54.87%，保留R@50下降及WebSocket候选丢失。新完整P3两臂各41轮全部执行成功、各一轮截断，82条输入/历史/正文/receipt核对无错误；语义审查仍发现SDK扩写和跨平台混用。

- [双路融合真实检索](current_topic_fusion_validation.md)完成82/82尝试、246阶段结果；86次query编码、4100对重排。新增75对全部实际评分，R@50恢复至76.07%，最终R@5/nDCG保持54.87%/51.49%；38/41 top5相同，3轮材料变化已进入[完整P3两臂验证](current_topic_fusion_p3_validation.md)：82/82成功、各一轮生成截断，完整输入重建和检索绑定均通过；定义/链接有局部改善，但无据API、版本和产品混用仍存在，默认single保持不变。

## 仍需完成的研究验收

当前没有正式回答质量非劣、SLO goodput提升或整体优于强基线的结论。[研发策略](research_strategy.md)是最新投入顺序：先检验共同提示下额外规划的机制价值，保留或替换后才接入自有历史P2、共同Cloud P3及有限B/C联合。normal-EOS、定向标定、完整模型评分、第二模型、共享pin等已完成项不再作为待办重跑。

533答案/2665维评分已经完整，但12候选均未获联合资格；当前缺口是评价器可靠性、统计证据与真实人工盲审，不是缺评分工具。318份Qwen3关键论断审查属于assistant审查，不补造人工评分或总体错误率。最终代码/模型/采样/资源/方法与评价协议固定后，才执行预声明heldout主比较；失败、不确定和无加速同样进入交付。

工程可运行、研究批次完整、质量/性能结论成立分别记录。保留现有强简单基线与负结果，复杂机制须证明净收益；不机械补齐所有容量/写策略/模型矩阵，也不通过改epsilon、重复调参或换judge取得想要的结论。安装与实跑方法见[部署说明](deployment.md)、[同模型原生基线](native_model_scope.md)和[执行计划](implementation_plan.md)。
