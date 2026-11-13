"""Fresh source-grounded training dialogues for relational and version task heads."""
from __future__ import annotations

from collections import defaultdict
import random

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from controller.serialization import to_jsonable


def coverage_states():
    """Intent-only contrast pairs; no benchmark questions or answers are used."""
    pairs = (
        ("Explain the eligibility rules for this grant.", "List every eligible application across the supplied files."),
        ("What steps are needed to renew the licence?", "Which licences in the complete register have expired?"),
        ("How is this protocol's timeout configured?", "Find every inconsistent timeout definition across the protocol documents."),
        ("Show an example of a dependency between two components.", "Determine whether the complete component dependency graph contains a cycle."),
        ("Why was this contract terminated?", "Identify all contracts terminated before their expiry dates in the supplied collection."),
        ("Give all steps of the standard reimbursement procedure.", "Calculate the total reimbursement across all submitted claims."),
        ("Compare the stated limits in these two policy versions.", "Find every changed limit across the full set of policy versions."),
        ("What does the manual mean by an unavailable service?", "Establish whether any supplied incident report describes an unavailable service."),
        ("请解释这个项目的申请条件。", "请列出资料中全部符合申请条件的项目。"),
        ("怎样办理员工离职手续？", "请找出所给人事记录中所有尚未完成离职手续的员工。"),
        ("说明两份技术说明中接口定义的区别。", "检查整个接口目录是否存在互相矛盾的定义。"),
        ("举例说明设备之间的连接方式。", "根据所有设备连接记录判断整个网络是否连通。"),
        ("这份报告中的异常原因是什么？", "汇总全部报告中的异常事件，并去除重复事件。"),
        ("请说明完整的采购审批流程。", "查找全部订单中没有经过审批的订单。"),
        ("这两项服务的收费有什么区别？", "列出所给价目表中每项服务的最低收费。"),
        ("给出一个满足要求的供应商。", "确认提供的所有供应商是否都满足要求。"),
        ("What caused the sample build failure?", "Identify every failed build in the complete deployment log."),
        ("Explain one path between the named stations.", "Determine the longest path in the supplied acyclic route network."),
        ("请给出一个符合标准的产品实例。", "请检查全部检验记录是否存在不符合标准的产品。"),
        ("这篇研究介绍了什么方法？", "整理全部研究之间的依赖关系并判断是否存在闭环。"),
    )
    result = [{"task_id": f"coverage-{index}-{kind}", "conversation_id": f"coverage-{index}",
        "split": "dev" if index >= 16 else "train", "turn": 1, "history": [], "question": question,
        "coverage_only": True, "task_state": {"coverage_scope": scope}}
        for index, pair in enumerate(pairs) for kind, (question, scope) in enumerate(zip(pair,
            ("selected_sources", "all_supplied_documents")))]
    domains = (
        ("buffer manager", "retention limit", "缓存管理器", "保留期限"),
        ("message broker", "delivery guarantee", "消息服务", "交付保证"),
        ("transport contract", "expiry date", "运输合同", "到期日期"),
        ("grant programme", "eligibility rule", "资助项目", "申请条件"),
        ("sensor module", "operating range", "传感器模块", "运行范围"),
        ("insurance policy", "coverage amount", "保险条款", "保障金额"),
        ("filing procedure", "submission deadline", "申报流程", "提交期限"),
        ("data format", "required fields", "数据格式", "必填字段"),
        ("payment service", "transaction limit", "支付服务", "交易限额"),
        ("maintenance plan", "inspection interval", "维护方案", "检查间隔"),
    )
    for index, (entity, attribute, chinese_entity, chinese_attribute) in enumerate(domains):
        version = f"{index+1}.3.2"
        variants = (
            (f"What is the {attribute} of the {entity} in version {version}?", 0, ()),
            (f"How does the {entity} behave in version {version}?", 0, ()),
            (f"List every {attribute} change across all supplied versions of the {entity}.", 1, ()),
            (f"Find all inconsistent {attribute} statements throughout the supplied documents.", 1, ()),
            (f"在 {version} 版中，{chinese_entity}的{chinese_attribute}是什么？", 0, ()),
            (f"请找出全部资料中{chinese_entity}的每项{chinese_attribute}变更。", 1, ()),
            (f"The {attribute}?", 0, (f"Explain the {entity} in version {version}.",)),
            (f"Now include every {entity} in all the supplied records.", 1,
             (f"What is the {attribute} of the first {entity}?",)),
        )
        for kind, (question, population, history) in enumerate(variants):
            result.append({"task_id": f"coverage-domain-{index}-{kind}", "conversation_id": f"coverage-domain-{index}",
                "split": "train" if index < 8 else "dev", "turn": len(history)+1,
                "history": [{"turn_id": i+1, "speaker": "user", "text": text} for i, text in enumerate(history)],
                "question": question, "coverage_only": True,
                "task_state": {"coverage_scope": "all_supplied_documents" if population else "selected_sources"}})
    # Different domains share the same quantifier and relationship semantics.
    # Long formatting instructions are deliberately independent of the scope label.
    populations = (
        ("warehouses", "storage capacity", "仓库", "容量"),
        ("farms", "annual rainfall", "农场", "年降水量"),
        ("machines", "operating temperature", "机器", "运行温度"),
        ("schools", "enrolment", "学校", "招生人数"),
        ("batteries", "cycle life", "电池", "循环寿命"),
        ("reservoirs", "water level", "水库", "水位"),
        ("stations", "daily passengers", "车站", "日客流量"),
        ("hospitals", "bed capacity", "医院", "床位数量"),
    )
    for index, (entities, field, cn_entities, cn_field) in enumerate(populations):
        questions = (
            (f"Group the {entities} in the supplied records by {field}, using low (below 40), medium (40 to 80), and high (above 80).", 1),
            (f"What is the {field} of the named {entities} in record A?", 0),
            (f"For record A, identify the incoming and outgoing dependencies among these {entities}.", 1),
            (f"Explain the meaning of {field} and give an example from the records.", 0),
            (f"将资料中的{cn_entities}按{cn_field}分为低于40、40至80和超过80三组。每个成员只计入对应的一组。", 1),
            (f"编号A的{cn_entities}，其{cn_field}是多少？", 0),
            (f"找出所给{cn_entities}中与A有直接依赖关系的成员，分别列出依赖A和A依赖的成员。", 1),
            (f"解释资料中{cn_field}的含义，并提供一个实例。", 0),
        )
        for kind, (question, population) in enumerate(questions):
            for long_form in (False, True):
                wrapper = (("请根据提供的资料完成以下任务。保持实体、单位和时期的对应关系；"
                    "没有提及的信息无需推测。输出JSON，不添加解释。\n任务：" if kind >= 4 else
                    "Use the supplied records to complete the task. Preserve entities, units and periods. "
                    "Leave unmentioned information unspecified. Output JSON without explanations.\nTask: ")
                    if long_form else "")
                result.append({"task_id": f"coverage-population-{index}-{kind}-{int(long_form)}",
                    "conversation_id": f"coverage-population-{index}",
                    "split": "train" if index < 6 else "dev", "turn": 1, "history": [],
                    "question": wrapper+question, "coverage_only": True,
                    "task_state": {"coverage_scope": "all_supplied_documents" if population else "selected_sources"}})
    return result


def controlled_states(tokenizer, *, train_families=2, dev_families=1, seed=42):
    """Generate new training sources; no evaluation questions or answers are read."""
    randomizer = random.Random(seed)
    result = []
    for index in range(train_families+dev_families):
        family = f"shared-training-{seed}-{index}"
        split = "train" if index < train_families else "dev"
        names = [f"Account {index}-{letter}" for letter in ("A", "B", "C")]
        transactions = [{"id": f"R{row+1}", "entity": names[row % 3],
            "region": "North" if row % 2 == 0 else "South", "period": 2023 + row % 2,
            "amount": randomizer.randrange(2, 40)*30} for row in range(6)]
        table = "# Transactions\n| Record | Entity | Region | Period | Amount (credits) |\n|---|---|---|---|---|\n"
        table += "".join(f"| {row['id']} | {row['entity']} | {row['region']} | {row['period']} | {row['amount']} |\n"
                         for row in transactions)
        members = "# Participants\n| Entity | Status |\n|---|---|\n" + "".join(
            f"| {name} | {'active' if position != 1 else 'inactive'} |\n" for position, name in enumerate(names))
        values = {("2021", "Windows"): randomizer.randrange(50, 100), ("2021", "Linux"): randomizer.randrange(50, 100)}
        values.update({("2025", platform): values["2021", platform]+randomizer.randrange(10, 40)
                       for platform in ("Windows", "Linux")})
        documents = [{"passage_id": family+":transactions", "document_id": family+":transactions",
            "version": "2025", "title": "Transactions", "text": table},
            {"passage_id": family+":participants", "document_id": family+":participants",
             "version": "2025", "title": "Participants", "text": members}]
        for version in ("2021", "2025"):
            policy = f"# Filing policy {version}\n| Platform | Allowance (credits) |\n|---|---|\n"
            policy += "".join(f"| {platform} | {values[version, platform]} |\n" for platform in ("Windows", "Linux"))
            documents.append({"passage_id": family+":policy:"+version, "document_id": family+":policy",
                "document_family": family+":policy", "version": version, "version_order": [int(version)],
                "current": version == "2025", "valid_from": version+"-01-01",
                "valid_to": "2025-01-01" if version == "2021" else None,
                "title": f"Filing policy {version}", "text": policy})
        for document in documents:
            document["source_uri"] = "training://"+document["passage_id"]
        corpus = CorpusIndex.from_records(documents, corpus_id="shared_training", snapshot_id=family)
        directory = SourceDirectory(corpus, tokenizer)
        by_passage = defaultdict(list)
        for unit in directory.units_by_id.values():
            by_passage[unit.source_spans[0].source_key.passage_id].append(unit)
        total = sum(row["amount"] for row in transactions)
        totals = {region: sum(row["amount"] for row in transactions if row["region"] == region)
                  for region in ("North", "South")}
        active_total = sum(row["amount"] for row in transactions if row["entity"] != names[1])
        transaction_ids = [family+":transactions"]
        policy_ids = [family+":policy:2021", family+":policy:2025"]
        cases = [
            ("Count all records in the complete Transactions table.", "6 records", transaction_ids),
            ("Using those transactions, what is the sum of Amount for the North region?", f"{totals['North']} credits", transaction_ids),
            ("Change the region to South. Give its total Amount.", f"{totals['South']} credits", transaction_ids),
            ("Remove the region condition and sum Amount over every transaction.", f"{total} credits", transaction_ids),
            ("How many distinct Entity values occur across all transactions?", "3 entities", transaction_ids),
            ("What is the mean Amount across all transactions?", f"{total // len(transactions)} credits", transaction_ids),
            ("What is the minimum Amount among all transactions?", f"{min(row['amount'] for row in transactions)} credits", transaction_ids),
            ("What is the maximum Amount among all transactions?", f"{max(row['amount'] for row in transactions)} credits", transaction_ids),
            ("Group all transactions by Region and report each group's total Amount.",
                f"North: {totals['North']} credits; South: {totals['South']} credits", transaction_ids),
            ("Join Transactions to Participants on Entity. Sum Amount only for active participants.",
                f"{active_total} credits", [*transaction_ids, family+":participants"]),
            ("What is the current Windows filing allowance in the supplied policy?",
                f"{values['2025', 'Windows']} credits", policy_ids),
            ("Change the platform to Linux, keeping the current policy.", f"{values['2025', 'Linux']} credits", policy_ids),
            ("What was the Linux filing allowance before that policy?", f"{values['2021', 'Linux']} credits", policy_ids),
            ("Compare the Linux filing allowances in versions 2021 and 2025.",
                f"2021: {values['2021', 'Linux']} credits; 2025: {values['2025', 'Linux']} credits", policy_ids),
            ("For Windows, which allowance applied on 2022-06-01?", f"{values['2021', 'Windows']} credits", policy_ids),
        ]
        history = []
        for turn, (question, answer, passages) in enumerate(cases, 1):
            units = tuple(unit for passage in passages for unit in by_passage[passage])
            catalogue = [{"family": family+":policy", "version": version, "valid_from": version+"-01-01",
                "valid_to": "2025-01-01" if version == "2021" else None, "current": version == "2025"}
                for version in ("2021", "2025")] if passages == policy_ids else []
            grounding = [{"passage_id": passage, "start": 0, "end": len(next(document["text"]
                for document in documents if document["passage_id"] == passage))} for passage in passages]
            result.append({"task_id": f"{family}:{turn}", "conversation_id": family, "turn": turn,
                "split": split, "dataset": "structured_training", "question": question, "history": list(history),
                "catalogue": catalogue, "reference_answer": answer, "grounding": grounding,
                "units": to_jsonable(units)})
            history.extend(({"turn_id": turn, "speaker": "user", "text": question},
                            {"turn_id": turn, "speaker": "assistant", "text": answer}))
    return result
