"""Fresh source-grounded training dialogues for relational and version task heads."""
from __future__ import annotations

from collections import defaultdict
import random

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from controller.serialization import to_jsonable


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
