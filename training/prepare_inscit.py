"""Build matched single-article and cross-article follow-ups from InSCIt's train split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from controller.schema import EvidenceKey, EvidenceUnit, SourceSpan
from controller.serialization import to_jsonable


REVISION = "2319fb85932b9528a13417223ffc6fc629ae8087"


def article(passage_id):
    return passage_id.rsplit(":", 1)[0]


def paired_turns(dialogues, dev_sessions=20, seed=42):
    candidates = []
    for session, dialogue in sorted(dialogues.items()):
        turns = dialogue["turns"]
        articles = {dialogue["seedArticle"]["title"].replace("_", " ").casefold()}
        single, multi = [], []
        for i, turn in enumerate(turns):
            labels = turn["labels"]
            articles.update(article(p["passage_id"]).replace("_", " ").casefold()
                            for label in labels for p in label["evidence"])
            if i == 0:
                continue
            label = labels[0]
            if i + 1 < len(turns):
                reply = turns[i+1]["context"][-2].strip()
                label = next(item for item in labels if item["response"].strip() == reply)
            if label["responseType"] == "directAnswer" and label["evidence"]:
                documents = {article(p["passage_id"]) for p in label["evidence"]}
                (single if len(documents) == 1 else multi).append((i, label))
        if single and multi:
            cross = multi[0]
            control = min(single, key=lambda item: abs(item[0] - cross[0]))
            candidates.append((session, articles, sorted((control, cross))))

    random.Random(seed).shuffle(candidates)
    held_out = set().union(*(docs for _, docs, _ in candidates[:dev_sessions]))
    for index, (session, docs, pair) in enumerate(candidates):
        split = "dev" if index < dev_sessions else "train"
        if split == "train" and not docs.isdisjoint(held_out):
            continue
        for i, label in pair:
            turn = dialogues[session]["turns"][i]
            cited = {p["passage_id"]: p for p in label["evidence"]}
            alternatives = {p["passage_id"] for other in turn["labels"] for p in other["evidence"]}
            history_passages = {p["passage_id"]: p for group in reversed(turn["prevEvidence"])
                                for p in group if p["passage_id"] not in alternatives}
            substitutes = list(history_passages)[:len(cited)]
            passages = [*cited.values(), *(history_passages[k] for k in substitutes)]
            task_id = f"inscit:{session}:{i+1}"
            random.Random(task_id).shuffle(passages)
            choices = [tuple(cited)]
            for doc in dict.fromkeys(map(article, cited)):
                kept = tuple(k for k in cited if article(k) != doc)
                choices.append(kept)
                count = len(cited) - len(kept)
                if len(substitutes) >= count:
                    replacements = iter(substitutes)
                    choices.append(tuple(next(replacements) if article(k) == doc else k for k in cited))
            yield {"dataset": "inscit", "task_id": task_id, "conversation_id": f"inscit:{session}",
                   "turn": i+1, "split": split, "question": turn["context"][-1],
                   "history": [{"turn_id": j//2+1, "speaker": "user" if j%2 == 0 else "assistant",
                                "text": text} for j, text in enumerate(turn["context"][:-1])],
                   "reference_answer": label["response"], "gold_passage_ids": list(cited),
                   "paired_sets": list(dict.fromkeys(choices)), "passages": passages}


def materialize(row, tokenizer):
    units = []
    for p in row["passages"]:
        pid, text = p["passage_id"], p["passage_text"]
        key = EvidenceKey("inscit", article(pid), pid, REVISION, "")
        units.append(EvidenceUnit(pid, (SourceSpan(pid, key, 0, len(text)),), text,
                                  " / ".join(p["passage_titles"]), document_family=key.document_id,
                                  body_tokens=len(tokenizer.encode(text, add_special_tokens=False))))
    return {k: v for k, v in row.items() if k != "passages"} | {
        "catalogue": [], "units": to_jsonable(units)}


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = paired_turns(json.loads(args.train_json.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            output.write(json.dumps(materialize(row, tokenizer), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
