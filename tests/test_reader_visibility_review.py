"""Reader targets use visible facts; full-source review evaluates the fixed target."""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from training.reader_data import (
    history_digest, prepare_reader_data, reader_review_body, reader_target_body,
    review_reader_trace, rows, write_rows,
)


def answer_trace():
    history = [{"turn_id": 1, "user_text": "Read the whole register.",
                "assistant_text": "HIDDEN_ACTUAL_HISTORY"}]
    source_text = "Dispatch D01: 1 kg.\nDispatch D36: 36 kg.\nRegister total: 666 kg."
    return {
        "example_id": "attempt:ANSWER", "attempt_id": "attempt", "task_id": "dispatch",
        "conversation_id": "conversation", "split": "train", "phase": "ANSWER",
        "question": "Count all dispatches and sum their mass.",
        "messages": [{"role": "system", "content": "Use full-scope reading conclusions."},
            {"role": "user", "content": "Dispatch D01: 1 kg.\nComputed: count=36; sum=655000 g."}],
        "actual_history": history, "actual_history_sha256": history_digest(history),
        "prompt_token_hash": "recorded-prompt", "reader_identity": "R0",
        "read_protocol_version": None, "task_state": {"extra_state": "HIDDEN_STATE"},
        "actual_output": "OLD_UNSUPPORTED_OUTPUT",
        "original_sources": [{"source_key": {"document_id": "dispatch"},
            "text": source_text, "start": 0, "end": len(source_text)}],
        "reading_steps": [{"phase": "REDUCE", "step_id": "reduce:mass", "operation": "sum",
            "claims": [{"claim_id": "reduce:mass:claim:0", "value": "655000", "unit": "g"}],
            "covered_units": []}],
    }


@dataclass
class Completion:
    text: str
    input_tokens: int
    usage: dict
    status: str = "completed"
    finish_reason: str = "stop"
    elapsed_ms: float = 1.0


class Tokens:
    def encode(self, text, **kwargs):
        return list(text.encode())


class ReaderVisibilityTests(unittest.TestCase):
    def test_hidden_sources_and_history_are_only_in_the_review_view(self):
        trace = answer_trace()
        target = reader_target_body(trace)
        self.assertEqual(target["messages"], trace["messages"])
        self.assertEqual(set(target), {"phase", "messages", "target_output_budget_tokens"})
        target_text = json.dumps(target)
        for hidden in ("D36", "666 kg", "HIDDEN_ACTUAL_HISTORY", "HIDDEN_STATE", "OLD_UNSUPPORTED_OUTPUT"):
            self.assertNotIn(hidden, target_text)
        review = reader_review_body(trace, target_text="36 dispatches; 655 kg.")
        self.assertIn("D36", review["original_sources"][0]["text"])
        self.assertEqual(review["additional_actual_history"], trace["actual_history"])
        self.assertEqual(review["reading_steps"], trace["reading_steps"])

    def test_independent_review_records_upstream_error_without_rewriting_target(self):
        trace = answer_trace()
        calls, costs = [], []
        fixed_target = "36 dispatches; 655 kg."

        class Client:
            tokenizer = Tokens()

            def complete(self, messages, *, response_format):
                stage = response_format["json_schema"]["name"]
                body = json.loads(messages[-1]["content"])
                calls.append((stage, body))
                if stage == "reader_target":
                    assert "D36" not in json.dumps(body)
                    return Completion(json.dumps({"target": fixed_target}), 40, {"completion_tokens": 8})
                assert body["target_text"] == fixed_target
                assert "D36" in body["original_sources"][0]["text"]
                assert "target" not in response_format["json_schema"]["schema"]["properties"]
                return Completion(json.dumps({"quality": .2, "source_support_verified": False,
                    "correct_task_binding": True, "visible_prompt_support": True,
                    "upstream_errors": [{"phase": "REDUCE", "step_id": "reduce:mass",
                        "reason": "The completed register totals 666 kg; the visible root reports 655 kg."}]}),
                    80, {"completion_tokens": 20})

        record = asyncio.run(review_reader_trace(trace, Client(), minimum_quality=.85,
            reviewer_identity={"fixture": True}, review_input_budget=32768, record_call=costs.append))
        self.assertEqual([name for name, _ in calls], ["reader_target", "reader_review"])
        self.assertEqual(record["target_text"], fixed_target)
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["upstream_errors"][0]["step_id"], "reduce:mass")
        self.assertEqual([row["stage"] for row in costs], ["target", "review"])
        self.assertEqual(sum(row["model_input_tokens"] for row in costs), 120)
        self.assertEqual(sum(row["model_output_tokens"] for row in costs), 28)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root/"traces.jsonl", [trace])
            write_rows(root/"reviews.jsonl", [record])
            prepare_reader_data({"reader_training": {"data": {"output_dir": directory,
                "read_protocol_versions": [2], "minimum_quality": .85,
                "validation_fraction": 0, "split_seed": 42}}})
            self.assertEqual(rows(root/"train.jsonl"), [])
            self.assertEqual(rows(root/"reviews.jsonl")[0]["target_text"], fixed_target)

    def test_source_true_but_prompt_hidden_fact_is_not_an_answer_training_target(self):
        trace = answer_trace()

        class Client:
            tokenizer = Tokens()

            def __init__(self, visible):
                self.visible = visible

            def complete(self, messages, *, response_format):
                if response_format["json_schema"]["name"] == "reader_target":
                    return Completion(json.dumps({"target": "36 dispatches; 666 kg."}), 40,
                                      {"completion_tokens": 8})
                return Completion(json.dumps({"quality": 1, "source_support_verified": True,
                    "correct_task_binding": True, "visible_prompt_support": self.visible,
                    "upstream_errors": []}), 80, {"completion_tokens": 20})

        record = asyncio.run(review_reader_trace(trace, Client(False), minimum_quality=.85,
            reviewer_identity={"fixture": True}, review_input_budget=32768, record_call=lambda row: None))
        self.assertEqual(record["status"], "rejected")
        self.assertTrue(record["source_support_verified"])
        self.assertFalse(record["visible_prompt_support"])
        self.assertEqual(record["target_text"], "36 dispatches; 666 kg.")
        trace["messages"][-1]["content"] = trace["messages"][-1]["content"].replace("655000", "666000")
        accepted = asyncio.run(review_reader_trace(trace, Client(True), minimum_quality=.85,
            reviewer_identity={"fixture": True}, review_input_budget=32768, record_call=lambda row: None))
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["target_text"], record["target_text"])


if __name__ == "__main__":
    unittest.main()
