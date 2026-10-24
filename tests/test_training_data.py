"""Preparation preserves the causal input, target-only loss and failed denominator."""
from dataclasses import asdict, replace
from hashlib import sha256
import json

import pytest

from controller.schema import Message
from controller.templates import ByteTemplate
from training.data import load_prepared, prepare_examples
from training.tokenization import SequenceTooLong


class TrainingByteTemplate(ByteTemplate):
    enable_thinking = False

    def __init__(self):
        super().__init__()
        self.tokenizer.eos_token_id = 300
        self.tokenizer.all_special_ids = [300]


def example(identifier="one", **changes):
    return {
        "id": identifier, "split": "train",
        "messages": [{"role": "system", "content": "Original source condition."},
                     {"role": "user", "content": "Question?"},
                     {"role": "user", "content": "Original added detail."},
                     {"role": "assistant", "content": "Earlier answer."}],
        "target": "\n完整回答。 ",
        "provenance": {"source": "example://original", "review": {"note": "not model input"}},
        **changes,
    }


def write_source(tmp_path, rows):
    path = tmp_path / "input.jsonl"
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())
    return path


def read_manifest(output):
    return json.loads((output / "manifest.json").read_text())


def test_prepare_and_load_preserve_exact_prefix_mask_target_eos_and_counts(tmp_path):
    rows = [example(), example("two", target="", provenance={})]
    path, output, template = write_source(tmp_path, rows), tmp_path / "prepared", TrainingByteTemplate()
    manifest = prepare_examples(path, output, template, max_tokens=1000)
    saved = [json.loads(raw) for raw in (output / "examples.jsonl").read_text().splitlines()]
    prefix = template.render(tuple(Message(**m) for m in rows[0]["messages"])).input_ids
    target = list(rows[0]["target"].encode()) + [300]
    assert saved[0]["input_ids"] == list(prefix) + target
    assert saved[0]["labels"] == [-100] * len(prefix) + target
    assert saved[0]["attention_mask"] == [1] * (len(prefix) + len(target))
    assert saved[0]["provenance"] == rows[0]["provenance"]
    assert saved[1]["labels"] == [-100] * len(prefix) + [300]
    assert manifest["fingerprint"] == asdict(template.fingerprint)
    assert manifest["counts"] == {"read": 2, "written": 2, "failed": 0, "unprocessed": 0}
    assert manifest["tokens"] == {"prompt": len(prefix) * 2, "target": len(target) + 1,
                                  "total": len(prefix) * 2 + len(target) + 1,
                                  "max_sequence": len(prefix) + len(target)}
    assert manifest["max_tokens"] == 1000 and manifest["target_tokens_include_eos"]
    assert manifest["source"]["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert manifest["output"]["sha256"] == sha256((output / "examples.jsonl").read_bytes()).hexdigest()
    # Neither the original source nor a tokenizer is required at load time.
    path.unlink()
    features, loaded_manifest = load_prepared(output, template.fingerprint)
    assert loaded_manifest == manifest == read_manifest(output)
    assert features == [{key: row[key] for key in ("input_ids", "labels", "attention_mask")} for row in saved]
    assert all(set(row) == {"input_ids", "labels", "attention_mask"} for row in features)


def test_overflow_preserves_written_rows_and_complete_source_denominator(tmp_path):
    rows = [example(target="small"), example("overflow", target="Untruncated target. " * 100),
            example("after", target="must not be encoded")]
    path, output, template = write_source(tmp_path, rows), tmp_path / "prepared", TrainingByteTemplate()
    prompt = len(template.render(tuple(Message(**m) for m in rows[0]["messages"])).input_ids)
    with pytest.raises(SequenceTooLong) as exc:
        prepare_examples(path, output, template, max_tokens=prompt + 20)
    assert exc.value.target_tokens == len(rows[1]["target"].encode()) + 1
    manifest = read_manifest(output)
    assert manifest["status"] == "failed" and manifest["failed_line"] == 2
    assert manifest["counts"] == {"read": 3, "written": 1, "failed": 1, "unprocessed": 1}
    assert manifest["tokens"]["target"] == len(b"small") + 1
    assert manifest["source"]["sha256"] == sha256(path.read_bytes()).hexdigest()
    saved = (output / "examples.jsonl").read_bytes()
    assert manifest["output"]["sha256"] == sha256(saved).hexdigest()
    assert len(saved.splitlines()) == 1
    assert json.loads(saved)["input_ids"][-6:] == list(b"small") + [300]
    with pytest.raises(ValueError, match="completed"):
        load_prepared(output, template.fingerprint)


@pytest.mark.parametrize("change", [{"split": "dev"}, {"split": "test"}, {"split": None}])
def test_only_explicit_training_split_is_accepted(tmp_path, change):
    path, output = write_source(tmp_path, [example(**change)]), tmp_path / "prepared"
    with pytest.raises(ValueError, match="train split"):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    assert read_manifest(output)["counts"] == {"read": 1, "written": 0, "failed": 1, "unprocessed": 0}


def test_duplicate_ids_fail_without_dropping_or_renaming(tmp_path):
    path, output = write_source(tmp_path, [example(), example()]), tmp_path / "prepared"
    with pytest.raises(ValueError, match="duplicate"):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    assert read_manifest(output)["counts"]["written"] == 1


def test_load_rejects_modified_features_even_if_json_is_still_valid(tmp_path):
    template, output = TrainingByteTemplate(), tmp_path / "prepared"
    prepare_examples(write_source(tmp_path, [example()]), output, template, max_tokens=1000)
    file = output / "examples.jsonl"
    row = json.loads(file.read_text())
    row["labels"][0] = 42
    file.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        load_prepared(output, template.fingerprint)


def test_load_rejects_wrong_template_and_false_count(tmp_path):
    template, output = TrainingByteTemplate(), tmp_path / "prepared"
    manifest = prepare_examples(write_source(tmp_path, [example()]), output, template, max_tokens=1000)
    with pytest.raises(ValueError, match="fingerprint"):
        load_prepared(output, replace(template.fingerprint, model_revision="different"))
    manifest["counts"]["written"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count"):
        load_prepared(output, template.fingerprint)


@pytest.mark.parametrize("change", [{"id": ""}, {"id": 4}, {"target": None},
                                   {"messages": "text"}, {"messages": [{"role": "user", "content": 4}]},
                                   {"messages": []}, {"provenance": []}])
def test_invalid_explicit_fields_fail_at_preparation_boundary(tmp_path, change):
    path, output = write_source(tmp_path, [example(**change)]), tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    assert read_manifest(output)["status"] == "failed"


def test_directory_is_exclusive_and_failed_output_is_not_overwritten(tmp_path):
    path, output = write_source(tmp_path, [example(split="test")]), tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    assert (output / "manifest.json").read_bytes() == original


def test_empty_file_is_a_failed_preparation(tmp_path):
    path, output = write_source(tmp_path, []), tmp_path / "prepared"
    with pytest.raises(ValueError, match="no examples"):
        prepare_examples(path, output, TrainingByteTemplate(), max_tokens=1000)
    assert read_manifest(output)["counts"] == {"read": 0, "written": 0, "failed": 0, "unprocessed": 0}
