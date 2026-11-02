"""Checkpoint and gradient tests on a tiny synthetic encoder, not study assets."""
from dataclasses import asdict
import json

import pytest

from adapters.support_model import SupportModel
from controller.context_schema import EvidenceKey, EvidencePool, EvidenceUnit, Need, SourceSpan, TaskState
from training.support_model import train_support, train_set_utility

torch = pytest.importorskip("torch")


def test_support_and_preference_training_checkpoint_round_trip(tmp_path):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "linux", "windows", "eligible",
        "submit", "application", "question", "user", "only", "procedure", "resident"])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.normalizer = normalizers.BertNormalizer(lowercase=True)
    core.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tokenizer.save_pretrained(tmp_path/"base")
    encoder = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=512))
    encoder.save_pretrained(tmp_path/"base")
    units = []
    for uid, text in (("eligible", "linux eligible resident"), ("wrong", "windows only application")):
        key = EvidenceKey("synthetic-test", uid, uid, "v1", uid)
        units.append(EvidenceUnit(uid, (SourceSpan(uid, key, 0, len(text)),), text, body_tokens=3))
    rows = [{"task_id": "synthetic-test", "need_id": "n", "split": "train",
        "query": "linux eligible application", "units": [asdict(unit)], "dependencies": [],
        "label_origin": "synthetic_implementation_test", "labels": {
            "applicability": float(index == 0), "direct_support": float(index == 0),
            "joint_support": float(index == 0), "bridge_value": float(index == 0),
            "disposition": "complete" if index == 0 else "contradict",
            "quality": int(index == 0), "answer_disposition": "answer" if index == 0 else "retrieval_gap",
            "source_spans": [asdict(span) for span in unit.source_spans] if index == 0 else []}}
        for index, unit in enumerate(units)]
    examples = tmp_path/"examples.jsonl"
    examples.write_text("".join(json.dumps(row)+"\n" for row in rows))
    metadata = train_support(examples, tmp_path/"base", tmp_path/"support", device="cpu",
        epochs=2, batch_size=2, max_length=512, graph_width=16,
        base_model_id="synthetic-implementation-test")
    assert len(metadata["support_training"]["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in metadata["support_training"]["losses"])
    model = SupportModel.from_pretrained(tmp_path/"support", history_resolver=lambda ref: "", device="cpu")
    state = TaskState("synthetic-test", "linux eligible application", needs=(Need("n"),))
    graphs = []
    for unit in units:
        scores = model.score_support(state, EvidencePool((unit,)))
        assert 0 <= model.predict_set(state, (unit,), scores.singles).quality <= 1
        graphs.append({"answer": "Synthetic implementation test answer", "synthetic_fixture": True,
            "reader_model": "qwen3_32b_nonthinking", "units": [asdict(unit)],
            "supports": [asdict(support) for support in scores.singles], "disposition": "answer",
            "support_inputs": [{"need_id": "n", "query": "linux eligible application",
                                "units": [asdict(unit)], "dependencies": []}]})
    assert 0 <= model.predict_set(state, (), ()).quality <= 1
    preferences = tmp_path/"preferences.jsonl"
    preferences.write_text(json.dumps({"task_state": asdict(state), "split": "train",
        "label_origin": "actual_reader_answers", "synthetic_fixture": True,
        "preferred": graphs[0], "dispreferred": graphs[1]})+"\n")
    utility = train_set_utility(preferences, tmp_path/"support", tmp_path/"utility", device="cpu", epochs=2)
    assert len(utility["utility_training"]["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in utility["utility_training"]["losses"])
    restored = SupportModel.from_pretrained(tmp_path/"utility", history_resolver=lambda ref: "", device="cpu")
    scores = restored.score_support(state, EvidencePool((units[0],)))
    assert 0 <= restored.predict_set(state, (units[0],), scores.singles).quality <= 1
    model.close()
    restored.close()
