from adapters.history_state import HistoryWindow, TaskPrediction
from controller.context_schema import Condition, Need, RetrievalState, TaskState, TextRef, VersionIntent
from controller.schema import OnlineTurn, TurnRecord
from training.task_model import SPAN_KINDS, encode_task, task_targets


class CharacterTokenizer:
    cls_token_id = 1000
    sep_token_id = 1001

    def __call__(self, text, **kwargs):
        result = {"input_ids": [ord(c) for c in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


def test_nested_need_subject_and_condition_supervision_keep_unicode_coordinates():
    question = "安装 ORCHID 到 Linux"
    turn = OnlineTurn("s", "t2", 2, "manual", question, (), (), "P3")
    old = Condition("windows", value_spans=(TextRef(1, "user", 9, 16),))
    memory = RetrievalState(1, prior_task_states=(TaskState("t1", "", conditions=(old,)),))
    window = HistoryWindow(TextRef(1, "user", 9, 16), "Windows", 7)
    encoded = encode_task(CharacterTokenizer(), turn, (window,), memory)
    condition = Condition("linux", value_spans=(TextRef(2, "user", 12, 17),),
                          relation_type="PLATFORM", update="REPLACE", target_refs=("windows",))
    prediction = TaskPrediction("correction", (window.ref,), (condition,), (
        Need("install", (TextRef(2, "user", 0, len(question)),), "procedure",
             subject_refs=(TextRef(2, "user", 3, 9),)),))
    targets = task_targets(encoded, prediction)
    position = next(i for i, ref in enumerate(encoded["refs"])
                    if ref is not None and ref.turn_id == 2 and ref.start == 3)
    assert targets["span"][position][SPAN_KINDS.index("NEED")] == 2
    assert targets["span"][position][SPAN_KINDS.index("SUBJECT")] == 1
    linux_position = next(i for i, ref in enumerate(encoded["refs"])
                          if ref is not None and ref.turn_id == 2 and ref.start == 12)
    assert targets["pointer"][linux_position] == 1
    assert targets["condition_relation"][linux_position] == 3
    assert encoded["condition_positions"] == [list(range(*encoded["ranges"][0][:2]))]


def test_preparing_training_exports_preserves_completed_actual_rollouts(tmp_path, monkeypatch):
    import json
    from training import task_data
    source = tmp_path/"source"
    source.mkdir()
    for name in ("multidoc2dial_doc.json", "multidoc2dial_dial_train.json"):
        (source/name).write_text("{}")
    root = tmp_path/"prepared"
    root.mkdir()
    actual = json.dumps({"task_id": "actual", "conversation_id": "c", "actual_history": []})+"\n"
    (root/"turns.jsonl").write_text(actual)
    monkeypatch.setattr(task_data, "convert", lambda *args, **kwargs: ([{
        "passage_id": "p", "document_id": "d", "version": "v", "text": "Source", "title": "Title",
        "source_uri": "dataset://p"}], [], {}))
    monkeypatch.setattr(task_data, "benchmark_rows", lambda supervision: iter(()))
    manifest = task_data.build_task_data({"training": {"data_dir": str(root), "multidoc_source": str(source)}})
    assert (root/"turns.jsonl").read_text() == actual
    assert manifest["actual_rollout_turns"] == 1


def test_temporal_direction_and_source_turn_pointer_are_supervised_separately():
    text = "Current policy"
    turn = OnlineTurn("s", "previous", 2, "manual", "What about before that?", (), (), "P3")
    window = HistoryWindow(TextRef(1, "user", 0, len(text)), text, len(text))
    memory = RetrievalState(1, (TurnRecord(1, text, "Current answer"),))
    encoded = encode_task(CharacterTokenizer(), turn, (window,), memory)
    prediction = TaskPrediction("followup", version_intent=VersionIntent("as_of", source_turns=(1,), temporal_relation="before"))
    targets = task_targets(encoded, prediction)
    assert targets["temporal_relation"] == 1 and targets["version_pointer"] == 1
