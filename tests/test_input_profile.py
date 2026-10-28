from copy import deepcopy

from bench.input_profile import profile_input
from controller.candidates import CandidatePolicy
from controller.templates import ByteTemplate


def row():
    return {"conversation_id": "case", "task_id": "selected-turn", "turn": 2, "Collection": "manual",
            "input": [{"speaker": "user", "text": "Which version?"},
                      {"speaker": "assistant", "text": "Version 2."},
                      {"speaker": "user", "text": "What is its exception?"}],
            "contexts": [{"document_id": "first", "text": "Original version 2 rule.\n"},
                         {"document_id": "second", "text": "Original exception at the end."}],
            "targets": [{"text": "PRIVATE GOLD ANSWER"}]}


def profile(value, *, window=32768, gpu=32768, reserve=100):
    return profile_input(value, corpus_id="fixture", snapshot_id="fixture-v1",
                         policy=CandidatePolicy(max_total_model_tokens=window, max_new_tokens=reserve),
                         template=ByteTemplate(), gpu_capacity_tokens=gpu)


def test_budget_accounts_for_full_history_sources_and_output_without_using_labels():
    original = row()
    before = deepcopy(original)
    result = profile(original)
    assert result["eligible"] and result["given_sources"] == result["required_sources"] == 2
    assert result["input_plus_output_tokens"] == result["input_tokens"] + 100
    assert original == before
    changed_gold = deepcopy(original)
    changed_gold["targets"] = [{"text": "ANOTHER PRIVATE GOLD ANSWER"}]
    assert profile(changed_gold) == result
    changed_history = deepcopy(original)
    changed_history["input"][1]["text"] += " Distinct actual context."
    assert profile(changed_history)["input_hash"] != result["input_hash"]
    changed_source = deepcopy(original)
    changed_source["contexts"][1]["text"] += " Another limiting condition."
    assert profile(changed_source)["input_tokens"] > result["input_tokens"]


def test_model_and_gpu_boundaries_retain_the_same_untruncated_input():
    original = row()
    result = profile(original)
    exact = result["input_plus_output_tokens"]
    assert profile(original, window=exact, gpu=exact)["eligible"]
    model_excluded = profile(original, window=exact-1)
    pool_excluded = profile(original, gpu=exact-1)
    assert not model_excluded["eligible"] and not pool_excluded["eligible"]
    assert model_excluded["reason"] != pool_excluded["reason"] == "input_plus_output_exceeds_gpu_pool"
    assert model_excluded["input_hash"] == pool_excluded["input_hash"] == result["input_hash"]
    assert model_excluded["required_sources"] == pool_excluded["required_sources"] == 2
