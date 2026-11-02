"""Import checks use explicit run/score fixtures, not measured experiments."""
from copy import deepcopy

import pytest

from controller.serialization import to_jsonable
from training.collect_plan_gains import collect_pair
from test_plan_gain import observed_pair


def packet_and_rows():
    plan, state, snapshot, direct, candidate = observed_pair()
    def row(branch):
        steps = ([{"step_id": "read:0", "task": {"kind": "READ", "output_budget": 64},
                   "receipt": branch["native_receipts"][0]}] if branch["read_count"] else [])
        return {**branch, "status": "completed", "frozen_state_key": branch["state_key"],
            "metrics": {"Q": branch["quality"]*100},
            "scoring": {"status": "scored", "identity": branch["scorer_identity"]},
            "executed_output_budgets": {**({"read:0": 64} if steps else {}), "ANSWER": 64},
            "attempt": {"plan": {"candidate": {"context_plan_id": branch["plan_id"],
                "output_budget": 64, "fingerprint": {"model_revision": branch["reader_checkpoint"]}}},
                        "result": {"receipt": branch["native_receipts"][0]}},
            "steps": steps, "response": {"read_count": branch["read_count"]}}
    packet = {"plan_candidate": to_jsonable(plan), "working_state": to_jsonable(state),
        "snapshot": to_jsonable(snapshot), "metric": "Q", "normalization": {
            "scale": .01, "offset": 0, "protocol_source": "explicit fixture 0-100 scale"},
        "scorer_identity": direct["scorer_identity"], "comparison_budget": direct["comparison_budget"]}
    return packet, row(direct), row(candidate)


def test_import_retains_raw_scores_receipts_and_exact_actual_answer_delta():
    packet, direct, candidate = packet_and_rows()
    record = collect_pair(packet, direct, candidate)
    assert record["delta_q"] == pytest.approx(.6)
    assert record["counterfactual_run"]["raw_metric"]["value"] == 80
    assert record["counterfactual_run"]["source_run"] == candidate
    assert record["direct_run"]["native_receipts"] == [direct["attempt"]["result"]["receipt"]]


@pytest.mark.parametrize("field,value", (("frozen_state_key", "not-the-observed-state"),
    ("reader_checkpoint", "another-reader"), ("split", "test"), ("plan_id", "unexecuted-plan")))
def test_import_cannot_relabel_state_reader_split_or_executed_plan(field, value):
    packet, direct, candidate = packet_and_rows()
    candidate = deepcopy(candidate)
    candidate[field] = value
    with pytest.raises(ValueError):
        collect_pair(packet, direct, candidate)
