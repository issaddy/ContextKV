from controller.context_schema import Condition, EvidenceKey, EvidenceUnit, SourceSpan, TaskState, TextRef, VersionIntent
from training.preference_rollout import condition_version_mismatches


def test_wrong_condition_version_intervention_requires_actual_scope_mismatch():
    def unit(uid, platform, version):
        key = EvidenceKey("train", "policy", uid, version, uid)
        return EvidenceUnit(uid, (SourceSpan(uid, key, 0, 4),), "text",
            applicability=(("PLATFORM", platform),) if platform else (), document_family="policy")

    correct = unit("linux-v2", "Linux", "v2")
    wrong_platform = unit("windows-v2", "Windows", "v2")
    wrong_version = unit("linux-v1", "Linux", "v1")
    background = unit("irrelevant-background", "", "v2")
    state = TaskState("train", "Linux", conditions=(Condition("platform", value_spans=(TextRef(1, "user", 0, 5),),
        relation_type="PLATFORM"),), version_intent=VersionIntent("explicit", version_ids=("v2",)), current_turn_id=1)
    mismatches = condition_version_mismatches(state, (correct, wrong_platform, wrong_version, background),
        lambda ref: state.raw_question[ref.start:ref.end])
    assert {item.unit_id for item, _ in mismatches} == {"windows-v2", "linux-v1"}
    assert {reason["kind"] for _, reasons in mismatches for reason in reasons} == {"condition", "version"}
