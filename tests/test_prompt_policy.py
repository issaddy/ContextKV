from dataclasses import replace

import pytest

from controller.candidates import CandidatePolicy
from controller.cli import main
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_prompt_change_rebuilds_even_under_append_policy(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    controller = Controller(store, MockEngine(), knobs=PlannerKnobs(strategy="always_append"))
    evidence = (make_evidence("Original evidence with conditions.", "a"),)
    try:
        await controller.serve_turn("s", "First question", evidence)
        old = store.read_snapshot("s")
        assert old.execution.messages[0].content == SYSTEM_PROMPT
        controller.policy = replace(controller.policy, system_prompt="Use exact evidence identifiers.\n")
        result = await controller.serve_turn("s", "Second question", evidence)
        current = store.read_snapshot("s")
        assert result.action == "REBUILD" and result.epoch_no > old.current_epoch.epoch_no
        assert current.execution.messages[0].content == controller.policy.system_prompt
        assert current.raw_history[:-1] == old.raw_history
        assert current.reference_map == old.reference_map
        assert current.execution.visible_blocks == {evidence[0].key}
        result = await controller.serve_turn("s", "Third question", evidence)
        assert result.action == "APPEND"
        observed = tuple(controller.cost_model._observed_output_lengths)
        assert len(observed) == 3
        assert len(controller._postplan_control_ms["REBUILD"]) == 2
        assert len(controller._postplan_control_ms["APPEND"]) == 1
        await controller.serve_turn("s", "Third question", evidence, request_id=result.request_id)
        assert tuple(controller.cost_model._observed_output_lengths) == observed
    finally:
        await controller.aclose()
        store.close()


@pytest.mark.parametrize("use_file", (False, True))
def test_cli_prompt_override_retains_exact_content_and_config_provenance(tmp_path, monkeypatch, use_file):
    from controller import cli
    text = "Custom prompt, with exact line endings.\r\nSecond line.\r\n"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(text)
    captured = []

    async def demo(config):
        captured.append(config)
    monkeypatch.setattr(cli, "demo", demo)
    option = ["--system-prompt-file", str(prompt)] if use_file else ["--system-prompt", text]
    main([*option, "demo"])
    assert captured[0].context.system_prompt == text
    assert captured[0].config_hash != replace(captured[0], context=CandidatePolicy()).config_hash
