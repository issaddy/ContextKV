"""Retrieval uses the committed dialogue snapshot without mutating it."""
from dataclasses import replace
import pytest
from controller.schema import OnlineMessage, OnlineTurn
from adapters.conversation import PriorContext, snapshot_context, controller_context_provider
from controller.evidence import make_evidence
from bench.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore


def fixture():
    history = (OnlineMessage("user", "Does IBM offer document databases?"),
               OnlineMessage("assistant", "IBM Cloudant supports documents; the limit is 50 MB. PhantomStore has 90 MB."),
               OnlineMessage("user", "Can I store an image or pdf with a document?"),
               OnlineMessage("assistant", "Use Cloudant. The invented limit is 70 MB. `FakeSDK` is a code name."))
    turn = OnlineTurn("s", "task", 3, "cloud", "Is there a file size limit? ", history, (), "P3")
    context = PriorContext(snapshot_version=2)
    return turn, context

def test_dialogue_only_projection_does_not_read_source_inventory():
    turn,_=fixture()
    class Snapshot:
        version=2
        raw_history=tuple(TurnRecord(i//2+1,turn.history[i].text,turn.history[i+1].text)
                          for i in range(0,len(turn.history),2))
        @property
        def archive(self):
            raise AssertionError('Dialogue interpretation does not use archived source names')
    result=snapshot_context(turn,Snapshot())
    assert result.snapshot_version==2 and result.interpretations==(None,None)

@pytest.mark.asyncio
async def test_controller_provider_reads_real_committed_state_without_creating_session(tmp_path):
    store = SessionStore(tmp_path / "state.sqlite3")
    service = Controller(store, MockEngine(response_text="Cloudant stores JSON."),
                         knobs=PlannerKnobs(strategy="always_rebuild"))
    provider = controller_context_provider(store, "run")
    first = OnlineTurn("s", "t1", 1, "cloud", "q1", (), (), "P3")
    assert provider(first) == PriorContext()
    assert store.db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    source = make_evidence("Cloudant stores JSON.", "cloudant")
    answer = await service.serve_turn("run:s", "q1", (source,))
    next_turn = replace(first, task_id="t2", turn=2, query="And attachments?", history=(
        OnlineMessage("user", "q1"), OnlineMessage("assistant", answer.assistant_text)))
    before = store.get_snapshot("run:s")
    context = provider(next_turn)
    assert context == PriorContext(before.version, (None,))
    assert store.get_snapshot("run:s") == before
    await service.aclose()
