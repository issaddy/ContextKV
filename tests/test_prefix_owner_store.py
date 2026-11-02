"""Independent prefix intent/recovery ledger, without simulated GPU results."""
import asyncio
from types import SimpleNamespace

import pytest

from controller.prefix_retention import PrefixObserver
from controller.schema import CompletionReceipt, ConfigFingerprint
from controller.session_store import IdempotencyConflict, SessionStore


class RuntimeChannel:
    fingerprint = ConfigFingerprint()
    def __init__(self):
        self.restart_id = "engine-1"
        self.owners, self.calls = {}, []
        self.lose_retain_ack = self.lose_release_ack = False
    async def info(self):
        return SimpleNamespace(engine_restart_id=self.restart_id)
    async def _post(self, path, body):
        self.calls.append((path, body))
        owner = body["owner_id"]
        if path.endswith("/retain"):
            ack = dict(owner_id=owner, state="RETAINED", engine_restart_id=self.restart_id,
                       reference_present=True, expires_at=500, clock="server_monotonic")
            self.owners[owner] = ack
            if self.lose_retain_ack:
                self.lose_retain_ack = False
                raise ConnectionError("retention ACK lost after registration")
            return ack
        if path.endswith("/release"):
            self.owners.pop(owner, None)
            if self.lose_release_ack:
                self.lose_release_ack = False
                raise ConnectionError("release ACK lost after native release")
            return dict(owner_id=owner, state="RELEASED", engine_restart_id=self.restart_id)
        return self.owners.get(owner, dict(owner_id=owner, state="RELEASED", engine_restart_id=self.restart_id))


def observation():
    receipt = CompletionReceipt("r", "native-read", "business-epoch", 1, "engine-1",
        "executed-input-hash", "stop", True, inserted_cache_tokens=4, session_ref_registered=True)
    return dict(owner_id="source-owner", session_id="business", attempt_id="turn", step_id="read:0",
        receipt=receipt, prefix_input_ids=(1, 2, 3, 4),
        sources=({"source_id": "policy", "version": "2025", "start": 0, "end": 80},),
        ttl_seconds=30, policy="fixed_ttl_mechanism", policy_choice={"qualification_only": True})


def test_prefix_intent_and_ack_survive_reopen_and_business_close(tmp_path):
    async def run():
        path = tmp_path/"state.sqlite"
        store = SessionStore(path)
        store.read_snapshot("business")
        runtime = RuntimeChannel()
        await PrefixObserver(store, runtime).observe(**observation())
        store.close_session("business")
        store.close()
        reopened = SessionStore(path)
        record = reopened.prefix_owner("source-owner")
        assert record.stage == "RETAINED" and record.receipt == observation()["receipt"]
        assert record.sources == observation()["sources"] and record.policy == "fixed_ttl_mechanism"
        assert record.policy_choice == {"qualification_only": True} and record.ttl_seconds == 30
        assert [row["event"] for row in reopened.prefix_owner_events("source-owner")] == [
            "retain_requested", "retain_acknowledged"]
        reconciled = await PrefixObserver(reopened, runtime).reconcile("business")
        assert reconciled[0].stage == "RETAINED"
        assert reconciled[0].runtime_receipt["expires_at"] == 500  # Reconcile never renews TTL.
        reopened.close()
    asyncio.run(run())


def test_lost_retain_ack_is_recovered_from_native_owner_without_replaying_read(tmp_path):
    async def run():
        store = SessionStore(tmp_path/"state.sqlite")
        runtime = RuntimeChannel()
        runtime.lose_retain_ack = True
        observer = PrefixObserver(store, runtime)
        with pytest.raises(ConnectionError, match="ACK lost"):
            await observer.observe(**observation())
        assert store.prefix_owner("source-owner").stage == "RETAINING"
        updated = await PrefixObserver(store, runtime).reconcile()
        assert updated[0].stage == "RETAINED"
        assert [path for path, _ in runtime.calls].count("/context/prefix/retain") == 1
        assert updated[0].receipt.attempt_id == "native-read"
        store.close()
    asyncio.run(run())


def test_lost_release_ack_and_engine_restart_are_observed_without_cache_recreation(tmp_path):
    async def run():
        store = SessionStore(tmp_path/"state.sqlite")
        runtime = RuntimeChannel()
        observer = PrefixObserver(store, runtime)
        await observer.observe(**observation())
        runtime.lose_release_ack = True
        with pytest.raises(ConnectionError, match="ACK lost"):
            await observer.release("source-owner", reason="expired_policy")
        assert store.prefix_owner("source-owner").stage == "RELEASING"
        assert (await observer.reconcile())[0].stage == "RELEASED"
        await observer.observe(**observation())
        runtime.restart_id = "engine-2"
        runtime.owners.clear()
        before = len(runtime.calls)
        record = (await PrefixObserver(store, runtime).reconcile())[0]
        assert record.stage == "LOST" and record.last_status["reason"] == "engine_restart"
        assert len(runtime.calls) == before
        assert not store.prefix_owners(active_only=True)
        store.close()
    asyncio.run(run())


def test_owner_cannot_silently_change_source_version_or_causal_path(tmp_path):
    async def run():
        store = SessionStore(tmp_path/"state.sqlite")
        runtime = RuntimeChannel()
        observer = PrefixObserver(store, runtime)
        await observer.observe(**observation())
        before = len(runtime.calls)
        with pytest.raises(IdempotencyConflict, match="another causal source path"):
            await observer.observe(**{**observation(), "sources": ({"version": "2024"},)})
        assert len(runtime.calls) == before
        assert store.prefix_owner("source-owner").sources == observation()["sources"]
        store.close()
    asyncio.run(run())
