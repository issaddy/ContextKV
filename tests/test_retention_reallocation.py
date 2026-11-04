"""Existing leases and new prefixes compete for the same retention capacity."""
from types import SimpleNamespace
import unittest

from controller.prefix_retention import PrefixForecast, PrefixReadObserver


def forecast(saved, *, horizon=60, control=0):
    return PrefixForecast(horizon, 1, saved, 0, horizon, 0, "scope", "paired",
                          retention_control_ms=control)


async def reallocate(old_curve, new_curve, *, old_pages=("old",), new_pages=("new",),
                     capacity=1, old_event="old-read", new_event="new-read", elapsed=0):
    """Exercise the runtime allocation with measured forecast curves and native owners."""
    now = 60_000_000_000
    calls = []
    curves = {"old": old_curve, "new": new_curve}

    def receipt(owner, pages):
        return dict(owner_id=owner, state="RETAINED", engine_restart_id="engine",
            cache_scope_hash="scope", boundary_tokens=len(pages), page_size=1,
            gpu_page_ids=list(pages), host_page_ids=[],
            pool={"host_capacity_tokens": 0},
            residency=dict(recoverable_prefix_tokens=len(pages),
                inflight_h2d_tokens=0, inflight_d2h_tokens=0))

    old_receipt, new_receipt = receipt("old", old_pages), receipt("new", new_pages)
    old = SimpleNamespace(owner_id="old", session_id="session", attempt_id="old-attempt",
        step_id="old-step", receipt=SimpleNamespace(engine_restart_id="engine"),
        prefix_input_ids=(1,), sources=({"owner": "old"},),
        policy_choice={"reuse_event_key": old_event}, ttl_seconds=60,
        lease_started_ns=now-int(elapsed*1e9))

    async def status(restart):
        return {"prefixes": [old_receipt]}

    async def release(owner_id, *, reason):
        calls.append(("release", owner_id))

    async def observe(**kwargs):
        calls.append(("retain", kwargs["owner_id"], kwargs["ttl_seconds"]))
        return {"owner_id": kwargs["owner_id"], "state": "RETAINED"}

    callback = PrefixReadObserver.__new__(PrefixReadObserver)
    callback.store = SimpleNamespace(prefix_owners=lambda **kwargs: (old,),
        prefix_owner=lambda owner: old if owner == "old" else None,
        prefix_accesses=lambda owner: ())
    callback.runtime = SimpleNamespace(status=status)
    callback.owners = SimpleNamespace(release=release, observe=observe)
    callback.settings = dict(gpu_budget_tokens=capacity, host_budget_tokens=0,
                             lambda_gpu=0, lambda_host=0)
    callback.model = SimpleNamespace(feature_bucket=lambda features: features)
    callback.model_digest = "paired-model"
    callback._features = lambda plan, sources, accesses, now: {"owner": sources[0]["owner"]}
    callback._native_features = lambda inspection: {}
    callback._forecasts = lambda features, scope: curves[features["owner"]]
    result = await callback._allocate_value(None,
                                           {"owner_id": "new", "receipt": SimpleNamespace(attempt_id="new-native-attempt")}, new_receipt,
                                           {"owner": "new"}, now, new_event)
    return result, calls


class RetentionReallocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_more_valuable_new_prefix_replaces_positive_old_lease(self):
        result, calls = await reallocate((forecast(10),), (forecast(100),))
        self.assertEqual(calls, [("release", "old"), ("retain", "new", 60)])
        self.assertEqual(result["state"], "RETAINED")
        self.assertEqual(result["existing_retention_owners"], [])


    async def test_distinct_reads_share_page_capacity(self):
        result, calls = await reallocate((forecast(100),), (forecast(90),),
            old_pages=("root", "old"), new_pages=("root", "new"), capacity=3)
        self.assertEqual(calls, [("retain", "new", 60)])
        self.assertEqual(result["existing_retention_owners"], ["old"])
        choice = result["retention_choices"][0]
        self.assertEqual(choice["marginal_gpu_tokens"], 1)
        self.assertEqual(choice["expected_saved_ms"], 90)

    async def test_same_read_replaces_short_prefix_without_duplicate_benefit(self):
        result, calls = await reallocate((forecast(70),), (forecast(100),),
            old_pages=("root", "source"), new_pages=("root", "source", "tail"),
            capacity=3, old_event="same-read", new_event="same-read")
        self.assertEqual(calls, [("release", "old"), ("retain", "new", 60)])
        self.assertEqual(len(result["retention_choices"]), 1)
        self.assertEqual(result["retention_choices"][0]["expected_saved_ms"], 100)


if __name__ == "__main__":
    unittest.main()
