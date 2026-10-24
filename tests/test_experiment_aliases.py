"""A frozen B6 is a publication alias, never an extra generation or sample."""
import asyncio
from copy import deepcopy
import json

import pytest

from bench import experiments
from test_experiments import completed_results, spec


def test_alias_keeps_exact_winner_configuration_and_real_repetitions(tmp_path):
    value = spec(tmp_path)
    value.update(protocols=["P1", "P2"], concurrency=[1, 4], repetitions=[1, 2])
    value["tuning"]["quality_eligible"] = {"B3": {"0": False, "1": True}, "full": {"0": True}}
    frozen = experiments.select_development(value, completed_results(value))
    original = deepcopy(frozen)
    jobs = experiments.matrix_jobs(value, split="test", frozen=frozen)
    assert len(jobs) == 16  # Two methods, not three labels; eight workload blocks.
    assert len({job["job_id"] for job in jobs}) == 16
    selected = [job for job in jobs if job["method"] == "B3"]
    assert len(selected) == 8
    assert all(job["method_aliases"] == ["B6"] and job["source_method"] == "B3"
               and job["candidate_index"] == 1
               and job["overrides"]["planner"]["fixed_token_threshold"] == 4000 for job in selected)
    assert all("method_aliases" not in job for job in jobs if job["method"] != "B3")
    assert frozen == original


def test_actual_execution_and_latency_reference_resolve_to_one_canonical_job(tmp_path, monkeypatch):
    value = spec(tmp_path)
    value["reference_method"] = "B6"
    frozen = experiments.select_development(value, completed_results(value))
    calls, comparisons = [], []

    async def execute(_, job, directory):
        calls.append(job["job_id"])
        return {**job, "status": "completed", "raw_output": str(directory / "raw.json"),
                "summary": {"canonical_method": job["method"]}}

    def pair(candidate, reference, **kwargs):
        comparisons.append((candidate["canonical_method"], reference["canonical_method"]))
        return {"status": "CPU_only_contract"}

    monkeypatch.setattr(experiments, "paired_latency", pair)
    output = tmp_path / "test"
    rows = asyncio.run(experiments.run_matrix(value, output, stage="test", frozen=frozen, executor=execute))
    assert len(calls) == len(rows) == 2
    assert comparisons == [("full", "B3")]
    winner = next(row for row in rows if row["method"] == "B3")
    aliases = json.loads((output / "method-aliases.json").read_text())
    assert aliases["B6"]["canonical_method"] == "B3"
    assert aliases["B6"]["job_ids"] == [winner["job_id"]]
    saved = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(saved) == 2 and sum(row.get("method_aliases") == ["B6"] for row in saved) == 1
    pair_row, = json.loads((output / "comparisons.json").read_text())
    assert pair_row["reference"] == winner["job_id"]
    assert not list(output.glob("B6-*"))


def test_failed_canonical_baseline_is_not_retried_or_hidden_by_alias(tmp_path, monkeypatch):
    value = spec(tmp_path)
    value["reference_method"] = "B6"
    frozen = experiments.select_development(value, completed_results(value))
    calls = []

    async def execute(_, job, directory):
        calls.append(job["method"])
        if job["method"] == "B3":
            raise RuntimeError("canonical generation failed")
        return {**job, "status": "completed", "summary": {}}

    output = tmp_path / "test"
    rows = asyncio.run(experiments.run_matrix(value, output, stage="test", frozen=frozen, executor=execute))
    assert calls == ["B3", "full"]
    winner, = [row for row in rows if row.get("method_aliases") == ["B6"]]
    assert winner["status"] == "failed"
    assert json.loads((output / "comparisons.json").read_text()) == []
    assert json.loads((output / "method-aliases.json").read_text())["B6"]["job_ids"] == [winner["job_id"]]


def test_matched_ablation_inherits_configuration_not_baseline_alias(tmp_path):
    value = spec(tmp_path)
    value["methods"].append({"id": "noC", "ablation_of": "B3",
                             "overrides": {"ablation": {"epoch_retirement": False}}})
    frozen = experiments.select_development(value, completed_results(value))
    jobs = experiments.matrix_jobs(value, frozen=frozen, ablations_only=True)
    assert len(jobs) == 1 and jobs[0]["method"] == "noC" and jobs[0]["source_method"] == "B3"
    assert "method_aliases" not in jobs[0]
    assert jobs[0]["overrides"]["ablation"]["epoch_retirement"] is False


def test_separately_declared_B6_cannot_be_confused_with_another_winner(tmp_path):
    value = spec(tmp_path)
    value["methods"][1]["id"] = "B6"
    frozen = experiments.select_development(value, completed_results(value))
    with pytest.raises(ValueError, match="alias conflicts"):
        experiments.matrix_jobs(value, frozen=frozen)
