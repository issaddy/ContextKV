from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from adapters.mtrag import load_rows, online_turn
from bench.metrics import TurnTiming, occupancy_integral, summarize_timings, transfer_totals
from bench.quality import export_evaluation, paired_cluster_bootstrap, retrieval_quality
from profiling.common import Measurement, summarize_measurements
from profiling.prefill import calibration_grid
from profiling.transfers import from_completed_events


class MetricTests(unittest.TestCase):
    def test_ttft_tpot_commit_tail_and_slo_window(self):
        sample = TurnTiming("s", "1", "success", 3, 0, 1, 3, 4, 5, token_observation="token_events", truncated=False)
        self.assertEqual(sample.ttft, 1)
        self.assertEqual(sample.tpot, 1)
        self.assertEqual(sample.turn_seconds, 5)
        single = TurnTiming("s", "2", "success", 1, 2, 3, 3, 4, 4, truncated=False)
        zero = TurnTiming("s", "3", "success", 0, 2, None, None, 4, 4)
        failed = TurnTiming("s", "4", "failed", 0, 4)
        report = summarize_timings([sample, single, zero, failed], window_start=0, window_end=5,
                                   ttft_slo=1, tpot_slo=1, response_mode="streaming")
        self.assertEqual(report["completed_in_window"], 2)
        self.assertEqual(report["latency_slo_goodput_requests_per_second"], .2)
        self.assertEqual(report["failure_fraction"], .25)
        self.assertEqual(report["undefined_tpot_count"], 2)

    def test_cluster_bootstrap_pairs_whole_sessions_and_no_significance_shortcut(self):
        reference = {"a": [.8, .8], "b": [.8], "c": [.8, .8, .8]}
        equal = paired_cluster_bootstrap(reference, reference, samples=100)
        self.assertEqual(equal["decision"], "noninferior")
        lower = {sid: [v - .05 for v in values] for sid, values in reference.items()}
        self.assertEqual(paired_cluster_bootstrap(lower, reference, samples=100)["decision"], "inferior")
        uncertain = {"a": [.7, .7], "b": [.9], "c": [.8, .8, .8]}
        self.assertEqual(paired_cluster_bootstrap(uncertain, reference, samples=100)["decision"], "inconclusive")
        self.assertEqual(paired_cluster_bootstrap({"a": [1]}, {"a": [1]})["decision"], "inconclusive")
        with self.assertRaises(ValueError):
            paired_cluster_bootstrap({"a": [1]}, {"b": [1]})

    def test_missing_qrels_explicit_not_zero(self):
        quality = retrieval_quality({"labelled": ["a", "b"], "missing": ["a"]}, {"labelled": {"a": 1}})
        self.assertEqual(quality["mean_recall"], 1)
        self.assertEqual(quality["mean_ndcg"], 1)
        self.assertEqual(quality["missing_qrels_task_ids"], ["missing"])
        self.assertEqual(quality["evaluated_tasks"], 1)

    def test_official_export_does_not_truncate_audit(self):
        row = load_rows(Path(__file__).parent / "fixtures/mtrag_small.jsonl")[0]
        turn = online_turn(row)
        with tempfile.TemporaryDirectory() as directory:
            official, audit = Path(directory) / "official.jsonl", Path(directory) / "audit.jsonl"
            export_evaluation(turn, "answer", official_path=official, audit_path=audit,
                              visible_evidence_manifest=[{"id": str(i)} for i in range(12)],
                              audit_evidence=[{"text": f"full original {i}"} for i in range(12)], input_ids=[1], output_ids=[2])
            self.assertEqual(len(json.loads(official.read_text())["contexts"]), 5)
            self.assertEqual(len(json.loads(audit.read_text())["audit_evidence"]), 12)
            self.assertNotIn("targets", json.loads(official.read_text()))

    def test_slots_and_transfer_completion_are_counted_once(self):
        report = occupancy_integral([{"timestamp": 0, "gpu_slots": 2, "host_slots": 3},
                                     {"timestamp": 2, "gpu_slots": 1, "host_slots": 3}], end_timestamp=3)
        self.assertEqual(report["gpu_slot_seconds"], 5)
        event = {"transfer_id": "a", "direction": "H2D", "bytes": 1024, "status": "completed"}
        self.assertEqual(transfer_totals([event, event])["completed_bytes"]["H2D"], 1024)

    def test_measurements_refuse_synthetic_hardware(self):
        sample = Measurement("prefill", 1024, 0, 128, 1, 0, 10, "mock")
        with self.assertRaises(ValueError):
            summarize_measurements([sample])
        self.assertEqual(summarize_measurements([sample], allow_mock=True)[0]["source"], "mock")
        self.assertTrue(all(c.prompt_tokens + c.output_tokens <= 32768 for c in calibration_grid()))
        with self.assertRaises(ValueError):
            from_completed_events([{"status":"completed", "transfer_id":"x", "clock":"wall", "direction":"H2D"}])


if __name__ == "__main__":
    unittest.main()
