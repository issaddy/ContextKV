"""Observe the remaining study once across shared devices and process groups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from bench.study_resources import integrate_physical, listening_processes, physical_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--gpu-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ports", default="30000,30001,30002,30004,30005,30006")
    parser.add_argument("--process-root", type=int, action="append", default=[])
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    devices = tuple(map(int, args.gpu_devices.split(",")))
    ports = tuple(map(int, args.ports.split(",")))
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    rows = []
    started_wall = datetime.now(timezone.utc).isoformat()
    with (root/"samples.jsonl").open("w") as stream:
        while True:
            processes = (*args.process_root, *listening_processes(ports), os.getpid())
            sample = physical_sample(devices, processes)
            rows.append(sample)
            stream.write(json.dumps(sample)+"\n")
            stream.flush()
            report = {"format": "contextkv_shared_study_observation_v1", "clock": "client_monotonic",
                "started_wall_utc": started_wall,
                "status": "completed" if stopped or time.time() >= deadline else "running",
                "resource_window": {"start": rows[0]["timestamp"], "end": sample["timestamp"]},
                "physical_cost": integrate_physical(rows, rows[0]["timestamp"], sample["timestamp"]),
                "sample_count": len(rows), "physical_samples_path": str((root/"samples.jsonl").resolve()),
                "gpu_devices": devices, "ports": ports, "additional_process_roots": args.process_root,
                "scope": "remaining study from observer start; shared GPUs counted once, declared process groups PSS",
                "overlap_rule": "this physical integral overlaps phase manifests; do not add both",
                "prior_unobserved_intervals": "consult earlier per-job observations; no retrospective zero filling"}
            (root/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
            if report["status"] == "completed":
                break
            time.sleep(args.interval)
    print(json.dumps({"samples": len(rows), "manifest": str(root/"manifest.json")}))


if __name__ == "__main__":
    main()
