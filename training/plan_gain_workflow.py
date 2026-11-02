"""One bounded entry: actual train/dev rollouts, completion labels and gain fit."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit

from training.plan_gain_completion import finalize, main as fit_main
from training.plan_gain_rollout import collect, write


async def workflow(config, settings):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import load_config
    root = Path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/"inputs"
    frozen.mkdir()
    provenance = {}
    for name in ("task_labels", "actual_rollouts", "sources"):
        original = Path(settings[name]).resolve()
        target = frozen/(name+".jsonl")
        shutil.copyfile(original, target)
        provenance[name] = {"original": str(original), "snapshot": str(target.resolve()),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        settings[name] = str(target.resolve())
    write(root/"input-manifest.json", provenance)
    run = load_config(settings["run_config"])
    reader_url = settings.get("reader_base_url") or run.engine.base_url
    ports = {urlsplit(url).port for url in (reader_url, settings["judge_base_url"], config["generation"]["teacher_url"])}
    sampler = StudyResourceSampler(tuple(sorted(set((*settings["online_gpu_devices"], *settings["judge_gpu_devices"])))),
        tuple(sorted(set((*listening_processes(ports), *settings["extra_process_roots"])))))
    await sampler.start()
    started = time.monotonic()
    result = {"status": "running", "cohorts": {}, "source_files": provenance,
        "reader_profile_id": settings["reader_profile_id"], "partition_seed": settings["partition_seed"],
        "train_states": settings["train_states"], "dev_states": settings["dev_states"],
        "model_generation_limit": settings["max_generations"]}
    used = 0
    try:
        train_cap = settings["max_generations"]*settings["train_states"]//(settings["train_states"]+settings["dev_states"])
        for split, count in (("train", settings["train_states"]), ("dev", settings["dev_states"])):
            cap = train_cap if split == "train" else settings["max_generations"]-used
            actual = {**settings, "split": split, "limit_tasks": count,
                "output_dir": str(root/split), "max_generations": cap}
            report = await collect(config, actual)
            used += report["generations"]["actual_calls"]
            completion = finalize(root/split, root/(split+"-supervision"))
            result["cohorts"][split] = {"collection": report, "supervision": completion}
            write(root/"workflow.json", result)
        asset = settings.get("asset") or str(root/"plan-gain.json")
        await asyncio.to_thread(fit_main, ["fit", "--train", str(root/"train-supervision"/"effective-gains.jsonl"),
            "--dev", str(root/"dev-supervision"/"effective-gains.jsonl"), "--output", asset])
        result.update(status="completed", asset=asset, actual_model_generations=used)
    finally:
        import sys
        error = sys.exc_info()[1]
        if error is not None:
            result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        ended = time.monotonic()
        await sampler.stop()
        result.update(elapsed_ms=(ended-started)*1000, physical_cost=integrate_physical(sampler.samples, started, ended),
            physical_observation=sampler.observation(), cost_scope="whole sequential workflow; subjob costs retained, overlapping scope is not added twice")
        write(root/"workflow.json", result)
    return result


def main(argv=None):
    import yaml
    def integers(value):
        return tuple(map(int, value.split(",")))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-config", "run-config", "reader-profile-id", "judge-config", "judge-base-url",
                 "task-labels", "actual-rollouts", "sources", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--reader-base-url")
    parser.add_argument("--asset")
    parser.add_argument("--train-states", type=int, default=8)
    parser.add_argument("--dev-states", type=int, default=4)
    parser.add_argument("--partition-seed", default="plan-gain-native-train-v1")
    parser.add_argument("--dev-fraction", type=float, default=.2)
    parser.add_argument("--read-outputs", type=integers, default=(512, 1024, 2048))
    parser.add_argument("--max-generations", type=int, default=1536)
    parser.add_argument("--online-gpu-devices", type=integers, required=True)
    parser.add_argument("--judge-gpu-devices", type=integers, required=True)
    parser.add_argument("--extra-process-roots", type=integers, default=())
    parser.add_argument("--cache-start", choices=("observed", "cold"), default="observed")
    parser.add_argument("--scope-scan", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if min(args.train_states, args.dev_states) < 1 or args.max_generations < 4 or not 0 < args.dev_fraction < 1:
        raise ValueError("actual train/dev state counts and total generation allocation must be positive")
    config = yaml.safe_load(Path(args.study_config).read_text())
    print(json.dumps(asyncio.run(workflow(config, vars(args))), ensure_ascii=False))


if __name__ == "__main__":
    main()
