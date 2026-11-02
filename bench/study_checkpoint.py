"""Capture an executable study configuration and its local source/weight identity."""
from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import time

import yaml

from controller.config import create_model_template, load_config
from controller.engine_client import SGLangEngineClient
from controller.serialization import to_jsonable


def digest_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(root):
    root = Path(root)
    files = sorted({path for name in ("controller", "adapters", "bench", "profiling", "training", "engine_patches", "configs")
        for path in (root/name).rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".patch"}})
    hashes = {str(path.relative_to(root)): digest_file(path) for path in files}
    hashes["pyproject.toml"] = digest_file(root/"pyproject.toml")
    return {"files": hashes, "sha256": sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def weight_inventory(directory):
    root = Path(directory).resolve()
    names = {path for pattern in ("*.safetensors", "*.bin", "*.pt", "*.json") for path in root.rglob(pattern)}
    files = {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest_file(path)}
             for path in sorted(names)}
    return {"path": str(root), "files": files,
        "sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


async def capture(config_path, output, *, assets=(), qualification_records=()):
    config = yaml.safe_load(Path(config_path).read_text())
    run = load_config(config["generation"]["run_config"])
    template = create_model_template(run.model)
    engine = SGLangEngineClient(run.engine.base_url, template.fingerprint,
        timeout=run.engine.timeout_seconds, sampling=run.sampling, stream=False,
        model_scope=run.model.scope.manifest())
    started = time.monotonic()
    try:
        info, metrics = await asyncio.gather(engine.info(), engine.metrics())
        source = source_inventory(Path.cwd())
        weights = await asyncio.to_thread(lambda: [weight_inventory(path) for path in
            dict.fromkeys((run.model.scope.model_path, *assets))])
        records = [{"path": str(Path(path).resolve()), "sha256": digest_file(path)}
                   for path in qualification_records]
        result = {"schema_version": 1, "role": "incumbent_candidate",
            "quality_selection": "pending_comparable_development_results",
            "code": source, "code_version": "sha256:"+source["sha256"],
            "config": config, "run_config": to_jsonable(run), "engine": to_jsonable(info),
            "native_state": metrics, "weights": weights, "qualification_records": records,
            "capture_seconds": time.monotonic()-started,
            "dense5_role": "historical_reference"}
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        return {"source_sha256": source["sha256"], "model_identities": [row["sha256"] for row in weights],
            "record": str(target), "quality_selection": result["quality_selection"]}
    finally:
        await engine.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--qualification-record", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(asyncio.run(capture(args.config, args.output, assets=args.asset,
        qualification_records=args.qualification_record)), ensure_ascii=False))


if __name__ == "__main__":
    main()
