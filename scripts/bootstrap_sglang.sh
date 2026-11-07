#!/usr/bin/env bash
# Reuse the selected Linux/NVIDIA environment. Never reinstall torch/CUDA here.
set -euo pipefail
CHECKOUT="${SGLANG_CHECKOUT:-vendor/sglang-0.5.18}"
ENGINE_PYTHON="${CONTEXTKV_ENGINE_PYTHON:-python3}"
: "${CONTEXTKV_MODEL_SCOPE_PATH:?Set to the existing local Qwen3 model scope JSON; bootstrap does not download weights}"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'SGLang serving requires Linux; prepare_sglang.py --source-only verifies the patch on CPU.' >&2
  exit 2
fi
mkdir -p runs
"$ENGINE_PYTHON" scripts/prepare_sglang.py --checkout "$CHECKOUT" --report runs/engine-source.json
# This source overlay imports its CUDA kernels from the existing environment.
# No editable install, dependency resolver, wheel replacement, or Rust build.
export PYTHONPATH="$(realpath "$CHECKOUT/python"):$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
"$ENGINE_PYTHON" - "$CONTEXTKV_MODEL_SCOPE_PATH" <<'PY_SCOPE'
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
from controller.config import ModelConfig, create_model_template
model = ModelConfig(scope_path=sys.argv[1])
template = create_model_template(model)
fingerprint = json.dumps(asdict(template.fingerprint))
Path("runs/fingerprint.json").write_text(json.dumps(asdict(template.fingerprint), indent=2)+"\n")
Path("runs/model-scope.json").write_text(json.dumps(model.scope.manifest(), indent=2)+"\n")
installed = sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
                   if dist.metadata["Name"])
Path("runs/engine-environment.txt").write_text("".join(f"{name}=={version}\n" for name, version in installed))
settings = {"PYTHONPATH": os.environ["PYTHONPATH"], "CONTEXTKV_ENGINE_PYTHON": sys.executable,
            "CONTEXTKV_MODEL_SCOPE_PATH": str(Path(sys.argv[1]).resolve()),
            "CONTEXTKV_FINGERPRINT_JSON": fingerprint,
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python"}
Path("runs/engine-runtime.env").write_text("".join(f"export {name}={shlex.quote(value)}\n" for name, value in settings.items()))
print("Prepared source integration with existing interpreter:", sys.executable)
print("Verified local Qwen3 model:", model.scope.model_path)
PY_SCOPE
echo 'Source runs/engine-runtime.env before launch. Installed CUDA/PyTorch packages were not changed.'
