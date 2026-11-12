#!/usr/bin/env bash
# Reuse the selected Linux/NVIDIA environment. Never reinstall torch/CUDA here.
set -euo pipefail
: "${CONTEXTKV_MODEL_SCOPE_PATH:?Set to the existing local Qwen3 model scope JSON; bootstrap does not download weights}"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'SGLang serving requires Linux; prepare_sglang.py --source-only verifies the patch on CPU.' >&2
  exit 2
fi
mkdir -p runs
python3 scripts/prepare_sglang.py --report runs/engine-source.json
python3 - "$CONTEXTKV_MODEL_SCOPE_PATH" <<'PY_SCOPE'
from pathlib import Path
import shlex
import sys
from engine_patches.contextkv_model_scope import load_model_scope
model = load_model_scope(sys.argv[1])
settings = {"CONTEXTKV_MODEL_SCOPE_PATH": str(Path(sys.argv[1]).resolve())}
Path("runs/engine-runtime.env").write_text("".join(f"export {name}={shlex.quote(value)}\n" for name, value in settings.items()))
print("Prepared installed SGLang with:", sys.executable)
print("Qwen3 model:", model.model_path)
PY_SCOPE
echo 'Source runs/engine-runtime.env before launch. Installed CUDA/PyTorch packages were not changed.'
