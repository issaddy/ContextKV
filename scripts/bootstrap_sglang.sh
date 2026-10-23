#!/usr/bin/env bash
# Run from the repository root on Linux/NVIDIA. Proxy settings are inherited;
# credentials must never be added to this script or committed configuration.
set -euo pipefail
CHECKOUT="${SGLANG_CHECKOUT:-vendor/sglang}"
ENV_DIR="${CONTEXTKV_ENGINE_ENV:-.venv-sglang}"
MODEL_DIR="${CONTEXTKV_MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
REVISION=a09a35458c702b33eeacc393d103063234e8bc28
if [[ "$(uname -s)" != Linux ]]; then
  echo 'SGLang runtime installation requires Linux; use prepare_sglang.py --source-only for CPU validation.' >&2
  exit 2
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR" --python 3.12
fi
"$ENV_DIR/bin/python" scripts/prepare_sglang.py --checkout "$CHECKOUT" --report runs/engine-source.json
# The fixed source owns torch/CUDA/kernel requirements. This environment is
# separate from the controller's locked CPU development environment.
SGLANG_BUILD_RUST_EXTS=none uv pip install --python "$ENV_DIR/bin/python" \
  -e "$CHECKOUT/python" -e '.[tokenizer,retrieval]'
mkdir -p runs
uv pip freeze --python "$ENV_DIR/bin/python" > runs/engine-environment.txt
"$ENV_DIR/bin/python" - "$MODEL_DIR" "$REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=sys.argv[2], local_dir=sys.argv[1],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
PY
"$ENV_DIR/bin/python" - "$MODEL_DIR" "$REVISION" <<'PY'
from dataclasses import asdict
import json
from pathlib import Path
import sys
from controller.templates import HuggingFaceTemplate
template = HuggingFaceTemplate(sys.argv[1], sys.argv[2], local_files_only=True)
Path("runs/fingerprint.json").write_text(json.dumps(asdict(template.fingerprint), indent=2)+"\n")
PY
echo "Prepared $ENV_DIR and $MODEL_DIR. Activate the environment, then run scripts/launch_sglang.sh."
