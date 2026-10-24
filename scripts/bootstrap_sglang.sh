#!/usr/bin/env bash
# Run from the repository root on Linux/NVIDIA. Proxy settings are inherited;
# credentials must never be added to this script or committed configuration.
set -euo pipefail
CHECKOUT="${SGLANG_CHECKOUT:-vendor/sglang}"
ENV_DIR="${CONTEXTKV_ENGINE_ENV:-.venv-sglang}"
: "${CONTEXTKV_MODEL_SCOPE_PATH:?Set to the existing local Qwen3 model scope JSON; bootstrap does not download weights}"
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
"$ENV_DIR/bin/python" - "$CONTEXTKV_MODEL_SCOPE_PATH" <<'PY_SCOPE'
from dataclasses import asdict
import json
from pathlib import Path
import sys
from controller.config import ModelConfig, create_model_template
model = ModelConfig(scope_path=sys.argv[1])
template = create_model_template(model)
Path("runs/fingerprint.json").write_text(json.dumps(asdict(template.fingerprint), indent=2)+"\n")
Path("runs/model-scope.json").write_text(json.dumps(model.scope.manifest(), indent=2)+"\n")
print("Verified existing local model:", model.scope.model_path)
PY_SCOPE
echo "Prepared $ENV_DIR using $CONTEXTKV_MODEL_SCOPE_PATH. No model weights downloaded."
echo "Activate the environment, export CONTEXTKV_FINGERPRINT_JSON from runs/fingerprint.json, then launch."
