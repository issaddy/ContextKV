#!/usr/bin/env bash
# Linux/CUDA only. Prepare the source integration in the selected runtime.
set -euo pipefail
: "${CONTEXTKV_MODEL_SCOPE_PATH:?Set to an explicit local Qwen3 model scope JSON}"
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_HICACHE_TMA_TRANSFER="${SGLANG_HICACHE_TMA_TRANSFER:-0}"
HICACHE_ARGS=()
JSON_ARGS=()
if [[ "${CONTEXTKV_JSON_FORMAT:-compact}" == compact ]]; then
  JSON_ARGS=(--constrained-json-disable-any-whitespace)
fi
if [[ "${CONTEXTKV_ENABLE_HICACHE:-1}" == 1 ]]; then
  HICACHE_ARGS=(--enable-hierarchical-cache
    --hicache-size "${CONTEXTKV_HICACHE_GB:-16}" --hicache-write-policy "${CONTEXTKV_WRITE_POLICY:-write_through}"
    --hicache-io-backend "${CONTEXTKV_HICACHE_IO_BACKEND:-kernel}"
    --hicache-mem-layout "${CONTEXTKV_HICACHE_LAYOUT:-layer_first}")
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$PROJECT_ROOT/engine_patches/contextkv_model_scope.py" \
  --scope "$CONTEXTKV_MODEL_SCOPE_PATH" --launch \
  --tp 1 --dtype bfloat16 --kv-cache-dtype bfloat16 --context-length 32768 \
  --max-total-tokens "${CONTEXTKV_GPU_TOKENS:-32768}" \
  --radix-eviction-policy priority \
  --page-size "${CONTEXTKV_PAGE_SIZE:-64}" \
  --enable-metrics ${HICACHE_ARGS[@]+"${HICACHE_ARGS[@]}"} \
  ${JSON_ARGS[@]+"${JSON_ARGS[@]}"} "$@"
