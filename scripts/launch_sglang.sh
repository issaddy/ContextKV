#!/usr/bin/env bash
# Linux/CUDA only. Run prepare_sglang.py for a full checkout first, install that
# pinned checkout's own dependencies, and activate its isolated environment.
set -euo pipefail
CHECKOUT="${SGLANG_CHECKOUT:-vendor/sglang}"
: "${CONTEXTKV_MODEL_SCOPE_PATH:?Set to an explicit local Qwen3 model scope JSON}"
: "${CONTEXTKV_FINGERPRINT_JSON:?Set the exact HFTemplateAdapter fingerprint JSON}"
if [[ -f "$CHECKOUT/.contextkv-source-only" ]]; then
  echo 'A source-only validation directory cannot run SGLang.' >&2
  exit 2
fi
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTHONPATH="$CHECKOUT/python${PYTHONPATH:+:$PYTHONPATH}"
# Optional host admission uses inherited CONTEXTKV_HOST_ADMISSION=off|fixed|conditioned.
# Off preserves native backup behavior and the previous cache configuration hash.
# HiCache execution remains native; the research quote classifies FULL GPU/host
# ready and pending copies without touching transfer queues or native locks.
HICACHE_ARGS=()
SESSION_ARGS=()
if [[ "${CONTEXTKV_ENABLE_SESSION_RADIX:-1}" == 1 ]]; then
  SESSION_ARGS=(--enable-session-radix-cache)
fi
if [[ "${CONTEXTKV_ENABLE_HICACHE:-1}" == 1 ]]; then
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-host-memory-mode cache
    --hicache-size "${CONTEXTKV_HICACHE_GB:-16}" --hicache-write-policy "${CONTEXTKV_WRITE_POLICY:-write_through}"
    --hicache-io-backend "${CONTEXTKV_HICACHE_IO_BACKEND:-kernel}"
    --hicache-mem-layout "${CONTEXTKV_HICACHE_LAYOUT:-page_first}")
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$PROJECT_ROOT/engine_patches/contextkv_model_scope.py" \
  --scope "$CONTEXTKV_MODEL_SCOPE_PATH" --launch \
  --tp 1 --dtype bfloat16 --kv-cache-dtype bfloat16 --context-length 32768 \
  --page-size "${CONTEXTKV_PAGE_SIZE:-64}" ${SESSION_ARGS[@]+"${SESSION_ARGS[@]}"} \
  --enable-metrics ${HICACHE_ARGS[@]+"${HICACHE_ARGS[@]}"} "$@"
