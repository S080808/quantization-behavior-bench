#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=.

HF_BIN="${HF_BIN:-hf}"
LOCAL_DIR_ROOT="${LOCAL_DIR_ROOT:-}"
MODEL_KEYS_STR="${MODEL_KEYS:-}"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
    echo "Could not find '$HF_BIN' in PATH." >&2
    echo "Install huggingface_hub CLI first, for example:" >&2
    echo "  python -m pip install -U huggingface_hub" >&2
    exit 1
fi

if [ -n "$MODEL_KEYS_STR" ]; then
    read -ra MODEL_KEYS <<< "$MODEL_KEYS_STR"
else
    mapfile -t MODEL_KEYS < <(python - <<'PY'
from src.code_model_registry import load_code_models
for key in load_code_models():
    print(key)
PY
)
fi

echo "=================================================="
echo "quant_table_repro code model prefetch"
echo "ROOT_DIR        : $ROOT_DIR"
echo "HF_BIN          : $HF_BIN"
echo "LOCAL_DIR_ROOT  : ${LOCAL_DIR_ROOT:-<hf cache>}"
echo "MODEL_KEYS      : ${MODEL_KEYS[*]}"
echo "=================================================="

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_ID="$(python - "$MODEL_KEY" <<'PY'
from src.code_model_registry import get_code_model
import sys
print(get_code_model(sys.argv[1])["model_id"])
PY
)"
    echo ""
    echo "==> Prefetching $MODEL_KEY"
    echo "    model_id=$MODEL_ID"
    if [ -n "$LOCAL_DIR_ROOT" ]; then
        TARGET_DIR="$LOCAL_DIR_ROOT/$MODEL_KEY"
        mkdir -p "$TARGET_DIR"
        "$HF_BIN" download "$MODEL_ID" --local-dir "$TARGET_DIR"
    else
        "$HF_BIN" download "$MODEL_ID"
    fi
done

echo ""
echo "Done."
