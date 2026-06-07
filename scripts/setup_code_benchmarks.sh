#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

PY_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PY_VERSION" in
    3.10|3.11|3.12)
        ;;
    *)
        echo "Unsupported Python version for RepoQA setup: $PY_VERSION" >&2
        echo "repoqa depends on tree-sitter-languages, which currently does not ship" >&2
        echo "usable wheels for Python $PY_VERSION on this setup." >&2
        echo "" >&2
        echo "Please create a Python 3.12 environment and rerun, for example:" >&2
        echo "  python3.12 -m venv .venv-codebench" >&2
        echo "  source .venv-codebench/bin/activate" >&2
        echo "  PYTHON_BIN=python bash scripts/setup_code_benchmarks.sh" >&2
        exit 1
        ;;
esac

echo "Installing Python dependencies..."
"$PYTHON_BIN" -m pip install --upgrade \
  "tree-sitter==0.21.3" \
  "tree-sitter-languages==1.10.2" \
  "repoqa[vllm]" \
  datasets pandas tqdm transformers accelerate

echo "Setup complete."
