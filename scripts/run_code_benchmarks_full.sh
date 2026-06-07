#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=.

GPU="${GPU:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-results/code_benchmarks_full}"
BENCHMARKS_STR="${BENCHMARKS:-repoqa sweqa cruxeval livecodebench}"
MODELS_STR="${MODELS:-qwen2.5-coder-7b-instruct qwen2.5-coder-7b-instruct-awq qwen2.5-coder-7b-instruct-gptq llama-3.1-8b-instruct llama-3.1-8b-instruct-awq llama-3.1-8b-instruct-gptq}"

REPOQA_BACKEND="${REPOQA_BACKEND:-vllm}"
REPOQA_CMD="${REPOQA_CMD:-python -m repoqa.search_needle_function}"
REPOQA_CODE_CONTEXT_SIZE="${REPOQA_CODE_CONTEXT_SIZE:-16384}"
REPOQA_MAX_NEW_TOKENS="${REPOQA_MAX_NEW_TOKENS:-1024}"
REPOQA_TENSOR_PARALLEL_SIZE="${REPOQA_TENSOR_PARALLEL_SIZE:-1}"
REPOQA_LANGUAGES="${REPOQA_LANGUAGES:-}"
REPOQA_TRUST_REMOTE_CODE="${REPOQA_TRUST_REMOTE_CODE:-1}"
REPOQA_BASE_URL="${REPOQA_BASE_URL:-}"
REPOQA_SYSTEM_MESSAGE="${REPOQA_SYSTEM_MESSAGE:-}"

GEN_BACKEND="${GEN_BACKEND:-vllm}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.85}"
GEN_TENSOR_PARALLEL_SIZE="${GEN_TENSOR_PARALLEL_SIZE:-1}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"

SWEQA_SPLIT="${SWEQA_SPLIT:-oracle}"
SWEQA_MAX_EXAMPLES="${SWEQA_MAX_EXAMPLES:-}"

CRUXEVAL_SPLIT="${CRUXEVAL_SPLIT:-test}"
CRUXEVAL_MAX_EXAMPLES="${CRUXEVAL_MAX_EXAMPLES:-}"

LIVECODEBENCH_SPLIT="${LIVECODEBENCH_SPLIT:-test}"
LIVECODEBENCH_MAX_EXAMPLES="${LIVECODEBENCH_MAX_EXAMPLES:-}"

read -ra MODEL_LIST <<< "$MODELS_STR"
read -ra BENCHMARK_LIST <<< "$BENCHMARKS_STR"

echo "=================================================="
echo "quant_table_repro code benchmarks full runner"
echo "ROOT_DIR                  : $ROOT_DIR"
echo "GPU                       : $GPU"
echo "RESULTS_ROOT              : $RESULTS_ROOT"
echo "BENCHMARKS                : ${BENCHMARK_LIST[*]}"
echo "MODELS                    : ${MODEL_LIST[*]}"
echo "REPOQA_BACKEND            : $REPOQA_BACKEND"
echo "REPOQA_CMD                : $REPOQA_CMD"
echo "REPOQA_BASE_URL           : ${REPOQA_BASE_URL:-<none>}"
echo "GEN_BACKEND               : $GEN_BACKEND"
echo "GEN_BATCH_SIZE            : $GEN_BATCH_SIZE"
echo "GEN_GPU_MEMORY_UTIL       : $GEN_GPU_MEMORY_UTILIZATION"
echo "=================================================="

mkdir -p "$RESULTS_ROOT"

for MODEL in "${MODEL_LIST[@]}"; do
    MODEL_ID="$(python - "$MODEL" <<'PY'
from src.code_model_registry import get_code_model
import sys
print(get_code_model(sys.argv[1])["model_id"])
PY
)"

    echo ""
    echo "=================================================="
    echo "MODEL: $MODEL"
    echo "MODEL_ID: $MODEL_ID"
    echo "=================================================="

    for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
        case "$BENCHMARK" in
            repoqa)
                MODEL_DIR="$RESULTS_ROOT/repoqa/$MODEL"
                mkdir -p "$MODEL_DIR"
                DONE_MARKER="$MODEL_DIR/repoqa_done.txt"
                if [ -f "$DONE_MARKER" ] && [ "${OVERWRITE:-0}" != "1" ]; then
                    echo "[repoqa][$MODEL] skipping existing run marker: $DONE_MARKER"
                    continue
                fi
                EXTRA_ARGS=()
                if [ -n "$REPOQA_LANGUAGES" ]; then
                    EXTRA_ARGS+=(--languages $REPOQA_LANGUAGES)
                fi
                if [ -n "$REPOQA_BASE_URL" ]; then
                    EXTRA_ARGS+=(--base-url "$REPOQA_BASE_URL")
                fi
                if [ -n "$REPOQA_SYSTEM_MESSAGE" ]; then
                    EXTRA_ARGS+=(--system-message "$REPOQA_SYSTEM_MESSAGE")
                fi
                if [ "$REPOQA_TRUST_REMOTE_CODE" = "1" ]; then
                    EXTRA_ARGS+=(--trust-remote-code)
                fi
                echo "[repoqa][$MODEL] running..."
                CUDA_VISIBLE_DEVICES="$GPU" bash -lc \
                    "$REPOQA_CMD \
                    --model \"$MODEL_ID\" \
                    --backend \"$REPOQA_BACKEND\" \
                    --code-context-size \"$REPOQA_CODE_CONTEXT_SIZE\" \
                    --max-new-tokens \"$REPOQA_MAX_NEW_TOKENS\" \
                    --tensor-parallel-size \"$REPOQA_TENSOR_PARALLEL_SIZE\" \
                    --result-dir \"$MODEL_DIR\" \
                    ${EXTRA_ARGS[*]}"
                date -Iseconds > "$DONE_MARKER"
                ;;
            sweqa)
                MODEL_DIR="$RESULTS_ROOT/sweqa/$MODEL"
                mkdir -p "$MODEL_DIR"
                if [ -f "$MODEL_DIR/summary_overall.csv" ] && [ "${OVERWRITE:-0}" != "1" ]; then
                    echo "[sweqa][$MODEL] skipping existing summary"
                    continue
                fi
                EXTRA_ARGS=()
                if [ -n "$GEN_MAX_MODEL_LEN" ]; then
                    EXTRA_ARGS+=(--max-model-len "$GEN_MAX_MODEL_LEN")
                fi
                if [ -n "$SWEQA_MAX_EXAMPLES" ]; then
                    EXTRA_ARGS+=(--max-examples "$SWEQA_MAX_EXAMPLES")
                fi
                if [ "${OVERWRITE:-0}" = "1" ]; then
                    EXTRA_ARGS+=(--overwrite)
                fi
                echo "[sweqa][$MODEL] running..."
                CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_swe_qa_benchmark.py \
                    --model-key "$MODEL" \
                    --output-dir "$MODEL_DIR" \
                    --split "$SWEQA_SPLIT" \
                    --backend "$GEN_BACKEND" \
                    --batch-size "$GEN_BATCH_SIZE" \
                    --max-new-tokens 16 \
                    --temperature "$GEN_TEMPERATURE" \
                    --tensor-parallel-size "$GEN_TENSOR_PARALLEL_SIZE" \
                    --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
                    "${EXTRA_ARGS[@]}"
                ;;
            cruxeval)
                MODEL_DIR="$RESULTS_ROOT/cruxeval/$MODEL"
                mkdir -p "$MODEL_DIR"
                if [ -f "$MODEL_DIR/summary_overall.csv" ] && [ "${OVERWRITE:-0}" != "1" ]; then
                    echo "[cruxeval][$MODEL] skipping existing summary"
                    continue
                fi
                EXTRA_ARGS=()
                if [ -n "$GEN_MAX_MODEL_LEN" ]; then
                    EXTRA_ARGS+=(--max-model-len "$GEN_MAX_MODEL_LEN")
                fi
                if [ -n "$CRUXEVAL_MAX_EXAMPLES" ]; then
                    EXTRA_ARGS+=(--max-examples "$CRUXEVAL_MAX_EXAMPLES")
                fi
                if [ "${OVERWRITE:-0}" = "1" ]; then
                    EXTRA_ARGS+=(--overwrite)
                fi
                echo "[cruxeval][$MODEL] running..."
                CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_cruxeval_benchmark.py \
                    --model-key "$MODEL" \
                    --output-dir "$MODEL_DIR" \
                    --split "$CRUXEVAL_SPLIT" \
                    --backend "$GEN_BACKEND" \
                    --batch-size "$GEN_BATCH_SIZE" \
                    --max-new-tokens 64 \
                    --temperature "$GEN_TEMPERATURE" \
                    --tensor-parallel-size "$GEN_TENSOR_PARALLEL_SIZE" \
                    --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
                    "${EXTRA_ARGS[@]}"
                ;;
            livecodebench)
                MODEL_DIR="$RESULTS_ROOT/livecodebench/$MODEL"
                mkdir -p "$MODEL_DIR"
                if [ -f "$MODEL_DIR/summary_overall.csv" ] && [ "${OVERWRITE:-0}" != "1" ]; then
                    echo "[livecodebench][$MODEL] skipping existing summary"
                    continue
                fi
                EXTRA_ARGS=()
                if [ -n "$GEN_MAX_MODEL_LEN" ]; then
                    EXTRA_ARGS+=(--max-model-len "$GEN_MAX_MODEL_LEN")
                fi
                if [ -n "$LIVECODEBENCH_MAX_EXAMPLES" ]; then
                    EXTRA_ARGS+=(--max-examples "$LIVECODEBENCH_MAX_EXAMPLES")
                fi
                if [ "${OVERWRITE:-0}" = "1" ]; then
                    EXTRA_ARGS+=(--overwrite)
                fi
                echo "[livecodebench][$MODEL] running..."
                CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_livecodebench_execution.py \
                    --model-key "$MODEL" \
                    --output-dir "$MODEL_DIR" \
                    --split "$LIVECODEBENCH_SPLIT" \
                    --backend "$GEN_BACKEND" \
                    --batch-size "$GEN_BATCH_SIZE" \
                    --max-new-tokens 128 \
                    --temperature "$GEN_TEMPERATURE" \
                    --tensor-parallel-size "$GEN_TENSOR_PARALLEL_SIZE" \
                    --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
                    "${EXTRA_ARGS[@]}"
                ;;
            *)
                echo "Unknown benchmark: $BENCHMARK" >&2
                exit 1
                ;;
        esac
    done
done

echo ""
echo "Done."
echo "Results root: $RESULTS_ROOT"
