# Quantization Benchmark Reproduction

Minimal generation and evaluation code for studying weight quantization on table and code benchmarks.

The repository contains reproducible pipelines for:

- frozen WikiTableQuestions (WTQ) and synthetic table-understanding tasks (SUC);
- RepoQA repository-level function retrieval;
- SWE-QA multiple-choice reasoning over code;
- CRUXEval exact Python output prediction;
- LiveCodeBench execution output prediction.

## Repository Layout

```text
quant_table_repro/
  configs/             Model manifests
  data/                Frozen inputs and selected table IDs
  scripts/             Generation and evaluation entrypoints
  src/                 Shared pipeline code
  tests/               Prompt-shape sanity tests
```

## Environment

Run commands from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export PYTHONPATH=.
export HF_TOKEN=...
```

Table pipelines support Hugging Face and vLLM backends. `llama-quip` is supported only through the Hugging Face backend.

The main requirements are pinned from the tested Python 3.13.12 environment. RepoQA is the exception: its `tree-sitter-languages` dependency does not support Python 3.13, so run RepoQA in a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-codebench
source .venv-codebench/bin/activate
bash scripts/setup_code_benchmarks.sh
```

## Table Benchmarks

### Prepare Frozen WTQ and SUC Inputs

Create the canonical 300-table WTQ snapshot:

```bash
python scripts/create_fixed_wtq_300.py
```

Generate frozen SUC tasks:

```bash
python scripts/generate_gold_suc.py
```

Select 150 prompt-safe tables:

```bash
python scripts/select_complete_tables.py \
  --model llama \
  --formats html json col_sep prose row_obj \
  --n-tables 150 \
  --output-json data/selected_tables_150.json \
  --diagnostics-json data/selected_tables_150_diagnostics.json
```

### SUC Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_predictions.py \
  --model llama-gptq \
  --snapshot-dir data/fixed_wtq_300 \
  --tasks-dir data/fixed_wtq_300_suc \
  --selected-table-ids-json data/selected_tables_150.json \
  --formats html json col_sep prose row_obj \
  --output-dir results/baseline_150_selected \
  --backend vllm
```

### SUC One-Shot

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_predictions.py \
  --model llama-gptq \
  --snapshot-dir data/fixed_wtq_300 \
  --tasks-dir data/fixed_wtq_300_suc \
  --selected-table-ids-json data/selected_tables_150.json \
  --formats html json col_sep prose row_obj \
  --output-dir results/oneshot_150_selected \
  --backend vllm \
  --one-shot
```

### WTQ Across All Formats

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_wtq_predictions.py \
  --model llama-gptq \
  --snapshot-dir data/fixed_wtq_300 \
  --output-dir results/wtq_300_all_formats \
  --n-tables 300 \
  --formats tsv md prose csv csv_q row_obj col_sep sec_sep json html yaml xml \
  --backend vllm
```

### Judge Table Predictions

`run_judge_many.py` loads the judge once and evaluates one or more prediction files:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_judge_many.py \
  --predictions-csvs \
    results/baseline_150_selected/llama/llama_predictions.csv \
    results/baseline_150_selected/llama-awq/llama-awq_predictions.csv \
  --output-csvs \
    results/baseline_150_selected/llama/judge_raw.csv \
    results/baseline_150_selected/llama-awq/judge_raw.csv \
  --judge-model-id google/gemma-2-9b-it \
  --backend vllm \
  --target-field raw
```

## Code Benchmarks

Configured models are listed in [`configs/code_bench_models.json`](configs/code_bench_models.json):

- Qwen2.5-Coder-7B-Instruct: fp16, AWQ, GPTQ;
- Meta-Llama-3.1-8B-Instruct: fp16, AWQ, GPTQ.

Optionally prefetch all configured models:

```bash
bash scripts/prefetch_code_models.sh
```

### Smoke Test

Run small subsets before starting the full experiment:

```bash
GPU=0 \
MODELS="qwen2.5-coder-7b-instruct" \
BENCHMARKS="sweqa cruxeval livecodebench" \
SWEQA_MAX_EXAMPLES=16 \
CRUXEVAL_MAX_EXAMPLES=16 \
LIVECODEBENCH_MAX_EXAMPLES=16 \
RESULTS_ROOT=results/code_benchmarks_smoke \
bash scripts/run_code_benchmarks_full.sh
```

### Full Reproduction

The default command evaluates all six configured models on RepoQA, SWE-QA, CRUXEval, and LiveCodeBench execution:

```bash
GPU=0 \
RESULTS_ROOT=results/code_benchmarks_full \
bash scripts/run_code_benchmarks_full.sh
```

Completed outputs are resumed or skipped. To replace existing outputs:

```bash
GPU=0 OVERWRITE=1 bash scripts/run_code_benchmarks_full.sh
```

Generated layout:

```text
results/code_benchmarks_full/
  repoqa/<model>/
  sweqa/<model>/
  cruxeval/<model>/
  livecodebench/<model>/
```

Run only selected models or benchmarks:

```bash
GPU=0 \
MODELS="llama-3.1-8b-instruct llama-3.1-8b-instruct-awq llama-3.1-8b-instruct-gptq" \
BENCHMARKS="sweqa cruxeval livecodebench" \
GEN_BACKEND=vllm \
GEN_BATCH_SIZE=32 \
GEN_GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/run_code_benchmarks_full.sh
```

### RepoQA With a vLLM Server

Start the server:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --max-model-len 20480
```

Run RepoQA in another shell:

```bash
MODELS="qwen2.5-coder-7b-instruct" \
BENCHMARKS="repoqa" \
REPOQA_BACKEND=openai \
REPOQA_BASE_URL=http://127.0.0.1:8000/v1 \
bash scripts/run_code_benchmarks_full.sh
```

### Individual Code Benchmark Entrypoints

SWE-QA:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_swe_qa_benchmark.py \
  --model-key qwen2.5-coder-7b-instruct \
  --output-dir results/code_benchmarks_full/sweqa/qwen2.5-coder-7b-instruct \
  --split oracle \
  --backend vllm
```

CRUXEval:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_cruxeval_benchmark.py \
  --model-key qwen2.5-coder-7b-instruct \
  --output-dir results/code_benchmarks_full/cruxeval/qwen2.5-coder-7b-instruct \
  --split test \
  --backend vllm
```

LiveCodeBench execution:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_livecodebench_execution.py \
  --model-key qwen2.5-coder-7b-instruct \
  --output-dir results/code_benchmarks_full/livecodebench/qwen2.5-coder-7b-instruct \
  --split test \
  --backend vllm
```

RepoQA is invoked through its installed Python module by `run_code_benchmarks_full.sh`. Benchmark datasets and generated outputs are not vendored.

## Tests

```bash
pytest -q
```
