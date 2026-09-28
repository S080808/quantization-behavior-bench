# Structured Table Access under Quantization

Reproducible generation and evaluation code for studying how weight quantization affects structured-table access.

The repository contains reproducible pipelines for:

- a reproducible 300-table WikiTableQuestions (WTQ) sample;
- synthetic structural-understanding tasks (SUC) derived from those tables;
- inference across multiple table serializations;
- automatic judging of generated answers.

## Repository Layout

```text
quantization-behavior-bench/
  data/                Reproducible WTQ and SUC inputs
  scripts/             Generation and evaluation entrypoints
  src/                 Shared pipeline code
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

## Table Benchmarks

### Prepare WTQ and SUC Inputs

Create the canonical 300-table WTQ dataset:

```bash
python scripts/prepare_wtq_data.py
```

Generate SUC tasks:

```bash
python scripts/prepare_suc_tasks.py
```

Select 150 prompt-safe tables:

```bash
python scripts/select_wtq_tables.py \
  --model llama \
  --formats html json col_sep prose row_obj \
  --n-tables 150 \
  --output-json data/wtq_selected_150.json \
  --diagnostics-json data/wtq_selected_150_diagnostics.json
```

### SUC Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_suc_predictions.py \
  --model llama-gptq \
  --wtq-dir data/wtq_300 \
  --suc-dir data/suc_300 \
  --selected-table-ids-json data/wtq_selected_150.json \
  --formats html json col_sep prose row_obj \
  --output-dir results/suc_baseline_150 \
  --backend vllm
```

### SUC One-Shot

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_suc_predictions.py \
  --model llama-gptq \
  --wtq-dir data/wtq_300 \
  --suc-dir data/suc_300 \
  --selected-table-ids-json data/wtq_selected_150.json \
  --formats html json col_sep prose row_obj \
  --output-dir results/suc_oneshot_150 \
  --backend vllm \
  --one-shot
```

### Activation Patching

Patch FP16 residual states into the AWQ or GPTQ model at format-token positions
in layers 24--31 during prefill. The random control uses 20 deterministic,
budget-matched token masks and writes one prediction file per mask.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_activation_patching.py \
  --quant-model llama-awq \
  --wtq-dir data/wtq_300 \
  --suc-dir data/suc_300 \
  --selected-table-ids-json data/wtq_selected_150.json \
  --formats col_sep html json \
  --layers 24 25 26 27 28 29 30 31 \
  --patch-groups format_late random10pct \
  --random-masks 20 \
  --output-dir results/activation_patching
```

Run the command once with `--quant-model llama-awq` and once with
`--quant-model llama-gptq`. Existing rows are reused, so interrupted runs can
be resumed with the same command.

### WTQ Across All Formats

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_wtq_predictions.py \
  --model llama-gptq \
  --wtq-dir data/wtq_300 \
  --output-dir results/wtq_300_all_formats \
  --n-tables 300 \
  --formats tsv md prose csv csv_q row_obj col_sep sec_sep json html yaml xml \
  --backend vllm
```

### Judge Table Predictions

`judge_predictions.py` loads the judge once and evaluates one or more prediction files:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/judge_predictions.py \
  --predictions-csvs \
    results/suc_baseline_150/llama/llama_predictions.csv \
    results/suc_baseline_150/llama-awq/llama-awq_predictions.csv \
  --output-csvs \
    results/suc_baseline_150/llama/judge_raw.csv \
    results/suc_baseline_150/llama-awq/judge_raw.csv \
  --judge-model-id google/gemma-2-9b-it \
  --backend vllm \
  --target-field raw
```
