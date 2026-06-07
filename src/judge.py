from __future__ import annotations

import gc
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


CURRENT_JUDGE_PROMPT = """You are evaluating a table understanding answer.

Task: {task}
Question: {question}
Gold answer: {gold}
Model prediction: {prediction}

Rubric:
1. content_correct=true only if the prediction's actual answer content is fully correct and contains no answer-like additions.
2. If the prediction adds extra answer-like content, extra row values, extra column values, repeated values, wrong labels, wrong bullets, wrong numbering, wrong continuations, field names, row/column names, templates, or values from other rows/columns, set content_correct=false.
3. Use overinclusive=true when the prediction contains extra answer-like content beyond the gold answer. In this benchmark, overinclusive=true almost always implies content_correct=false.
4. A short conversational preface or suffix such as "Sure" or "That's a great question" may be treated as instruction_following_error=true without changing content correctness, but only if the answer content itself is otherwise exactly correct and the extra text is purely conversational. If the extra text introduces labels, numbering, bullets, field names, row names, column names, explanations, quotes around the answer slot, or any answer-like structure, set content_correct=false.
5. Never use "the values are correct but the format is wrong" as a reason to keep content_correct=true when the format change adds answer-like structure. For this benchmark, answer-like wrappers contaminate the answer content.
6. For list-style tasks like row_retrieval and col_retrieval, the answer must contain exactly the requested items and no extra items or scaffolding. Missing items, repeated items, extra blank filler, labels, bullets, numbering, field names, or extra items from other rows/columns mean content_correct=false.
7. For cell_lookup, the answer must be exactly the cell value and nothing else. Extra labels like "ColumnName:", "ColumnName =", or "The answer is ..." mean content_correct=false.
8. For reverse_lookup, the answer must match the requested template exactly in substance: row number and column name only. Quotes around the slot value, extra punctuation, or wrapper phrases like "Row X, Column ..." when they do not match the requested template should be treated as content_correct=false.
9. For size_detection, the answer must match the requested template exactly. Variants like "N rows: ..." or "M columns: ..." are not fully correct.
10. If overinclusive=true because of answer-like additions, content_correct must be false.
11. content_partial=true only when the answer is partially correct but not fully correct.

Return valid JSON with exactly these keys:
- content_correct: true/false
- content_partial: true/false
- format_correct: true/false
- overinclusive: true/false
- instruction_following_error: true/false
- score: number between 0 and 1
- reason: short explanation
"""

SUC_V1_JUDGE_PROMPT = """You are grading a table structural-understanding benchmark.

You will see the task type, the question, the gold answer, and a model prediction.
Do not solve the task from scratch. Grade only whether the prediction matches the gold answer.

Return JSON only with these fields:
{{
  "content_correct": true/false,
  "content_partial": true/false,
  "format_correct": true/false,
  "overinclusive": true/false,
  "instruction_following_error": true/false,
  "score": 0, 0.5, or 1,
  "reason": "short reason"
}}

Guidelines:
- content_correct=true only if the required answer values/coordinates/counts are all present and correct.
- content_partial=true if some but not all required list values are present.
- format_correct=false if the answer ignores the requested format, includes explanations, table markup, copied rows, or extra unrelated values.
- overinclusive=true if the prediction includes many extra table values/rows even when the gold appears inside it.
- For numeric gold values, do not accept accidental substring matches inside larger numbers.

Task: {task}
Question: {question}
Gold answer:
{gold}

Model prediction:
{prediction}
"""

JUDGE_PROMPTS = {
    "current": CURRENT_JUDGE_PROMPT,
    "suc_v1": SUC_V1_JUDGE_PROMPT,
}

BOOL_COLUMNS = [
    "content_correct",
    "content_partial",
    "format_correct",
    "overinclusive",
    "instruction_following_error",
]


def build_judge_prompt(
    *,
    task: str,
    question: str,
    gold: str,
    prediction: str,
    prompt_version: str = "current",
) -> str:
    if prompt_version not in JUDGE_PROMPTS:
        raise ValueError(f"Unsupported judge prompt version: {prompt_version}")
    return JUDGE_PROMPTS[prompt_version].format(
        task=task,
        question=question,
        gold=gold,
        prediction=prediction,
    )


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return _log


def _parse_bool(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return pd.NA


def parse_judge_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {
        "content_correct": None,
        "content_partial": None,
        "format_correct": None,
        "overinclusive": None,
        "instruction_following_error": None,
        "score": None,
        "reason": text[:500],
    }


def render_judge_prompt_text(
    tokenizer,
    *,
    task: str,
    question: str,
    gold: str,
    prediction: str,
    prompt_version: str = "current",
) -> str:
    prompt = build_judge_prompt(
        task=task,
        question=question,
        gold=gold,
        prediction=prediction,
        prompt_version=prompt_version,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_judge_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in BOOL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(_parse_bool).astype("boolean")
    df["relaxed_accuracy"] = (df["content_correct"] == True).astype("Float64")
    df["accuracy_clean"] = (
        (df["content_correct"] == True)
        & (df["overinclusive"] == False)
    ).astype("Float64")
    df["strict_accuracy"] = (
        (df["content_correct"] == True)
        & (df["overinclusive"] == False)
        & (df["instruction_following_error"] == False)
    ).astype("Float64")
    df["format_correct_rate"] = (df["format_correct"] == True).astype("Float64")
    df["overinclusive_rate"] = (df["overinclusive"] == True).astype("Float64")
    df["instruction_following_error_rate"] = (
        df["instruction_following_error"] == True
    ).astype("Float64")
    return df


def summarize_judge_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_fmt_long = (
        df.groupby(["task", "fmt"], dropna=False)
        .agg(
            n=("table_id", "size"),
            relaxed_accuracy=("relaxed_accuracy", "mean"),
            strict_accuracy=("strict_accuracy", "mean"),
            format_correct_rate=("format_correct_rate", "mean"),
            overinclusive_rate=("overinclusive_rate", "mean"),
            instruction_following_error_rate=("instruction_following_error_rate", "mean"),
        )
        .reset_index()
    )

    task_fmt = task_fmt_long.pivot(
        index="task",
        columns="fmt",
        values=[
            "n",
            "relaxed_accuracy",
            "strict_accuracy",
            "format_correct_rate",
            "overinclusive_rate",
            "instruction_following_error_rate",
        ],
    )
    task_fmt.columns = [f"{fmt}_{metric}" for metric, fmt in task_fmt.columns]
    task_fmt = task_fmt.reset_index()

    fmt_only = (
        df.groupby(["fmt"], dropna=False)
        .agg(
            n=("table_id", "size"),
            relaxed_accuracy=("relaxed_accuracy", "mean"),
            strict_accuracy=("strict_accuracy", "mean"),
            format_correct_rate=("format_correct_rate", "mean"),
            overinclusive_rate=("overinclusive_rate", "mean"),
            instruction_following_error_rate=("instruction_following_error_rate", "mean"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "n": len(df),
                "relaxed_accuracy": df["relaxed_accuracy"].mean(),
                "strict_accuracy": df["strict_accuracy"].mean(),
                "format_correct_rate": df["format_correct_rate"].mean(),
                "overinclusive_rate": df["overinclusive_rate"].mean(),
                "instruction_following_error_rate": df["instruction_following_error_rate"].mean(),
            }
        ]
    )

    for frame in (fmt_only, overall):
        for column in [
            "relaxed_accuracy",
            "strict_accuracy",
            "format_correct_rate",
            "overinclusive_rate",
            "instruction_following_error_rate",
        ]:
            frame[column] = frame[column].round(4)

    for column in task_fmt.columns:
        if column == "task":
            continue
        if column.endswith("_n"):
            continue
        task_fmt[column] = task_fmt[column].round(4)

    return task_fmt, fmt_only, overall


def write_judge_summaries(
    judge_csv: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    summary_prefix: str | None = None,
) -> dict[str, Path]:
    judge_path = Path(judge_csv)
    out_dir = Path(output_dir) if output_dir else judge_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary_prefix is None:
        summary_prefix = ""
    elif summary_prefix and not summary_prefix.endswith("_"):
        summary_prefix = summary_prefix + "_"

    outputs = {
        "task_fmt": out_dir / f"{summary_prefix}judge_accuracy_by_task_fmt.csv",
        "fmt": out_dir / f"{summary_prefix}judge_accuracy_by_fmt.csv",
        "overall": out_dir / f"{summary_prefix}judge_accuracy_overall.csv",
    }
    if not overwrite and all(path.exists() for path in outputs.values()):
        return outputs

    df = load_judge_results(judge_path)
    task_fmt, fmt_only, overall = summarize_judge_results(df)
    task_fmt.to_csv(outputs["task_fmt"], index=False)
    fmt_only.to_csv(outputs["fmt"], index=False)
    overall.to_csv(outputs["overall"], index=False)
    return outputs


def load_judge(model_id: str, hf_token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


def load_vllm_judge(
    model_id: str,
    hf_token: str | None,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
):
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if hf_token:
        llm_kwargs["hf_token"] = hf_token
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    model = LLM(**llm_kwargs)
    return model, tokenizer


@torch.no_grad()
def judge_one(
    model,
    tokenizer,
    *,
    task: str,
    question: str,
    gold: str,
    prediction: str,
    prompt_version: str = "current",
) -> dict:
    text = render_judge_prompt_text(
        tokenizer,
        task=task,
        question=question,
        gold=gold,
        prediction=prediction,
        prompt_version=prompt_version,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    result = parse_judge_json(raw)
    result["raw_judge"] = raw
    return result


def judge_batch_vllm(
    model,
    tokenizer,
    rows: list[dict],
    *,
    max_new_tokens: int,
    temperature: float,
    prompt_version: str = "current",
) -> list[dict]:
    from vllm import SamplingParams

    prompts = [
        render_judge_prompt_text(
            tokenizer,
            task=row["task"],
            question=row["question"],
            gold=row["gold"],
            prediction=row["prediction"],
            prompt_version=prompt_version,
        )
        for row in rows
    ]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    results: list[dict] = []
    for output in outputs:
        raw = output.outputs[0].text.strip() if output.outputs else ""
        parsed = parse_judge_json(raw)
        parsed["raw_judge"] = raw
        results.append(parsed)
    return results


def judge_batch_vllm_server(
    rows: list[dict],
    *,
    model_id: str,
    server_base_url: str,
    max_new_tokens: int,
    temperature: float,
    prompt_version: str = "current",
) -> list[dict]:
    base = server_base_url.rstrip("/")
    max_workers = min(len(rows), 16) or 1

    def _judge_row(row: dict) -> dict:
        prompt = build_judge_prompt(
            task=row["task"],
            question=row["question"],
            gold=row["gold"],
            prediction=row["prediction"],
            prompt_version=prompt_version,
        )
        payload = json.dumps(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            parsed = parse_judge_json(raw)
            parsed["raw_judge"] = raw
            return parsed
        except Exception as exc:
            raw = str(exc)
            parsed = parse_judge_json(raw)
            parsed["raw_judge"] = raw
            return parsed
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        parsed = parse_judge_json(content)
        parsed["raw_judge"] = content
        return parsed

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_judge_row, rows))


def _load_pending_judge_rows(
    *,
    predictions_csv: str | Path,
    output_csv: str | Path,
    target_field: str,
    log,
) -> tuple[pd.DataFrame, Path, list[dict], list[pd.Series]]:
    df = pd.read_csv(predictions_csv)
    output_path = Path(output_csv)
    rows_out: list[dict] = []
    done_keys: set[tuple] = set()
    if output_path.exists():
        previous = pd.read_csv(output_path)
        rows_out = previous.to_dict("records")
        done_keys = {
            (row["table_id"], row["task"], row["fmt"], row["question"])
            for row in rows_out
        }
        log(f"resume_detected=yes existing_rows={len(rows_out)} existing_keys={len(done_keys)}")
    else:
        log("resume_detected=no")

    pending_rows = []
    skipped_prediction_rows = 0
    for _, row in df.iterrows():
        key = (row["table_id"], row["task"], row["fmt"], row["question"])
        if key not in done_keys:
            prediction_text = str(row[target_field]).replace("\\n", "\n")
            if prediction_text.startswith("[SKIPPED:"):
                skipped_prediction_rows += 1
                continue
            pending_rows.append(row)
    log(
        f"total_prediction_rows={len(df)} pending_judge_rows={len(pending_rows)} "
        f"skipped_prediction_rows={skipped_prediction_rows}"
    )
    return df, output_path, rows_out, pending_rows


def _write_judge_batch_results(
    *,
    output_path: Path,
    rows_out: list[dict],
    batch_inputs: list[dict],
    batch_results: list[dict],
    log,
) -> None:
    for batch_row, data in zip(batch_inputs, batch_results):
        out_row = {
            "table_id": batch_row["table_id"],
            "task": batch_row["task"],
            "fmt": batch_row["fmt"],
            "question": batch_row["question"],
            "gold": batch_row["gold"],
            "target_field": batch_row["target_field"],
            "prediction_sent_to_judge": batch_row["prediction"],
            "content_correct": data.get("content_correct"),
            "content_partial": data.get("content_partial"),
            "format_correct": data.get("format_correct"),
            "overinclusive": data.get("overinclusive"),
            "instruction_following_error": data.get("instruction_following_error"),
            "score": data.get("score"),
            "reason": data.get("reason", ""),
            "raw_judge": data.get("raw_judge", ""),
        }
        for passthrough_key in [
            "patch_group",
            "patch_layer",
            "base_model",
            "quant_model",
            "source_flip_metric",
            "source_flip_type",
        ]:
            if passthrough_key in batch_row:
                out_row[passthrough_key] = batch_row[passthrough_key]
        rows_out.append(out_row)
    pd.DataFrame(rows_out).to_csv(output_path, index=False)
    if len(rows_out) % 25 == 0 and batch_inputs:
        last = batch_inputs[-1]
        log(
            f"checkpoint rows_written={len(rows_out)} "
            f"last_table={last['table_id']} last_task={last['task']} last_fmt={last['fmt']}"
        )


def _run_judge_job_with_loaded_backend(
    *,
    model,
    tokenizer,
    backend: str,
    model_id: str,
    predictions_csv: str | Path,
    output_csv: str | Path,
    target_field: str,
    write_summary: bool,
    overwrite_summary: bool,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    server_base_url: str | None,
    prompt_version: str,
    summary_prefix: str | None,
) -> Path:
    output_path = Path(output_csv)
    log_path = output_path.with_suffix(".log")
    log = _make_logger(log_path)
    log("=== run_judge start ===")
    log(f"predictions_csv={predictions_csv}")
    log(f"output_csv={output_csv}")
    log(f"target_field={target_field}")
    log(f"write_summary={write_summary}")
    log(f"overwrite_summary={overwrite_summary}")
    log(f"backend={backend}")
    log(f"batch_size={batch_size}")
    log(f"max_new_tokens={max_new_tokens}")
    log(f"temperature={temperature}")
    log(f"prompt_version={prompt_version}")
    log(f"summary_prefix={summary_prefix}")

    _, output_path, rows_out, pending_rows = _load_pending_judge_rows(
        predictions_csv=predictions_csv,
        output_csv=output_csv,
        target_field=target_field,
        log=log,
    )

    if backend not in {"hf", "vllm", "vllm_server"}:
        raise ValueError(f"Unsupported judge backend: {backend}")

    def make_batch_input(row) -> dict:
        batch_input = {
            "table_id": row["table_id"],
            "task": str(row["task"]),
            "fmt": row["fmt"],
            "question": str(row["question"]),
            "gold": str(row["gold"]),
            "prediction": str(row[target_field]).replace("\\n", "\n"),
            "target_field": target_field,
        }
        for passthrough_key in [
            "patch_group",
            "patch_layer",
            "base_model",
            "quant_model",
            "source_flip_metric",
            "source_flip_type",
        ]:
            if passthrough_key in row:
                batch_input[passthrough_key] = row[passthrough_key]
        return batch_input

    if backend == "vllm":
        iterator = tqdm(
            range(0, len(pending_rows), batch_size),
            desc=f"judge {output_path.parent.name}",
            unit="batch",
        )
        for start in iterator:
            batch_rows = pending_rows[start : start + batch_size]
            batch_inputs = [make_batch_input(row) for row in batch_rows]
            batch_results = judge_batch_vllm(
                model,
                tokenizer,
                batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_version=prompt_version,
            )
            _write_judge_batch_results(
                output_path=output_path,
                rows_out=rows_out,
                batch_inputs=batch_inputs,
                batch_results=batch_results,
                log=log,
            )
    elif backend == "vllm_server":
        if not server_base_url:
            raise ValueError("server_base_url is required for backend='vllm_server'")
        iterator = tqdm(
            range(0, len(pending_rows), batch_size),
            desc=f"judge {output_path.parent.name}",
            unit="batch",
        )
        for start in iterator:
            batch_rows = pending_rows[start : start + batch_size]
            batch_inputs = [make_batch_input(row) for row in batch_rows]
            batch_results = judge_batch_vllm_server(
                batch_inputs,
                model_id=model_id,
                server_base_url=server_base_url,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt_version=prompt_version,
            )
            _write_judge_batch_results(
                output_path=output_path,
                rows_out=rows_out,
                batch_inputs=batch_inputs,
                batch_results=batch_results,
                log=log,
            )
    else:
        iterator = tqdm(
            pending_rows,
            desc=f"judge {output_path.parent.name}",
            unit="instance",
        )
        for row in iterator:
            prediction_text = str(row[target_field]).replace("\\n", "\n")
            data = judge_one(
                model,
                tokenizer,
                task=str(row["task"]),
                question=str(row["question"]),
                gold=str(row["gold"]),
                prediction=prediction_text,
                prompt_version=prompt_version,
            )
            _write_judge_batch_results(
                output_path=output_path,
                rows_out=rows_out,
                batch_inputs=[{**make_batch_input(row), "prediction": prediction_text}],
                batch_results=[data],
                log=log,
            )

    log(f"rows_written_total={len(rows_out)}")
    if write_summary:
        outputs = write_judge_summaries(
            output_path,
            output_dir=output_path.parent,
            overwrite=overwrite_summary,
            summary_prefix=summary_prefix,
        )
        log(f"summary_task_fmt={outputs['task_fmt']}")
        log(f"summary_fmt={outputs['fmt']}")
        log(f"summary_overall={outputs['overall']}")
    log("=== run_judge done ===")
    return output_path


def run_judge_jobs(
    *,
    jobs: list[tuple[str | Path, str | Path]],
    model_id: str,
    hf_token: str | None,
    target_field: str = "raw",
    write_summary: bool = True,
    overwrite_summary: bool = False,
    backend: str = "vllm",
    batch_size: int = 32,
    max_new_tokens: int = 180,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    server_base_url: str | None = None,
    prompt_version: str = "current",
    summary_prefix: str | None = None,
) -> list[Path]:
    if backend not in {"hf", "vllm", "vllm_server"}:
        raise ValueError(f"Unsupported judge backend: {backend}")
    if prompt_version not in JUDGE_PROMPTS:
        raise ValueError(f"Unsupported judge prompt version: {prompt_version}")
    if not jobs:
        return []

    if backend == "vllm":
        model, tokenizer = load_vllm_judge(
            model_id,
            hf_token,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    elif backend == "vllm_server":
        model, tokenizer = None, None
    else:
        model, tokenizer = load_judge(model_id, hf_token)

    outputs: list[Path] = []
    try:
        for predictions_csv, output_csv in jobs:
            outputs.append(
                _run_judge_job_with_loaded_backend(
                    model=model,
                    tokenizer=tokenizer,
                    backend=backend,
                    model_id=model_id,
                    predictions_csv=predictions_csv,
                    output_csv=output_csv,
                    target_field=target_field,
                    write_summary=write_summary,
                    overwrite_summary=overwrite_summary,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    server_base_url=server_base_url,
                    prompt_version=prompt_version,
                    summary_prefix=summary_prefix,
                )
            )
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return outputs


def run_judge(
    *,
    predictions_csv: str | Path,
    output_csv: str | Path,
    model_id: str,
    hf_token: str | None,
    target_field: str = "raw",
    write_summary: bool = True,
    overwrite_summary: bool = False,
    backend: str = "vllm",
    batch_size: int = 32,
    max_new_tokens: int = 180,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    server_base_url: str | None = None,
    prompt_version: str = "current",
    summary_prefix: str | None = None,
) -> Path:
    return run_judge_jobs(
        jobs=[(predictions_csv, output_csv)],
        model_id=model_id,
        hf_token=hf_token,
        target_field=target_field,
        write_summary=write_summary,
        overwrite_summary=overwrite_summary,
        backend=backend,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        server_base_url=server_base_url,
        prompt_version=prompt_version,
        summary_prefix=summary_prefix,
    )[0]
