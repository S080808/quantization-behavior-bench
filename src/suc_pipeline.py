from __future__ import annotations

import json
import re
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from src.model_registry import load_model, load_tokenizer, load_vllm_model, unload_model
from src.suc_tasks import LIST_TASKS, load_tasks_jsonl, load_wtq_tables
from src.table_formats import FORMATTERS


SYSTEM_PROMPT_CHAT = (
    "You are a table question answering assistant. "
    "Answer questions based only on the provided table. "
    "Give only the final answer value, nothing else."
)

EX_HEADER = ["Player", "Club", "Goals"]
EX_ROWS = [["Messi", "Barcelona", "91"], ["Ronaldo", "Real Madrid", "450"]]

EX_QA: dict[str, dict[str, str]] = {
    "size_detection": {
        "question": (
            "Count the rows in the table one by one, then count the columns. "
            "How many rows and columns does this table have? "
            "Answer exactly as: N rows, M columns."
        ),
        "answer": (
            "Row 1 (Messi), Row 2 (Ronaldo) = 2 rows. "
            "Columns: Player, Club, Goals = 3 columns. "
            "2 rows, 3 columns"
        ),
    },
    "cell_lookup": {
        "question": "What value is in row 1, column 'Club'?",
        "answer": "Barcelona",
    },
    "reverse_lookup": {
        "question": (
            "In which row and column does the value '450' appear? "
            "Answer exactly as: row N, column 'ColName'."
        ),
        "answer": "row 2, column 'Goals'",
    },
    "row_retrieval": {
        "question": "List all values in row 2, one value per line.",
        "answer": "Ronaldo\nReal Madrid\n450",
    },
    "col_retrieval": {
        "question": "List all values in the column 'Goals', one value per line.",
        "answer": "91\n450",
    },
}

QA_PROMPT_TEMPLATE = (
    "Table:\n{table_text}\n"
    "---\n"
    "Using only the table above, answer the question.\n"
    "Give a short answer: one word, number, or phrase. No explanation.\n\n"
    "Question: {question}\n"
    "Answer:"
)

LIST_PROMPT_TEMPLATE = (
    "Table:\n{table_text}\n"
    "---\n"
    "Using only the table above, answer the question.\n"
    "List each value on a separate line. No extra text.\n\n"
    "Question: {question}\n"
    "Answer:"
)

STOP_STRINGS_QA = [
    "\n\n", "\nQuestion:", "\nTable:", "\n###", "\n---",
    "Human:", "\nHuman:", "Assistant:", "\nAssistant:",
]
STOP_STRINGS_LIST = [
    "\nQuestion:", "\nTable:", "\n###", "\n---",
    "Human:", "\nHuman:", "Assistant:", "\nAssistant:",
]

MAX_NEW_TOKENS_SHORT = 64
MAX_NEW_TOKENS_LIST = 256


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_ids: list[list[int]]):
        self.stop_ids = [s for s in stop_ids if s]

    def __call__(self, input_ids: torch.LongTensor, scores, **_) -> bool:
        if input_ids.shape[0] != 1:
            return False
        row = input_ids[0].tolist()
        return any(row[-len(s):] == s for s in self.stop_ids if len(row) >= len(s))


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return _log


def load_selected_table_ids(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ids = data.get("table_ids", [])
    else:
        ids = data
    return [str(value) for value in ids]


def load_table_index(
    wtq_dir: str | Path,
    *,
    n_tables: int | None = None,
    selected_table_ids: list[str] | None = None,
) -> dict[str, dict]:
    tables = load_wtq_tables(wtq_dir)
    if selected_table_ids is not None:
        selected = set(selected_table_ids)
        tables = [table for table in tables if table.id in selected]
        order = {table_id: idx for idx, table_id in enumerate(selected_table_ids)}
        tables.sort(key=lambda table: order.get(table.id, 10**9))
    elif n_tables is not None:
        tables = tables[:n_tables]
    return {
        table.id: {
            "header": table.header,
            "rows": table.rows,
        }
        for table in tables
    }


def build_prompt_body(table_text: str, question: str, is_list: bool) -> str:
    template = LIST_PROMPT_TEMPLATE if is_list else QA_PROMPT_TEMPLATE
    return template.format(table_text=table_text, question=question)


def build_oneshot_prompt_body(
    *,
    table_text: str,
    question: str,
    task: str,
    is_list: bool,
    fmt: str,
) -> str:
    ex = EX_QA.get(task, EX_QA["cell_lookup"])
    fmt_fn = FORMATTERS[fmt]
    ex_table = fmt_fn(EX_HEADER, EX_ROWS)
    instruction = (
        "List each value on a separate line. No extra text."
        if is_list
        else "Give a short answer: one word, number, or phrase. No explanation."
    )
    return (
        f"Table:\n{ex_table}\n"
        f"---\n"
        f"Using only the table above, answer the question.\n"
        f"{instruction}\n\n"
        f"Question: {ex['question']}\n"
        f"Answer: {ex['answer']}\n\n"
        f"Table:\n{table_text}\n"
        f"---\n"
        f"Using only the table above, answer the question.\n"
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def render_chat_prompt(tokenizer, prompt_body: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": prompt_body},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_prediction(raw: str, is_list: bool, stop_strings: list[str]) -> str:
    for stop in stop_strings:
        if stop in raw:
            raw = raw.split(stop)[0]
    if "Answer:" in raw:
        raw = raw.rsplit("Answer:", 1)[1]
    raw = raw.strip()
    if is_list:
        lines = [line.strip().strip("\"'`.,;-") for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)
    for line in raw.splitlines():
        line = line.strip().strip("\"'`.,;")
        if line:
            match = re.match(r"^-\s+\S[^:]*:\s+(.*)", line)
            if match:
                line = match.group(1).strip().strip("\"'`.,;")
            if len(line) > 120:
                for delimiter in (",", "\t", " | ", " § ", " <COL> "):
                    if delimiter in line:
                        line = line.split(delimiter)[0].strip().strip("\"'`.,;")
                        break
                else:
                    line = line[:120]
            return line
    return ""


def prepare_instance_prompt(inst: dict, tokenizer, *, one_shot: bool) -> dict:
    is_list = inst["task"] in LIST_TASKS
    stop_strings = STOP_STRINGS_LIST if is_list else STOP_STRINGS_QA
    max_new_tokens = MAX_NEW_TOKENS_LIST if is_list else MAX_NEW_TOKENS_SHORT
    if one_shot:
        prompt_body = build_oneshot_prompt_body(
            table_text=inst["table_text"],
            question=inst["question"],
            task=inst["task"],
            is_list=is_list,
            fmt=inst["fmt"],
        )
    else:
        prompt_body = build_prompt_body(inst["table_text"], inst["question"], is_list)
    prompt = render_chat_prompt(tokenizer, prompt_body)
    n_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return {
        **inst,
        "is_list": is_list,
        "stop_strings": stop_strings,
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "prompt_tokens": n_tokens,
    }


@torch.no_grad()
def generate_raw(model, tokenizer, prompt: str, max_new_tokens: int, stop_strings: list[str]) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnSequence(stop_ids)]),
    )
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def generate_raw_batch_vllm(model, batch_inputs: list[dict]) -> list[str]:
    from vllm import SamplingParams

    groups: dict[tuple[bool, int, tuple[str, ...]], list[tuple[int, dict]]] = {}
    for idx, item in enumerate(batch_inputs):
        key = (
            item["is_list"],
            item["max_new_tokens"],
            tuple(item["stop_strings"]),
        )
        groups.setdefault(key, []).append((idx, item))

    outputs: list[str] = [""] * len(batch_inputs)
    for (_, max_new_tokens, stop_strings), group in groups.items():
        prompts = [item["prompt"] for _, item in group]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=list(stop_strings),
        )
        group_outputs = model.generate(prompts, sampling_params, use_tqdm=False)
        for (orig_idx, _), out in zip(group, group_outputs):
            outputs[orig_idx] = out.outputs[0].text if out.outputs else ""
    return outputs


def build_instances(
    wtq_dir: str | Path,
    suc_dir: str | Path,
    formats: list[str],
    *,
    n_tables: int | None = None,
    selected_table_ids: list[str] | None = None,
) -> list[dict]:
    table_index = load_table_index(
        wtq_dir,
        n_tables=n_tables,
        selected_table_ids=selected_table_ids,
    )
    table_ids = set(table_index.keys())
    tasks = load_tasks_jsonl(suc_dir, table_ids=table_ids)
    instances: list[dict] = []
    for task in tasks:
        table = table_index[task["table_id"]]
        for fmt in formats:
            table_text = FORMATTERS[fmt](table["header"], table["rows"])
            instances.append(
                {
                    **task,
                    "fmt": fmt,
                    "table_text": table_text,
                }
            )
    return instances


def run_predictions(
    *,
    model_key: str,
    wtq_dir: str | Path,
    suc_dir: str | Path,
    output_dir: str | Path,
    formats: list[str],
    hf_token: str | None,
    max_prompt_tokens: int = 3900,
    n_tables: int | None = None,
    selected_table_ids: list[str] | None = None,
    seed_predictions_csv: str | Path | None = None,
    one_shot: bool = False,
    backend: str = "hf",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
) -> Path:
    instances = build_instances(
        wtq_dir,
        suc_dir,
        formats,
        n_tables=n_tables,
        selected_table_ids=selected_table_ids,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pred_path = output_path / f"{model_key}_predictions.csv"
    log_path = output_path / f"{model_key}_run.log"
    log = _make_logger(log_path)
    log("=== run_predictions start ===")
    log(f"model={model_key}")
    log(f"wtq_dir={wtq_dir}")
    log(f"suc_dir={suc_dir}")
    log(f"formats={formats}")
    log(f"n_tables={n_tables}")
    log(f"selected_table_ids_count={len(selected_table_ids) if selected_table_ids is not None else 0}")
    log(f"max_prompt_tokens={max_prompt_tokens}")
    log(f"one_shot={one_shot}")
    log(f"backend={backend}")
    log(f"batch_size={batch_size}")
    log(f"tensor_parallel_size={tensor_parallel_size}")
    log(f"gpu_memory_utilization={gpu_memory_utilization}")
    log(f"max_model_len={max_model_len}")
    log(f"prediction_dump={pred_path}")
    log(f"seed_predictions_csv={seed_predictions_csv}")

    rows_out: list[dict] = []
    existing_keys: set[tuple] = set()
    target_keys = {
        (inst["table_id"], inst["task"], inst["fmt"], inst["question"])
        for inst in instances
    }

    if pred_path.exists():
        df_existing = pd.read_csv(pred_path)
        rows_out = [
            row
            for row in df_existing.to_dict("records")
            if (row["table_id"], row["task"], row["fmt"], row["question"]) in target_keys
        ]
        existing_keys = {
            (row["table_id"], row["task"], row["fmt"], row["question"])
            for row in rows_out
        }
        log(f"resume_detected=yes existing_rows={len(rows_out)} existing_keys={len(existing_keys)}")
    else:
        log("resume_detected=no")
        if seed_predictions_csv:
            seed_path = Path(seed_predictions_csv)
            if seed_path.exists():
                df_seed = pd.read_csv(seed_path)
                rows_out = [
                    row
                    for row in df_seed.to_dict("records")
                    if (row["table_id"], row["task"], row["fmt"], row["question"]) in target_keys
                ]
                existing_keys = {
                    (row["table_id"], row["task"], row["fmt"], row["question"])
                    for row in rows_out
                }
                pd.DataFrame(rows_out).to_csv(pred_path, index=False)
                log(f"seed_loaded=yes seed_rows={len(rows_out)} seed_keys={len(existing_keys)}")
            else:
                log("seed_loaded=no seed_file_missing")

    pending_instances = []
    for inst in instances:
        key = (inst["table_id"], inst["task"], inst["fmt"], inst["question"])
        if key not in existing_keys:
            pending_instances.append(inst)

    log(f"total_instances={len(instances)} pending_instances={len(pending_instances)}")

    skipped = 0
    if backend not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported prediction backend: {backend}")

    if backend == "vllm":
        model, tokenizer = load_vllm_model(
            model_key,
            hf_token,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        try:
            prepared_instances = [
                prepare_instance_prompt(inst, tokenizer, one_shot=one_shot)
                for inst in pending_instances
            ]
            pending_batches = math.ceil(len(prepared_instances) / batch_size) if prepared_instances else 0
            log(f"pending_batches={pending_batches}")
            iterator = tqdm(
                range(0, len(prepared_instances), batch_size),
                desc=f"{model_key} predictions",
                unit="batch",
            )
            for start in iterator:
                batch = prepared_instances[start : start + batch_size]
                generation_batch: list[dict] = []
                generation_positions: list[int] = []
                batch_raws: list[str] = [""] * len(batch)
                for idx, inst in enumerate(batch):
                    if max_prompt_tokens > 0 and inst["prompt_tokens"] > max_prompt_tokens:
                        skipped += 1
                        batch_raws[idx] = f"[SKIPPED: {inst['prompt_tokens']} tokens > {max_prompt_tokens}]"
                    else:
                        generation_positions.append(idx)
                        generation_batch.append(inst)
                if generation_batch:
                    generated = generate_raw_batch_vllm(model, generation_batch)
                    for pos, raw in zip(generation_positions, generated):
                        batch_raws[pos] = raw
                for inst, raw in zip(batch, batch_raws):
                    pred = ""
                    if not raw.startswith("[SKIPPED:"):
                        pred = extract_prediction(raw, inst["is_list"], inst["stop_strings"])
                    rows_out.append(
                        {
                            "table_id": inst["table_id"],
                            "task": inst["task"],
                            "fmt": inst["fmt"],
                            "question": inst["question"],
                            "gold": inst["gold"],
                            "n_rows": inst["n_rows"],
                            "n_cols": inst["n_cols"],
                            "prompt": inst["prompt"],
                            "prompt_tokens": inst["prompt_tokens"],
                            "pred": pred,
                            "raw": raw.replace("\n", "\\n"),
                        }
                    )
                pd.DataFrame(rows_out).to_csv(pred_path, index=False)
                if len(rows_out) % 25 == 0:
                    last = batch[-1]
                    log(
                        f"checkpoint rows_written={len(rows_out)} "
                        f"last_table={last['table_id']} last_task={last['task']} last_fmt={last['fmt']}"
                    )
        finally:
            unload_model(model)
    else:
        model, tokenizer = load_model(model_key, hf_token)
        try:
            iterator = tqdm(
                pending_instances,
                desc=f"{model_key} predictions",
                unit="instance",
            )
            for inst in iterator:
                prepared = prepare_instance_prompt(inst, tokenizer, one_shot=one_shot)
                if max_prompt_tokens > 0 and prepared["prompt_tokens"] > max_prompt_tokens:
                    skipped += 1
                    raw = f"[SKIPPED: {prepared['prompt_tokens']} tokens > {max_prompt_tokens}]"
                    pred = ""
                else:
                    raw = generate_raw(
                        model,
                        tokenizer,
                        prepared["prompt"],
                        prepared["max_new_tokens"],
                        prepared["stop_strings"],
                    )
                    pred = extract_prediction(raw, prepared["is_list"], prepared["stop_strings"])
                rows_out.append(
                    {
                        "table_id": prepared["table_id"],
                        "task": prepared["task"],
                        "fmt": prepared["fmt"],
                        "question": prepared["question"],
                        "gold": prepared["gold"],
                        "n_rows": prepared["n_rows"],
                        "n_cols": prepared["n_cols"],
                        "prompt": prepared["prompt"],
                        "prompt_tokens": prepared["prompt_tokens"],
                        "pred": pred,
                        "raw": raw.replace("\n", "\\n"),
                    }
                )
                pd.DataFrame(rows_out).to_csv(pred_path, index=False)
                if len(rows_out) % 25 == 0:
                    log(
                        f"checkpoint rows_written={len(rows_out)} "
                        f"last_table={prepared['table_id']} last_task={prepared['task']} last_fmt={prepared['fmt']}"
                    )
        finally:
            unload_model(model)

    manifest = {
        "model": model_key,
        "formats": formats,
        "wtq_dir": str(wtq_dir),
        "suc_dir": str(suc_dir),
        "n_tables_from_wtq_data": n_tables,
        "selected_table_ids_count": len(selected_table_ids) if selected_table_ids is not None else 0,
        "one_shot": one_shot,
        "backend": backend,
        "n_rows": len(rows_out),
        "skipped_long_prompts": skipped,
        "judge_target_field": "raw",
    }
    (output_path / f"{model_key}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"skipped_long_prompts={skipped}")
    log(f"rows_written_total={len(rows_out)}")
    log("=== run_predictions done ===")
    return pred_path


def select_complete_table_ids(
    *,
    wtq_dir: str | Path,
    suc_dir: str | Path,
    formats: list[str],
    model_key: str,
    hf_token: str | None,
    max_prompt_tokens: int,
    n_tables_target: int,
) -> tuple[list[str], list[dict]]:
    table_index = load_table_index(wtq_dir)
    ordered_table_ids = list(table_index.keys())
    tokenizer = load_tokenizer(model_key, hf_token)

    selected: list[str] = []
    diagnostics: list[dict] = []

    tasks = load_tasks_jsonl(suc_dir)
    tasks_by_table: dict[str, list[dict]] = {}
    for task in tasks:
        tasks_by_table.setdefault(task["table_id"], []).append(task)

    iterator = tqdm(ordered_table_ids, desc="select tables", unit="table")
    for table_id in iterator:
        table = table_index[table_id]
        all_ok = True
        max_seen = 0
        failing: list[dict] = []
        for task in tasks_by_table.get(table_id, []):
            is_list = task["task"] in LIST_TASKS
            for fmt in formats:
                table_text = FORMATTERS[fmt](table["header"], table["rows"])
                prompt_body = build_prompt_body(table_text, task["question"], is_list)
                prompt = render_chat_prompt(tokenizer, prompt_body)
                n_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                max_seen = max(max_seen, n_tokens)
                if max_prompt_tokens > 0 and n_tokens > max_prompt_tokens:
                    all_ok = False
                    failing.append(
                        {
                            "task": task["task"],
                            "fmt": fmt,
                            "prompt_tokens": n_tokens,
                        }
                    )
        diagnostics.append(
            {
                "table_id": table_id,
                "all_complete": all_ok,
                "max_prompt_tokens_seen": max_seen,
                "n_failing_instances": len(failing),
                "failing_instances": failing,
            }
        )
        if all_ok:
            selected.append(table_id)
            iterator.set_postfix(selected=len(selected), target=n_tables_target)
            if len(selected) >= n_tables_target:
                break

    return selected, diagnostics


def estimate_judge_prompt_lengths(
    predictions_csv: str | Path,
    judge_model_id: str,
    hf_token: str | None,
    *,
    target_field: str = "raw",
) -> pd.DataFrame:
    from src.judge import build_judge_prompt

    tokenizer = AutoTokenizer.from_pretrained(judge_model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    df = pd.read_csv(predictions_csv)
    lengths = []
    for _, row in df.iterrows():
        prediction_text = str(row[target_field]).replace("\\n", "\n")
        prompt = build_judge_prompt(
            task=str(row["task"]),
            question=str(row["question"]),
            gold=str(row["gold"]),
            prediction=prediction_text,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        lengths.append(
            {
                "table_id": row["table_id"],
                "task": row["task"],
                "fmt": row["fmt"],
                "target_field": target_field,
                "judge_prompt_tokens": n_tokens,
            }
        )
    return pd.DataFrame(lengths)
