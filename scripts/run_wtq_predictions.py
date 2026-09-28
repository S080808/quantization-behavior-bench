from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_registry import load_model, load_tokenizer, load_vllm_model, unload_model
from src.suc_pipeline import (
    _make_logger,
    build_prompt_body,
    extract_prediction,
    generate_raw,
    generate_raw_batch_vllm,
    render_chat_prompt,
)
from src.table_formats import FORMATTERS
from src.wtq_data import load_wtq_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "llama",
            "llama-awq",
            "llama-gptq",
            "llama-quip",
            "qwen",
            "qwen-awq",
            "qwen-gptq",
        ],
    )
    parser.add_argument(
        "--wtq-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_300"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "wtq_300"),
    )
    parser.add_argument("--n-tables", type=int, default=300)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["html", "json", "xml"])
    parser.add_argument("--max-prompt-tokens", type=int, default=3900)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def build_instances(wtq_dir: str | Path, formats: list[str], *, n_tables: int | None) -> list[dict]:
    examples = load_wtq_data(wtq_dir, n_tables=n_tables)
    instances: list[dict] = []
    for example in examples:
        for fmt in formats:
            table_text = FORMATTERS[fmt](example.header, example.rows)
            instances.append(
                {
                    "table_id": example.id,
                    "task": "wtq_qa",
                    "fmt": fmt,
                    "question": example.question,
                    "gold": "\n".join(example.answers),
                    "answers": example.answers,
                    "n_rows": len(example.rows),
                    "n_cols": len(example.header),
                    "table_text": table_text,
                }
            )
    return instances


def prepare_instance_prompt(inst: dict, tokenizer) -> dict:
    prompt_body = build_prompt_body(inst["table_text"], inst["question"], is_list=False)
    prompt = render_chat_prompt(tokenizer, prompt_body)
    n_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return {
        **inst,
        "is_list": False,
        "stop_strings": ["\n\n", "\nQuestion:", "\nTable:", "\n###", "\n---", "Human:", "\nHuman:", "Assistant:", "\nAssistant:"],
        "max_new_tokens": 64,
        "prompt": prompt,
        "prompt_tokens": n_tokens,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.model}_predictions.csv"
    log_path = output_dir / f"{args.model}_run.log"
    log = _make_logger(log_path)

    instances = build_instances(args.wtq_dir, args.formats, n_tables=args.n_tables)
    log("=== run_wtq_predictions start ===")
    log(f"model={args.model}")
    log(f"wtq_dir={args.wtq_dir}")
    log(f"formats={args.formats}")
    log(f"n_tables={args.n_tables}")
    log(f"backend={args.backend}")
    log(f"batch_size={args.batch_size}")
    log(f"tensor_parallel_size={args.tensor_parallel_size}")
    log(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    log(f"max_model_len={args.max_model_len}")
    log(f"prediction_dump={pred_path}")

    rows_out: list[dict] = []
    existing_keys: set[tuple] = set()
    if pred_path.exists():
        df_existing = pd.read_csv(pred_path)
        rows_out = df_existing.to_dict("records")
        existing_keys = {
            (row["table_id"], row["task"], row["fmt"], row["question"])
            for row in rows_out
        }
        log(f"resume_detected=yes existing_rows={len(rows_out)} existing_keys={len(existing_keys)}")
    else:
        log("resume_detected=no")

    pending_instances = [
        inst
        for inst in instances
        if (inst["table_id"], inst["task"], inst["fmt"], inst["question"]) not in existing_keys
    ]
    log(f"total_instances={len(instances)} pending_instances={len(pending_instances)}")

    skipped = 0
    if args.backend == "vllm":
        model, tokenizer = load_vllm_model(
            args.model,
            args.hf_token,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        try:
            prepared_instances = [prepare_instance_prompt(inst, tokenizer) for inst in pending_instances]
            pending_batches = math.ceil(len(prepared_instances) / args.batch_size) if prepared_instances else 0
            log(f"pending_batches={pending_batches}")
            iterator = tqdm(
                range(0, len(prepared_instances), args.batch_size),
                desc=f"{args.model} wtq",
                unit="batch",
            )
            for start in iterator:
                batch = prepared_instances[start : start + args.batch_size]
                generation_batch: list[dict] = []
                generation_positions: list[int] = []
                batch_raws: list[str] = [""] * len(batch)
                for idx, inst in enumerate(batch):
                    if args.max_prompt_tokens > 0 and inst["prompt_tokens"] > args.max_prompt_tokens:
                        skipped += 1
                        batch_raws[idx] = f"[SKIPPED: {inst['prompt_tokens']} tokens > {args.max_prompt_tokens}]"
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
                            "answers": json.dumps(inst["answers"], ensure_ascii=False),
                            "n_rows": inst["n_rows"],
                            "n_cols": inst["n_cols"],
                            "prompt": inst["prompt"],
                            "prompt_tokens": inst["prompt_tokens"],
                            "pred": pred,
                            "raw": raw.replace("\n", "\\n"),
                        }
                    )
                pd.DataFrame(rows_out).to_csv(pred_path, index=False)
        finally:
            unload_model(model)
    else:
        model, tokenizer = load_model(args.model, args.hf_token)
        try:
            iterator = tqdm(pending_instances, desc=f"{args.model} wtq", unit="instance")
            for inst in iterator:
                prepared = prepare_instance_prompt(inst, tokenizer)
                if args.max_prompt_tokens > 0 and prepared["prompt_tokens"] > args.max_prompt_tokens:
                    skipped += 1
                    raw = f"[SKIPPED: {prepared['prompt_tokens']} tokens > {args.max_prompt_tokens}]"
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
                        "answers": json.dumps(prepared["answers"], ensure_ascii=False),
                        "n_rows": prepared["n_rows"],
                        "n_cols": prepared["n_cols"],
                        "prompt": prepared["prompt"],
                        "prompt_tokens": prepared["prompt_tokens"],
                        "pred": pred,
                        "raw": raw.replace("\n", "\\n"),
                    }
                )
                pd.DataFrame(rows_out).to_csv(pred_path, index=False)
        finally:
            unload_model(model)

    manifest = {
        "model": args.model,
        "formats": args.formats,
        "wtq_dir": str(args.wtq_dir),
        "task_family": "wtq_plain",
        "n_tables_from_wtq_data": args.n_tables,
        "backend": args.backend,
        "n_rows": len(rows_out),
        "skipped_long_prompts": skipped,
        "judge_target_field": "raw",
    }
    (output_dir / f"{args.model}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"skipped_long_prompts={skipped}")
    log(f"rows_written_total={len(rows_out)}")
    log("=== run_wtq_predictions done ===")
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
