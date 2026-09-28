from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.activation_patching import (
    capture_clean_states,
    decoder_layers,
    format_token_positions,
    patch_residual_stream,
    random_token_positions,
)
from src.model_registry import load_model, unload_model
from src.suc_pipeline import (
    build_instances,
    extract_prediction,
    generate_raw,
    load_selected_table_ids,
    prepare_instance_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late-layer activation patching for SUC")
    parser.add_argument("--quant-model", required=True, choices=["llama-awq", "llama-gptq"])
    parser.add_argument("--wtq-dir", default="data/wtq_300")
    parser.add_argument("--suc-dir", default="data/suc_300")
    parser.add_argument("--selected-table-ids-json", default="data/wtq_selected_150.json")
    parser.add_argument("--formats", nargs="+", default=["col_sep", "html", "json"])
    parser.add_argument("--n-tables", type=int, default=150)
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(24, 32)))
    parser.add_argument(
        "--patch-groups",
        nargs="+",
        choices=["format_late", "random10pct"],
        default=["format_late", "random10pct"],
    )
    parser.add_argument("--random-masks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=3900)
    parser.add_argument("--output-dir", default="results/activation_patching")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def output_jobs(args: argparse.Namespace) -> list[tuple[str, int | None, Path]]:
    root = Path(args.output_dir) / args.quant_model
    jobs: list[tuple[str, int | None, Path]] = []
    if "format_late" in args.patch_groups:
        jobs.append(("format_late", None, root / "format_late_predictions.csv"))
    if "random10pct" in args.patch_groups:
        for mask_index in range(args.random_masks):
            jobs.append(
                (
                    "random10pct",
                    args.seed + mask_index,
                    root / f"random10pct_seed{mask_index:02d}_predictions.csv",
                )
            )
    return jobs


def read_rows(path: Path) -> tuple[list[dict], set[tuple[str, str, str, str]]]:
    if not path.exists():
        return [], set()
    rows = pd.read_csv(path).to_dict("records")
    keys = {
        (str(row["table_id"]), str(row["task"]), str(row["fmt"]), str(row["question"]))
        for row in rows
    }
    return rows, keys


def main() -> None:
    args = parse_args()
    selected_ids = load_selected_table_ids(args.selected_table_ids_json)
    instances = build_instances(
        args.wtq_dir,
        args.suc_dir,
        args.formats,
        n_tables=args.n_tables,
        selected_table_ids=selected_ids[: args.n_tables],
    )
    instances = [instance for instance in instances if instance["task"] == "col_retrieval"]

    jobs = output_jobs(args)
    rows_by_path: dict[Path, list[dict]] = {}
    done_by_path: dict[Path, set[tuple[str, str, str, str]]] = {}
    for _, _, path in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_by_path[path], done_by_path[path] = read_rows(path)

    source_model, source_tokenizer = load_model(
        "llama", args.hf_token, local_files_only=args.local_files_only
    )
    quant_model, quant_tokenizer = load_model(
        args.quant_model, args.hf_token, local_files_only=args.local_files_only
    )
    try:
        _, source_layers = decoder_layers(source_model)
        _, quant_layers = decoder_layers(quant_model)
        if not args.layers:
            raise ValueError("At least one patch layer is required")
        if min(args.layers) < 0 or max(args.layers) >= min(len(source_layers), len(quant_layers)):
            raise ValueError("Requested patch layer is outside the model")

        for instance in tqdm(instances, desc=args.quant_model, unit="instance"):
            prepared = prepare_instance_prompt(instance, quant_tokenizer, one_shot=False)
            key = (
                str(prepared["table_id"]),
                str(prepared["task"]),
                str(prepared["fmt"]),
                str(prepared["question"]),
            )
            pending = [job for job in jobs if key not in done_by_path[job[2]]]
            if not pending or prepared["prompt_tokens"] > args.max_prompt_tokens:
                continue

            quant_inputs = quant_tokenizer(prepared["prompt"], return_tensors="pt")
            source_inputs = source_tokenizer(prepared["prompt"], return_tensors="pt")
            if not torch.equal(quant_inputs["input_ids"], source_inputs["input_ids"]):
                raise ValueError("FP16 and quantized tokenizers produced different token IDs")
            source_device = next(source_model.parameters()).device
            source_inputs = {
                name: value.to(source_device) for name, value in source_inputs.items()
            }
            clean_states = capture_clean_states(source_model, source_inputs, args.layers)
            format_positions = format_token_positions(
                quant_tokenizer,
                prepared["prompt"],
                prepared["table_text"],
                prepared["fmt"],
            )
            sequence_length = quant_inputs["input_ids"].shape[1]

            for patch_group, random_seed, path in pending:
                positions = format_positions
                if patch_group == "random10pct":
                    positions = random_token_positions(
                        sequence_length,
                        len(format_positions),
                        seed=int(random_seed),
                        instance_key="|".join(key),
                    )
                with patch_residual_stream(quant_model, clean_states, positions):
                    raw = generate_raw(
                        quant_model,
                        quant_tokenizer,
                        prepared["prompt"],
                        prepared["max_new_tokens"],
                        prepared["stop_strings"],
                    )
                row = {
                    "table_id": prepared["table_id"],
                    "task": prepared["task"],
                    "fmt": prepared["fmt"],
                    "question": prepared["question"],
                    "gold": prepared["gold"],
                    "n_rows": prepared["n_rows"],
                    "n_cols": prepared["n_cols"],
                    "prompt": prepared["prompt"],
                    "prompt_tokens": prepared["prompt_tokens"],
                    "pred": extract_prediction(
                        raw, prepared["is_list"], prepared["stop_strings"]
                    ),
                    "raw": raw.replace("\n", "\\n"),
                    "patch_group": patch_group,
                    "patch_layer": f"{min(args.layers)}-{max(args.layers)}",
                    "base_model": "llama",
                    "quant_model": args.quant_model,
                    "patch_positions": len(positions),
                    "random_seed": "" if random_seed is None else random_seed,
                }
                rows_by_path[path].append(row)
                done_by_path[path].add(key)
                pd.DataFrame(rows_by_path[path]).to_csv(path, index=False)
    finally:
        unload_model(quant_model)
        unload_model(source_model)


if __name__ == "__main__":
    main()
