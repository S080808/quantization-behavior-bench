from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.suc_pipeline import load_selected_table_ids, run_predictions


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
        "--suc-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "suc_300"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results" / "suc_baseline_50"),
    )
    parser.add_argument("--n-tables", type=int, default=50)
    parser.add_argument("--selected-table-ids-json", default=None)
    parser.add_argument("--seed-predictions-csv", default=None)
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--formats", nargs="+", default=["html", "json", "col_sep", "prose", "row_obj"])
    parser.add_argument("--max-prompt-tokens", type=int, default=3900)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.model
    selected_table_ids = None
    if args.selected_table_ids_json:
        selected_table_ids = load_selected_table_ids(args.selected_table_ids_json)
    pred_path = run_predictions(
        model_key=args.model,
        wtq_dir=args.wtq_dir,
        suc_dir=args.suc_dir,
        output_dir=output_dir,
        formats=args.formats,
        hf_token=args.hf_token,
        max_prompt_tokens=args.max_prompt_tokens,
        n_tables=args.n_tables,
        selected_table_ids=selected_table_ids,
        seed_predictions_csv=args.seed_predictions_csv,
        one_shot=args.one_shot,
        backend=args.backend,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
