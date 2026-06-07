from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.judge import run_judge_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run judge over many prediction CSVs while keeping the backend loaded once."
    )
    parser.add_argument("--predictions-csvs", nargs="+", required=True)
    parser.add_argument("--output-csvs", nargs="+")
    parser.add_argument("--output-name", default="judge_raw.csv")
    parser.add_argument("--judge-model-id", default="google/gemma-2-9b-it")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf", "vllm_server"])
    parser.add_argument(
        "--target-field",
        default="raw",
        help="Column from the predictions CSV to send to the judge, for example raw, pred, or patched_raw.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--overwrite-summary", action="store_true")
    parser.add_argument("--summary-prefix", default=None)
    parser.add_argument("--prompt-version", default="current", choices=["current", "suc_v1"])
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--server-base-url", default=os.environ.get("JUDGE_SERVER_BASE_URL"))
    return parser.parse_args()


def resolve_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    predictions = [str(Path(path)) for path in args.predictions_csvs]
    missing_predictions = [path for path in predictions if not Path(path).exists()]
    if missing_predictions:
        joined = "\n".join(f"  - {path}" for path in missing_predictions)
        raise FileNotFoundError(
            "Some --predictions-csvs paths do not exist.\n"
            "Check for typos such as an accidental leading '/'.\n"
            f"{joined}"
        )
    if args.output_csvs:
        outputs = [str(Path(path)) for path in args.output_csvs]
        if len(outputs) != len(predictions):
            raise ValueError("--output-csvs must have the same length as --predictions-csvs")
        return list(zip(predictions, outputs))
    return [
        (predictions_csv, str(Path(predictions_csv).with_name(args.output_name)))
        for predictions_csv in predictions
    ]


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)
    outputs = run_judge_jobs(
        jobs=jobs,
        model_id=args.judge_model_id,
        hf_token=args.hf_token,
        target_field=args.target_field,
        write_summary=not args.no_summary,
        overwrite_summary=args.overwrite_summary,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        server_base_url=args.server_base_url,
        prompt_version=args.prompt_version,
        summary_prefix=args.summary_prefix,
    )
    for output in outputs:
        print(f"Saved judge rows to {output}")


if __name__ == "__main__":
    main()
