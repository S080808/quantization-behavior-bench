from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.suc_pipeline import select_complete_table_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wtq-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_300"),
    )
    parser.add_argument(
        "--suc-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "suc_300"),
    )
    parser.add_argument(
        "--model",
        default="llama",
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
    parser.add_argument("--formats", nargs="+", default=["html", "json", "col_sep"])
    parser.add_argument("--max-prompt-tokens", type=int, default=3900)
    parser.add_argument("--n-tables", type=int, default=50)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_selected_50.json"),
    )
    parser.add_argument(
        "--diagnostics-json",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_selected_50_diagnostics.json"),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_ids, diagnostics = select_complete_table_ids(
        wtq_dir=args.wtq_dir,
        suc_dir=args.suc_dir,
        formats=args.formats,
        model_key=args.model,
        hf_token=args.hf_token,
        max_prompt_tokens=args.max_prompt_tokens,
        n_tables_target=args.n_tables,
    )

    output_payload = {
        "model": args.model,
        "formats": args.formats,
        "max_prompt_tokens": args.max_prompt_tokens,
        "n_tables_target": args.n_tables,
        "n_tables_selected": len(selected_ids),
        "table_ids": selected_ids,
    }
    Path(args.output_json).write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.diagnostics_json).write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"selected_tables={len(selected_ids)}")
    print(f"output_json={args.output_json}")
    print(f"diagnostics_json={args.diagnostics_json}")


if __name__ == "__main__":
    main()
