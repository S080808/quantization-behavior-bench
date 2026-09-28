from __future__ import annotations

import argparse
from pathlib import Path

from src.wtq_data import load_wtq_examples, write_wtq_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--max-cols", type=int, default=12)
    parser.add_argument("--sort-by-size", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_300"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_rows = args.max_rows if args.max_rows > 0 else None
    max_cols = args.max_cols if args.max_cols > 0 else None

    examples = load_wtq_examples(
        split=args.split,
        n_samples=args.n,
        seed=args.seed,
        max_rows=max_rows,
        max_cols=max_cols,
        sort_by_size=args.sort_by_size,
    )

    write_wtq_data(
        args.output_dir,
        examples,
        split=args.split,
        seed=args.seed,
        max_rows=max_rows,
        max_cols=max_cols,
        sort_by_size=args.sort_by_size,
    )

    print(f"Saved {len(examples)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
