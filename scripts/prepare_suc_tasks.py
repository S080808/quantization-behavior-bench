from __future__ import annotations

import argparse
from pathlib import Path

from src.suc_tasks import generate_tasks_for_table, load_wtq_tables, write_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wtq-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "wtq_300"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "suc_300"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_wtq_tables(args.wtq_dir)

    all_tasks = []
    for example in examples:
        all_tasks.extend(generate_tasks_for_table(example, seed=args.seed))

    write_tasks(
        args.output_dir,
        all_tasks,
        source_wtq_dir=args.wtq_dir,
        seed=args.seed,
    )

    print(
        f"Saved {len(all_tasks)} task rows from {len(examples)} tables to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
