from __future__ import annotations

import argparse
from pathlib import Path

from src.suc_tasks import generate_tasks_for_table, load_snapshot_tables, write_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "fixed_wtq_300"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "fixed_wtq_300_suc"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_snapshot_tables(args.snapshot_dir)

    all_tasks = []
    for example in examples:
        all_tasks.extend(generate_tasks_for_table(example, seed=args.seed))

    write_tasks(
        args.output_dir,
        all_tasks,
        source_snapshot_dir=args.snapshot_dir,
        seed=args.seed,
    )

    print(
        f"Saved {len(all_tasks)} task rows from {len(examples)} tables to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
