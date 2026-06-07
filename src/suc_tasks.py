from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.wtq_snapshot import WTQExample, load_frozen_snapshot


TASKS_PER_TABLE: dict[str, int] = {
    "size_detection": 1,
    "cell_lookup": 3,
    "reverse_lookup": 3,
    "row_retrieval": 2,
    "col_retrieval": 2,
}

LIST_TASKS = {"row_retrieval", "col_retrieval"}


@dataclass
class SUCTask:
    table_id: str
    task: str
    question: str
    gold: str
    n_rows: int
    n_cols: int


def generate_tasks_for_table(example: WTQExample, seed: int) -> list[SUCTask]:
    rng = random.Random(seed)
    header = example.header
    rows = example.rows
    n_rows = len(rows)
    n_cols = len(header)
    tasks: list[SUCTask] = []

    tasks.append(
        SUCTask(
            table_id=example.id,
            task="size_detection",
            question=(
                "Count the rows in the table one by one, then count the columns. "
                "How many rows and columns does this table have? "
                "Answer exactly as: N rows, M columns."
            ),
            gold=f"{n_rows} rows, {n_cols} columns",
            n_rows=n_rows,
            n_cols=n_cols,
        )
    )

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for r_idx, c_idx in rng.sample(all_cells, min(TASKS_PER_TABLE["cell_lookup"], len(all_cells))):
        tasks.append(
            SUCTask(
                table_id=example.id,
                task="cell_lookup",
                question=f"What value is in row {r_idx + 1}, column '{header[c_idx]}'?",
                gold=rows[r_idx][c_idx],
                n_rows=n_rows,
                n_cols=n_cols,
            )
        )

    seen: dict[str, tuple[int, int]] = {}
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            if value not in seen:
                seen[value] = (r_idx, c_idx)
    unique_pool = list(seen.items())
    for value, (r_idx, c_idx) in rng.sample(
        unique_pool, min(TASKS_PER_TABLE["reverse_lookup"], len(unique_pool))
    ):
        tasks.append(
            SUCTask(
                table_id=example.id,
                task="reverse_lookup",
                question=(
                    f"In which row and column does the value '{value}' appear? "
                    f"Answer exactly as: row N, column 'ColName'."
                ),
                gold=f"row {r_idx + 1}, column '{header[c_idx]}'",
                n_rows=n_rows,
                n_cols=n_cols,
            )
        )

    for r_idx in rng.sample(range(n_rows), min(TASKS_PER_TABLE["row_retrieval"], n_rows)):
        tasks.append(
            SUCTask(
                table_id=example.id,
                task="row_retrieval",
                question=f"List all values in row {r_idx + 1}, one value per line.",
                gold="\n".join(rows[r_idx]),
                n_rows=n_rows,
                n_cols=n_cols,
            )
        )

    for c_idx in rng.sample(range(n_cols), min(TASKS_PER_TABLE["col_retrieval"], n_cols)):
        column_name = header[c_idx]
        gold_values = [rows[r][c_idx] for r in range(n_rows)]
        tasks.append(
            SUCTask(
                table_id=example.id,
                task="col_retrieval",
                question=f"List all values in the column '{column_name}', one value per line.",
                gold="\n".join(gold_values),
                n_rows=n_rows,
                n_cols=n_cols,
            )
        )

    return tasks


def load_snapshot_tables(snapshot_dir: str | Path, *, n_tables: int | None = None) -> list[WTQExample]:
    return load_frozen_snapshot(snapshot_dir, n_tables=n_tables)


def load_tasks_jsonl(
    tasks_dir: str | Path,
    *,
    table_ids: set[str] | None = None,
) -> list[dict]:
    tasks_path = Path(tasks_dir) / "tasks.jsonl"
    rows: list[dict] = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if table_ids is not None and row["table_id"] not in table_ids:
                continue
            rows.append(row)
    return rows


def write_tasks(
    output_dir: str | Path,
    all_tasks: list[SUCTask],
    *,
    source_snapshot_dir: str,
    seed: int,
) -> None:
    output_path = Path(output_dir)
    tasks_dir = output_path / "tasks"
    output_path.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_snapshot_dir": source_snapshot_dir,
        "seed": seed,
        "n_task_rows": len(all_tasks),
        "tasks_per_table": TASKS_PER_TABLE,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    grouped: dict[str, list[dict]] = {}
    with (output_path / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in all_tasks:
            record = asdict(task)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            grouped.setdefault(task.table_id, []).append(record)

    for table_id, records in grouped.items():
        (tasks_dir / f"{table_id}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
