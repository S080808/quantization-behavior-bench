from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from datasets import load_dataset


@dataclass
class WTQExample:
    id: str
    question: str
    answers: list[str]
    header: list[str]
    rows: list[list[str]]


def parse_table(item: dict) -> tuple[list[str], list[list[str]]]:
    table = item.get("table")

    if isinstance(table, dict):
        header = table.get("header") or table.get("column_names") or table.get("columns")
        rows = table.get("rows") or table.get("data") or table.get("content")
        if header is not None and rows is not None:
            return list(header), [list(row) for row in rows]

    header = item.get("table_column_names") or item.get("column_names")
    rows = item.get("table_content_values") or item.get("content_values")
    if header is not None and rows is not None:
        return list(header), [list(row) for row in rows]

    raise ValueError("Cannot parse table from item")


def normalize_answers(item: dict) -> list[str]:
    answers = (
        item.get("answer_text")
        or item.get("answers")
        or item.get("answer")
        or item.get("seq_out")
        or []
    )
    if isinstance(answers, str):
        answers = [answers]
    return [str(answer) for answer in answers if str(answer).strip()]


def load_wtq_examples(
    split: str = "test",
    n_samples: int | None = None,
    seed: int = 42,
    max_rows: int | None = 25,
    max_cols: int | None = 12,
    sort_by_size: bool = False,
) -> list[WTQExample]:
    try:
        dataset = load_dataset("TableQAKit/WTQ", split=split)
    except Exception:
        fallback = "train" if split != "train" else "test"
        print(f"[wtq_data] Split '{split}' not found, trying '{fallback}'")
        dataset = load_dataset("TableQAKit/WTQ", split=fallback)

    examples: list[WTQExample] = []
    for item in dataset:
        try:
            header, rows = parse_table(item)
        except (KeyError, TypeError, ValueError):
            continue

        if not header or not rows:
            continue

        if max_cols is not None:
            header = header[:max_cols]
            rows = [row[:max_cols] for row in rows]
        if max_rows is not None:
            rows = rows[:max_rows]

        header = [str(value) for value in header]
        rows = [[str(value) for value in row] for row in rows]

        answers = normalize_answers(item)
        if not answers:
            continue

        examples.append(
            WTQExample(
                id=str(item.get("id", len(examples))),
                question=str(item.get("question", "")).strip(),
                answers=answers,
                header=header,
                rows=rows,
            )
        )

    if sort_by_size:
        examples.sort(key=lambda ex: (-len(ex.rows), -len(ex.header)))
        if n_samples and n_samples < len(examples):
            examples = examples[:n_samples]
    elif n_samples and n_samples < len(examples):
        examples = random.Random(seed).sample(examples, n_samples)

    return examples


def load_wtq_data(
    wtq_dir: str | Path,
    *,
    n_tables: int | None = None,
) -> list[WTQExample]:
    wtq_path = Path(wtq_dir)
    examples: list[WTQExample] = []
    with (wtq_path / "tables.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(WTQExample(**record))
    if n_tables is not None and n_tables > 0:
        return examples[:n_tables]
    return examples


def write_wtq_data(
    output_dir: str | Path,
    examples: list[WTQExample],
    *,
    split: str,
    seed: int,
    max_rows: int | None,
    max_cols: int | None,
    sort_by_size: bool,
) -> None:
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    output_path.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "TableQAKit/WTQ",
        "split": split,
        "seed": seed,
        "n_tables": len(examples),
        "max_rows": max_rows,
        "max_cols": max_cols,
        "sort_by_size": sort_by_size,
        "selection_mode": "largest_after_truncation" if sort_by_size else "random_after_truncation",
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_path / "tables.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            record = asdict(example)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            (tables_dir / f"{example.id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
