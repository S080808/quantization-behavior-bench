from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_registry import MODELS, load_vllm_model, unload_model

FORMATS = {"html": "context_html", "latex": "context_latex", "markdown": "context_markdown"}
SYSTEM = "You are a helpful assistant. Think step-by-step and provide a detailed and helpful response."
TASK = "信息查询"


def load_ir(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row["task_name"] == TASK]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("TableEval Information Retrieval IDs are not unique")
    if len(rows) != 637:
        raise ValueError(f"Expected 637 Information Retrieval examples, found {len(rows)}")
    for row in rows:
        if len(row["question_list"]) != 1 or len(row["golden_answer_list"]) != 1:
            raise ValueError(f"Unexpected multi-turn Information Retrieval example: {row['id']}")
        for key in FORMATS.values():
            if key not in row["context"]:
                raise ValueError(f"Missing {key} for example {row['id']}")
    return rows


def prompt_for(row: dict, fmt: str, tokenizer) -> str:
    user = row["instruction"].format(
        context=row["context"][FORMATS[fmt]], question=row["question_list"][0]
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def save(path: Path, records: dict[str, dict]) -> None:
    ordered = sorted(records.values(), key=lambda row: int(row["id"]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TableEval Information Retrieval predictions")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--test-data", type=Path, required=True,
                        help="Upstream TableEval data/TableEval-test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tableeval"))
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    rows = load_ir(args.test_data)
    from vllm import SamplingParams

    model, tokenizer = load_vllm_model(
        args.model, args.hf_token,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    try:
        output_dir = args.output_dir / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        for fmt in args.formats:
            path = output_dir / f"{args.model}_cot_{fmt}_{TASK}.json"
            records = {}
            if path.exists():
                records = {str(row["id"]): row for row in json.loads(path.read_text(encoding="utf-8"))}
            pending = [row for row in rows if str(row["id"]) not in records]
            for start in tqdm(range(0, len(pending), args.batch_size), desc=f"{args.model} {fmt}"):
                batch = pending[start:start + args.batch_size]
                prompts = [prompt_for(row, fmt, tokenizer) for row in batch]
                outputs = model.generate(prompts, sampling, use_tqdm=False)
                for row, output in zip(batch, outputs, strict=True):
                    records[str(row["id"])] = {
                        "id": str(row["id"]),
                        "task_name": row["task_name"],
                        "sub_task_name": row["sub_task_name"],
                        "golden_answer_list": row["golden_answer_list"],
                        "prediction_list": [output.outputs[0].text],
                    }
                save(path, records)
            if len(records) != len(rows):
                raise RuntimeError(f"Incomplete {fmt} output: {len(records)}/{len(rows)}")
            print(path)
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()
