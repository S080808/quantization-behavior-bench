from __future__ import annotations

import argparse
import os

from src.code_benchmarks import run_cruxeval_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_cruxeval_output(
        model_key=args.model_key,
        output_dir=args.output_dir,
        split=args.split,
        max_examples=args.max_examples,
        hf_token=args.hf_token,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        overwrite=args.overwrite,
    )
    print(f"Saved CRUXEval predictions to {out}")


if __name__ == "__main__":
    main()
