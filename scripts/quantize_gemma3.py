#!/usr/bin/env python3
"""Create comparable Gemma 3 12B W4A16 AWQ or GPTQ checkpoints for vLLM.

Both methods use the same 30 MMLU-philosophy calibration examples, group size
128, and the same ignored multimodal/output modules.  The output is saved in
compressed-tensors format, which vLLM loads directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
from llmcompressor.utils import load_context


MODEL_ID = "google/gemma-3-12b-it"
CALIBRATION_SAMPLES = 30
MAX_SEQ_LENGTH = 2048


def calibration_dataset() -> Dataset:
    # The current cais/mmlu release exposes test/validation/dev (the former
    # auxiliary_train name is no longer present).  The public Gemma AWQ
    # metadata describes a 30-example mmlu_philosophy calibration set; use a
    # deterministic slice of the philosophy test split when reproducing it.
    ds = load_dataset("cais/mmlu", "philosophy", split="test")
    rows = []
    for item in ds.select(range(min(CALIBRATION_SAMPLES, len(ds)))):
        choices = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(item["choices"]))
        rows.append({"text": f"Question: {item['question']}\n{choices}\nAnswer:"})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("awq", "gptq"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ds = calibration_dataset()

    # Gemma 3 is multimodal; load_context makes LLM Compressor use its
    # conditional-generation class while calibration remains text-only.
    with load_context(Gemma3ForConditionalGeneration):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        if args.method == "awq":
            # This is the vLLM/LLM- Compressor AWQ recipe: AWQ scaling followed
            # by asymmetric W4A16 compressed-tensors quantization.
            recipe = [
                AWQModifier(
                    mappings=[
                        AWQMapping(
                            "re:.*language_model.layers.\\d+.self_attn.v_proj$",
                            ["re:.*language_model.layers.\\d+.self_attn.o_proj$"],
                        ),
                        AWQMapping(
                            "re:.*language_model.layers.\\d+.pre_feedforward_layernorm$",
                            [
                                "re:.*language_model.layers.\\d+.mlp.gate_proj$",
                                "re:.*language_model.layers.\\d+.mlp.up_proj$",
                            ],
                        ),
                        AWQMapping(
                            "re:.*language_model.layers.\\d+.mlp.up_proj$",
                            ["re:.*language_model.layers.\\d+.mlp.down_proj$"],
                        ),
                    ]
                ),
                QuantizationModifier(
                    targets="Linear",
                    scheme="W4A16_ASYM",
                    ignore=["re:.*vision.*", "re:.*lm_head", "re:.*embed_tokens$"],
                ),
            ]
        else:
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["re:.*vision.*", "re:.*lm_head", "re:.*embed_tokens$"],
            )

        oneshot(
            model=model,
            processor=processor,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQ_LENGTH,
            num_calibration_samples=CALIBRATION_SAMPLES,
            shuffle_calibration_samples=False,
            batch_size=1,
            output_dir=str(args.output),
        )

    processor.save_pretrained(args.output)
    print(f"saved {args.method} checkpoint to {args.output}")


if __name__ == "__main__":
    main()
