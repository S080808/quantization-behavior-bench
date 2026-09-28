from __future__ import annotations

import hashlib
import random
import re
from contextlib import contextmanager
from typing import Iterator

import torch


def decoder_layers(model) -> tuple[torch.nn.Module, list[torch.nn.Module]]:
    """Return the decoder backbone and its ordered transformer blocks."""
    queue = [model]
    seen: set[int] = set()
    while queue:
        module = queue.pop(0)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return module, list(layers)
        for name in ("model", "base_model", "transformer"):
            child = getattr(module, name, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)
    raise TypeError("Could not locate transformer decoder layers")


def _mark(mask: list[bool], start: int, end: int) -> None:
    for index in range(max(0, start), min(len(mask), end)):
        mask[index] = True


def format_character_mask(prompt: str, table_text: str, fmt: str) -> list[bool]:
    """Label serializer-owned characters for the supported patching formats."""
    table_start = prompt.find(table_text)
    if table_start < 0:
        raise ValueError("Serialized table was not found in the rendered prompt")

    local = [False] * len(table_text)
    if fmt == "col_sep":
        for match in re.finditer(r" <COL> ", table_text):
            _mark(local, match.start(), match.end())
        for match in re.finditer(r"\n", table_text):
            _mark(local, match.start(), match.end())
    elif fmt == "html":
        for match in re.finditer(r"<[^>]+>|^[ \t]+|\n", table_text, re.MULTILINE):
            _mark(local, match.start(), match.end())
    elif fmt == "json":
        local[:] = [True] * len(local)
        # Quotation marks remain format; string contents are table content.
        for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', table_text):
            for index in range(match.start(1), match.end(1)):
                local[index] = False
    else:
        raise ValueError(f"Unsupported patching format: {fmt}")

    mask = [False] * len(prompt)
    mask[table_start : table_start + len(table_text)] = local
    return mask


def format_token_positions(tokenizer, prompt: str, table_text: str, fmt: str) -> list[int]:
    char_mask = format_character_mask(prompt, table_text, fmt)
    offsets = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    return [
        token_index
        for token_index, (start, end) in enumerate(offsets)
        if end > start and any(char_mask[start:end])
    ]


def random_token_positions(
    sequence_length: int,
    count: int,
    *,
    seed: int,
    instance_key: str,
) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{instance_key}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return sorted(rng.sample(range(sequence_length), min(count, sequence_length)))


def _hidden(output):
    return output if torch.is_tensor(output) else output[0]


@torch.inference_mode()
def capture_clean_states(
    model,
    model_inputs: dict[str, torch.Tensor],
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    backbone, layers = decoder_layers(model)
    states: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layer_indices:

        def save_state(_module, _inputs, output, index=layer_index):
            states[index] = _hidden(output).detach()

        handles.append(layers[layer_index].register_forward_hook(save_state))
    try:
        backbone(**model_inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return states


@contextmanager
def patch_residual_stream(
    model,
    clean_states: dict[int, torch.Tensor],
    token_positions: list[int],
) -> Iterator[None]:
    """Patch block outputs on prefill; leave one-token decode steps untouched."""
    _, layers = decoder_layers(model)
    handles = []
    for layer_index, clean in clean_states.items():

        def patch(_module, _inputs, output, source=clean):
            hidden = _hidden(output)
            if hidden.shape[1] <= 1:
                return output
            positions = [position for position in token_positions if position < hidden.shape[1]]
            if not positions:
                return output
            patched = hidden.clone()
            patched[:, positions, :] = source[:, positions, :].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        handles.append(layers[layer_index].register_forward_hook(patch))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
