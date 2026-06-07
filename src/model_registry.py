from __future__ import annotations

import gc
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS: dict[str, dict] = {
    "llama": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "kind": "fp16",
        "torch_dtype": torch.float16,
    },
    "llama-awq": {
        "model_id": "TheBloke/Llama-2-7B-Chat-AWQ",
        "kind": "awq",
        "torch_dtype": torch.float16,
    },
    "llama-gptq": {
        "model_id": "TheBloke/Llama-2-7B-Chat-GPTQ",
        "kind": "gptq",
        "torch_dtype": torch.float16,
    },
    "llama-quip": {
        "model_id": "relaxml/Llama-2-7b-chat-E8P-2Bit",
        "kind": "quip",
        "torch_dtype": torch.float16,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "kind": "fp16",
        "torch_dtype": torch.float16,
    },
    "qwen-awq": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "kind": "awq",
        "torch_dtype": torch.float16,
    },
    "qwen-gptq": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "kind": "gptq",
        "torch_dtype": torch.float16,
    },
}


LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_tokenizer(model_key: str, hf_token: str | None, *, local_files_only: bool = False):
    model_id = MODELS[model_key]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    return tokenizer


def load_model(model_key: str, hf_token: str | None, *, local_files_only: bool = False):
    cfg = MODELS[model_key]
    model_id = cfg["model_id"]
    tokenizer = load_tokenizer(model_key, hf_token, local_files_only=local_files_only)

    if cfg["kind"] == "awq":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=cfg["torch_dtype"],
            token=hf_token,
            local_files_only=local_files_only,
        )
    elif cfg["kind"] == "gptq":
        from gptqmodel import GPTQModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = GPTQModel.from_quantized(
            model_id,
            device=device,
            token=hf_token,
            local_files_only=local_files_only,
        )
    elif cfg["kind"] == "quip":
        model = _load_quip(model_id, hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=cfg["torch_dtype"],
            token=hf_token,
            local_files_only=local_files_only,
        )

    model.eval()
    return model, tokenizer


def load_vllm_model(
    model_key: str,
    hf_token: str | None,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
):
    from vllm import LLM

    cfg = MODELS[model_key]
    if cfg["kind"] == "quip":
        raise ValueError(
            "llama-quip is not supported with vLLM in this repro branch. "
            "Use --backend hf and a quip-sharp installation."
        )
    model_id = cfg["model_id"]
    tokenizer = load_tokenizer(model_key, hf_token)

    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if hf_token:
        llm_kwargs["hf_token"] = hf_token
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    model = LLM(**llm_kwargs)
    return model, tokenizer


def _load_quip(model_id: str, hf_token: str | None):
    quip_dir = os.environ.get("QUIP_SHARP_DIR")
    if not quip_dir:
        raise ImportError(
            "QUIP_SHARP_DIR is not set. Clone and install quip-sharp:\n"
            "  git clone https://github.com/Cornell-RelaxML/quip-sharp.git\n"
            "  cd quip-sharp/quiptools && python setup.py install && cd ../..\n"
            "  export QUIP_SHARP_DIR=$(pwd)/quip-sharp"
        )
    from pathlib import Path

    root = Path(quip_dir).expanduser().resolve()
    candidates = [
        root,
        root.parent,
        root / "quip-sharp",
        root.parent / "quip-sharp",
    ]

    added_paths: list[str] = []
    for candidate in candidates:
        if (candidate / "lib" / "utils" / "unsafe_import.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            added_paths.append(candidate_str)

    if not added_paths:
        checked = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise ImportError(
            "Could not locate quip-sharp Python sources.\n"
            "Expected to find lib/utils/unsafe_import.py under one of:\n"
            f"{checked}\n"
            "Set QUIP_SHARP_DIR to the quip-sharp repo root."
        )

    try:
        from lib.utils.unsafe_import import model_from_hf_path
    except ModuleNotFoundError as exc:
        checked = "\n".join(f"  - {path}" for path in added_paths)
        raise ImportError(
            "Found candidate quip-sharp paths but still could not import "
            "`lib.utils.unsafe_import`.\n"
            f"Tried sys.path roots:\n{checked}"
        ) from exc

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _ = model_from_hf_path(model_id, use_cuda_graph=False, device_map=device)
    return model


def unload_model(model) -> None:
    for shutdown in (
        getattr(model, "shutdown", None),
        getattr(getattr(model, "llm_engine", None), "shutdown", None),
        getattr(getattr(getattr(model, "llm_engine", None), "engine_core", None), "shutdown", None),
    ):
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(2)
