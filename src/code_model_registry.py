from __future__ import annotations

import json
import os
import re
from pathlib import Path


MODEL_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "configs" / "code_bench_models.json"


def load_code_models() -> dict[str, dict]:
    with MODEL_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_code_model(model_key: str) -> dict:
    models = load_code_models()
    try:
        cfg = dict(models[model_key])
    except KeyError as exc:
        known = ", ".join(sorted(models))
        raise KeyError(f"Unknown code benchmark model '{model_key}'. Known: {known}") from exc
    override = os.environ.get(_model_id_override_env_name(model_key))
    if override:
        cfg["model_id"] = override
    return cfg


def _model_id_override_env_name(model_key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_key).strip("_").upper()
    return f"CODE_MODEL_ID_{normalized}"
