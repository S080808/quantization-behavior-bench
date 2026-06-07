from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.code_model_registry import get_code_model


SWE_QA_PROMPT = """You are answering a multiple-choice question about a codebase.

Read the code context and choose the single best option.
Return only the option letter: A, B, C, or D.

Question:
{question}

Code context:
{code}

Options:
{options_block}

Answer:"""


CODEXGLUE_PROMPT = """You are filling a single masked token in source code.

Return only the exact missing token that should replace <mask>.
Do not add explanation, punctuation, or extra text.

Docstring hint:
{nl_text}

Code:
{code_text}

Missing token:"""


CRUXEVAL_OUTPUT_PROMPT = """You are reasoning about a Python function.

Given the function and the exact Python input expression, return only the exact Python output expression.
Do not add explanation or extra text.

Function:
{code}

Input:
{input_expr}

Output:"""


LIVECODEBENCH_EXECUTION_PROMPT = """You are predicting the output of a Python code snippet.

Given the code and the exact input, return only the exact output text produced for that input.
Preserve line breaks if needed. Do not add explanation.

Function name:
{function_name}

Code:
{code}

Input:
{input_text}

Output:"""


def _make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    return _log


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_token(text: str) -> str:
    return text.strip().strip("`'\"").strip()


def _normalize_options(options) -> list[tuple[str, str]]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if isinstance(options, Mapping):
        pairs = []
        for label in labels:
            if label in options:
                pairs.append((label, str(options[label])))
        if pairs:
            return pairs
    return [(labels[i], str(option)) for i, option in enumerate(options)]


def _options_block(options) -> str:
    lines = [f"{label}. {option}" for label, option in _normalize_options(options)]
    return "\n".join(lines)


def render_swe_qa_prompt(code: str, question: str, options) -> str:
    return SWE_QA_PROMPT.format(
        question=question,
        code=code,
        options_block=_options_block(options),
    )


def render_codexglue_prompt(nl_tokens: list[str], pl_tokens: list[str]) -> str:
    return CODEXGLUE_PROMPT.format(
        nl_text=" ".join(nl_tokens),
        code_text=" ".join(pl_tokens),
    )


def render_cruxeval_output_prompt(code: str, input_expr: str) -> str:
    return CRUXEVAL_OUTPUT_PROMPT.format(code=code, input_expr=input_expr)


def render_livecodebench_execution_prompt(function_name: str, code: str, input_text: str) -> str:
    return LIVECODEBENCH_EXECUTION_PROMPT.format(
        function_name=function_name or "<unknown>",
        code=code,
        input_text=input_text,
    )


def parse_mc_prediction(text: str, options) -> tuple[str, str]:
    option_pairs = _normalize_options(options)
    option_by_label = dict(option_pairs)
    valid_labels = "".join(label for label, _ in option_pairs)
    cleaned = text.strip()

    explicit = re.findall(
        rf"(?:answer|option|choice|final answer)\s*(?:is|:|-)?\s*\(?([{re.escape(valid_labels)}])\)?\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if explicit:
        letter = explicit[-1].upper()
        return letter, option_by_label[letter]

    for line in cleaned.splitlines():
        line_clean = line.strip().strip("`").strip()
        match = re.match(rf"^\(?([{re.escape(valid_labels)}])\)?(?:[).:\s]|$)", line_clean, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            return letter, option_by_label[letter]

    normalized = _normalize_text(cleaned).lower()
    for label, option in option_pairs:
        if _normalize_text(option).lower() in normalized:
            return label, option

    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    return "", first_line


def parse_cloze_prediction(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    first_token = first_line.split()[0] if first_line else ""
    return _normalize_token(first_token)


def parse_exact_text_prediction(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:[A-Za-z0-9_+-]+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^\s*(?:answer|output|result)\s*(?:is|:|-)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def load_vllm_model(
    model_key: str,
    hf_token: str | None,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
):
    from vllm import LLM

    cfg = get_code_model(model_key)
    llm_kwargs = {
        "model": cfg["model_id"],
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
    }
    if hf_token:
        llm_kwargs["hf_token"] = hf_token
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    return LLM(**llm_kwargs)


def generate_batch_vllm(
    llm,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    texts: list[str] = []
    for item in outputs:
        if item.outputs:
            texts.append(item.outputs[0].text.strip())
        else:
            texts.append("")
    return texts


def _load_hf_model_and_tokenizer(model_key: str, hf_token: str | None):
    cfg = get_code_model(model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        token=hf_token,
        use_fast=True,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        token=hf_token,
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch_hf(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    outputs = model.generate(**encoded, **generation_kwargs)
    texts: list[str] = []
    for i, prompt_len in enumerate(prompt_lens):
        text = tokenizer.decode(outputs[i][int(prompt_len):], skip_special_tokens=True).strip()
        texts.append(text)
    return texts


def _dataset_to_records(dataset) -> list[dict]:
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return [dict(row) for row in dataset]


def _select_split(dataset_name: str, config_name: str | None, split_preference: str):
    def _load(split_name: str):
        if config_name is None:
            return load_dataset(dataset_name, split=split_name)
        return load_dataset(dataset_name, config_name, split=split_name)

    if split_preference != "auto":
        return _load(split_preference)
    for split_name in ("validation", "dev", "oracle", "test", "train"):
        try:
            return _load(split_name)
        except Exception:
            continue
    raise ValueError(f"Could not locate a usable split for dataset={dataset_name} config={config_name}")


def _normalized_multiline_text(text: str) -> str:
    return _safe_text(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def _python_literal_equal(prediction: str, target: str) -> bool:
    pred = _normalized_multiline_text(prediction)
    gold = _normalized_multiline_text(target)
    try:
        return ast.literal_eval(pred) == ast.literal_eval(gold)
    except Exception:
        return pred == gold


def _python_literal_type(text: str) -> str:
    try:
        value = ast.literal_eval(_normalized_multiline_text(text))
    except Exception:
        return "text"
    return type(value).__name__


def _numsteps_bucket(value: object) -> str:
    try:
        n = int(value)
    except Exception:
        return "unknown"
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    return "7+"


def _summarize_binary_accuracy(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    rows = []
    for group_key, frame in grouped:
        row: dict[str, object] = {"n": len(frame), "accuracy": round(float(frame["correct"].mean()), 4)}
        if group_cols:
            if len(group_cols) == 1:
                row[group_cols[0]] = group_key
            else:
                row.update(dict(zip(group_cols, group_key)))
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_correct_option(example: dict) -> tuple[str, str]:
    option_pairs = _normalize_options(example["options"])
    option_by_label = dict(option_pairs)
    correct = example["correct_answer"]
    if isinstance(correct, str):
        correct_clean = correct.strip()
        if correct_clean in option_by_label:
            return correct_clean, option_by_label[correct_clean]
        for label, option in option_pairs:
            if _normalize_text(option) == _normalize_text(correct_clean):
                return label, option
    raise ValueError(f"Could not resolve correct answer for SWE-QA example: {correct!r}")


def write_summary_files(
    output_dir: Path,
    *,
    overall: pd.DataFrame,
    by_group: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    overall_path = output_dir / "summary_overall.csv"
    overall.to_csv(overall_path, index=False)
    paths["overall"] = overall_path
    for name, frame in by_group.items():
        path = output_dir / f"summary_by_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def run_swe_qa(
    *,
    model_key: str,
    output_dir: str | Path,
    split: str = "oracle",
    max_examples: int | None = None,
    hf_token: str | None = None,
    backend: str = "vllm",
    batch_size: int = 32,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"
    log = _make_logger(output_dir / "run.log")
    log("=== run_swe_qa start ===")
    log(f"model_key={model_key}")
    log(f"split={split}")
    log(f"backend={backend}")
    log(f"overwrite={overwrite}")

    dataset = _select_split("lailaelkoussy/swe-qa", None, split)
    records = _dataset_to_records(dataset)
    if max_examples is not None:
        records = records[:max_examples]
    log(f"dataset_rows={len(records)}")

    rows_out: list[dict] = []
    done_ids: set[str] = set()
    if overwrite and output_csv.exists():
        output_csv.unlink()
        log(f"overwrite_removed_existing={output_csv}")
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows_out = previous.to_dict("records")
        done_ids = set(previous["row_id"].astype(str).tolist())
        log(f"resume_detected=yes existing_rows={len(rows_out)}")
    else:
        log("resume_detected=no")

    pending = []
    for idx, example in enumerate(records):
        row_id = str(example.get("id", idx))
        if row_id not in done_ids:
            pending.append((row_id, example))
    log(f"pending_rows={len(pending)}")

    model = tokenizer = None
    llm = None
    try:
        if backend == "vllm":
            llm = load_vllm_model(
                model_key,
                hf_token,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "hf":
            model, tokenizer = _load_hf_model_and_tokenizer(model_key, hf_token)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        total_batches = math.ceil(len(pending) / batch_size) if pending else 0
        iterator = tqdm(range(0, len(pending), batch_size), total=total_batches, desc="sweqa", unit="batch")
        for start in iterator:
            batch = pending[start : start + batch_size]
            prompts = []
            metadata = []
            for row_id, example in batch:
                options = example["options"]
                correct_label, correct_text = _resolve_correct_option(example)
                prompts.append(render_swe_qa_prompt(example["code"], example["question"], options))
                metadata.append((row_id, example, options, correct_label, correct_text))
            if llm is not None:
                texts = generate_batch_vllm(llm, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            else:
                texts = generate_batch_hf(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)

            for raw_text, (row_id, example, options, correct_label, correct_text) in zip(texts, metadata):
                pred_label, pred_text = parse_mc_prediction(raw_text, options)
                rows_out.append(
                    {
                        "row_id": row_id,
                        "question": example["question"],
                        "category": example.get("category", ""),
                        "correct_label": correct_label,
                        "correct_answer": correct_text,
                        "predicted_label": pred_label,
                        "predicted_answer": pred_text,
                        "raw_prediction": raw_text,
                        "correct": pred_label == correct_label,
                    }
                )
            pd.DataFrame(rows_out).to_csv(output_csv, index=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.read_csv(output_csv)
    overall = _summarize_binary_accuracy(df)
    by_category = _summarize_binary_accuracy(df, group_cols=["category"])
    outputs = write_summary_files(output_dir, overall=overall, by_group={"category": by_category})
    log(f"predictions_csv={output_csv}")
    for key, path in outputs.items():
        log(f"summary_{key}={path}")
    log("=== run_swe_qa done ===")
    return output_csv


def _find_codexglue_base_dir(codexglue_dir: str | Path) -> Path:
    root = Path(codexglue_dir).expanduser().resolve()
    candidates = [
        root / "Code-Code" / "ClozeTesting-all",
        root / "ClozeTesting-all",
        root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate CodeXGLUE ClozeTesting-all under {root}")


def _find_answer_file(base_dir: Path, language: str, *, answers_dir: str | Path | None = None) -> Path:
    if answers_dir:
        answer_root = Path(answers_dir).expanduser().resolve()
        candidates = [
            answer_root / language / "answer.txt",
            answer_root / language / "answers.txt",
            answer_root / f"{language}.answer.txt",
            answer_root / f"{language}.answers.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    direct_candidates = [
        base_dir / language / "answer.txt",
        base_dir / language / "answers.txt",
        base_dir / "data" / "cloze-all" / language / "answer.txt",
        base_dir / "data" / "cloze-all" / language / "answers.txt",
        base_dir / f"{language}_answer.txt",
        base_dir / f"{language}_answers.txt",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    matches = sorted(base_dir.rglob(f"{language}*answer*.txt")) + sorted(base_dir.rglob(f"{language}*answers*.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find answer file for language={language} under {base_dir}")


def _find_optional_answer_file(
    base_dir: Path,
    language: str,
    *,
    answers_dir: str | Path | None = None,
) -> Path | None:
    try:
        return _find_answer_file(base_dir, language, answers_dir=answers_dir)
    except FileNotFoundError:
        return None


def _load_answer_lookup(answer_file: Path) -> tuple[dict[str, str], list[str]]:
    by_idx: dict[str, str] = {}
    ordered: list[str] = []
    with answer_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = re.split(r"\s+", stripped)
            if len(parts) >= 2 and parts[0].startswith("all-"):
                key = parts[0]
                value = parts[-1]
                by_idx[key] = value
                ordered.append(value)
            else:
                ordered.append(parts[-1])
    return by_idx, ordered


def _summarize_codexglue(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    rows = []
    for group_key, frame in grouped:
        row: dict[str, object] = {"n": len(frame), "labeled_n": 0, "accuracy": ""}
        labeled = frame[frame["target"].fillna("").astype(str) != ""]
        row["labeled_n"] = len(labeled)
        if len(labeled):
            row["accuracy"] = round(float(labeled["correct"].mean()), 4)
        if group_cols:
            if len(group_cols) == 1:
                row[group_cols[0]] = group_key
            else:
                row.update(dict(zip(group_cols, group_key)))
        rows.append(row)
    return pd.DataFrame(rows)


def run_codexglue_cloze(
    *,
    model_key: str,
    output_dir: str | Path,
    codexglue_dir: str | Path,
    codexglue_answers_dir: str | Path | None = None,
    languages: list[str],
    split: str = "auto",
    max_examples_per_language: int | None = None,
    hf_token: str | None = None,
    backend: str = "vllm",
    batch_size: int = 64,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    require_answers: bool = False,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"
    log = _make_logger(output_dir / "run.log")
    log("=== run_codexglue_cloze start ===")
    log(f"model_key={model_key}")
    log(f"languages={languages}")
    log(f"backend={backend}")
    log(f"require_answers={require_answers}")
    log(f"overwrite={overwrite}")

    base_dir = _find_codexglue_base_dir(codexglue_dir)
    rows_out: list[dict] = []
    done_keys: set[tuple[str, str]] = set()
    if overwrite and output_csv.exists():
        output_csv.unlink()
        log(f"overwrite_removed_existing={output_csv}")
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows_out = previous.to_dict("records")
        done_keys = set(zip(previous["language"].astype(str), previous["idx"].astype(str)))
        log(f"resume_detected=yes existing_rows={len(rows_out)}")
    else:
        log("resume_detected=no")

    pending = []
    for language in languages:
        dataset = _select_split("google/code_x_glue_cc_cloze_testing_all", language, split)
        records = _dataset_to_records(dataset)
        if max_examples_per_language is not None:
            records = records[:max_examples_per_language]
        answer_file = _find_optional_answer_file(base_dir, language, answers_dir=codexglue_answers_dir)
        if answer_file is None:
            if require_answers:
                raise FileNotFoundError(
                    f"Could not find CodeXGLUE answer file for language={language}. "
                    "Official ClozeTesting-all gold answers are not public; provide "
                    "--codexglue-answers-dir with local labeled data or omit --require-answers "
                    "to run prediction-only."
                )
            log(f"answer_file[{language}]=<missing; prediction_only>")
            by_idx, ordered = {}, []
        else:
            log(f"answer_file[{language}]={answer_file}")
            by_idx, ordered = _load_answer_lookup(answer_file)
        for order_idx, example in enumerate(records):
            idx = str(example.get("idx", order_idx))
            if (language, idx) in done_keys:
                continue
            target = by_idx.get(idx, "")
            if not target and ordered:
                if order_idx >= len(ordered):
                    raise ValueError(
                        f"Answer lookup failed for language={language} idx={idx}. "
                        f"Answer file {answer_file} has only {len(ordered)} rows."
                    )
                target = ordered[order_idx]
            pending.append((language, idx, example, target))
    log(f"pending_rows={len(pending)}")

    model = tokenizer = None
    llm = None
    try:
        if backend == "vllm":
            llm = load_vllm_model(
                model_key,
                hf_token,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "hf":
            model, tokenizer = _load_hf_model_and_tokenizer(model_key, hf_token)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        total_batches = math.ceil(len(pending) / batch_size) if pending else 0
        iterator = tqdm(range(0, len(pending), batch_size), total=total_batches, desc="codexglue", unit="batch")
        for start in iterator:
            batch = pending[start : start + batch_size]
            prompts = [render_codexglue_prompt(example["nl_tokens"], example["pl_tokens"]) for _, _, example, _ in batch]
            if llm is not None:
                texts = generate_batch_vllm(llm, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            else:
                texts = generate_batch_hf(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            for raw_text, (language, idx, example, target) in zip(texts, batch):
                prediction = parse_cloze_prediction(raw_text)
                rows_out.append(
                    {
                        "language": language,
                        "idx": idx,
                        "target": _normalize_token(target) if target else "",
                        "prediction": prediction,
                        "raw_prediction": raw_text,
                        "correct": (prediction == _normalize_token(target)) if target else pd.NA,
                    }
                )
            pd.DataFrame(rows_out).to_csv(output_csv, index=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.read_csv(output_csv)
    overall = _summarize_codexglue(df)
    by_language = _summarize_codexglue(df, group_cols=["language"])
    outputs = write_summary_files(output_dir, overall=overall, by_group={"language": by_language})
    log(f"predictions_csv={output_csv}")
    for key, path in outputs.items():
        log(f"summary_{key}={path}")
    log("=== run_codexglue_cloze done ===")
    return output_csv


def run_cruxeval_output(
    *,
    model_key: str,
    output_dir: str | Path,
    split: str = "test",
    max_examples: int | None = None,
    hf_token: str | None = None,
    backend: str = "vllm",
    batch_size: int = 32,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"
    log = _make_logger(output_dir / "run.log")
    log("=== run_cruxeval_output start ===")
    log(f"model_key={model_key}")
    log(f"split={split}")
    log(f"backend={backend}")
    log(f"overwrite={overwrite}")

    dataset = _select_split("cruxeval-org/cruxeval", None, split)
    records = _dataset_to_records(dataset)
    if max_examples is not None:
        records = records[:max_examples]
    log(f"dataset_rows={len(records)}")

    rows_out: list[dict] = []
    done_ids: set[str] = set()
    if overwrite and output_csv.exists():
        output_csv.unlink()
        log(f"overwrite_removed_existing={output_csv}")
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows_out = previous.to_dict("records")
        done_ids = set(previous["row_id"].astype(str).tolist())
        log(f"resume_detected=yes existing_rows={len(rows_out)}")
    else:
        log("resume_detected=no")

    pending = []
    for idx, example in enumerate(records):
        row_id = str(example.get("id", idx))
        if row_id not in done_ids:
            pending.append((row_id, example))
    log(f"pending_rows={len(pending)}")

    model = tokenizer = None
    llm = None
    try:
        if backend == "vllm":
            llm = load_vllm_model(
                model_key,
                hf_token,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "hf":
            model, tokenizer = _load_hf_model_and_tokenizer(model_key, hf_token)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        total_batches = math.ceil(len(pending) / batch_size) if pending else 0
        iterator = tqdm(range(0, len(pending), batch_size), total=total_batches, desc="cruxeval", unit="batch")
        for start in iterator:
            batch = pending[start : start + batch_size]
            prompts = [render_cruxeval_output_prompt(example["code"], example["input"]) for _, example in batch]
            if llm is not None:
                texts = generate_batch_vllm(llm, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            else:
                texts = generate_batch_hf(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            for raw_text, (row_id, example) in zip(texts, batch):
                prediction = parse_exact_text_prediction(raw_text)
                gold = _safe_text(example.get("output", ""))
                rows_out.append(
                    {
                        "row_id": row_id,
                        "code": example.get("code", ""),
                        "input": example.get("input", ""),
                        "gold_output": gold,
                        "predicted_output": prediction,
                        "raw_prediction": raw_text,
                        "output_type": _python_literal_type(gold),
                        "correct": _python_literal_equal(prediction, gold),
                    }
                )
            pd.DataFrame(rows_out).to_csv(output_csv, index=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.read_csv(output_csv)
    overall = _summarize_binary_accuracy(df)
    by_output_type = _summarize_binary_accuracy(df, group_cols=["output_type"])
    outputs = write_summary_files(output_dir, overall=overall, by_group={"output_type": by_output_type})
    log(f"predictions_csv={output_csv}")
    for key, path in outputs.items():
        log(f"summary_{key}={path}")
    log("=== run_cruxeval_output done ===")
    return output_csv


def run_livecodebench_execution(
    *,
    model_key: str,
    output_dir: str | Path,
    split: str = "test",
    max_examples: int | None = None,
    hf_token: str | None = None,
    backend: str = "vllm",
    batch_size: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "predictions.csv"
    log = _make_logger(output_dir / "run.log")
    log("=== run_livecodebench_execution start ===")
    log(f"model_key={model_key}")
    log(f"split={split}")
    log(f"backend={backend}")
    log(f"overwrite={overwrite}")

    dataset = _select_split("livecodebench/execution", None, split)
    records = _dataset_to_records(dataset)
    if max_examples is not None:
        records = records[:max_examples]
    log(f"dataset_rows={len(records)}")

    rows_out: list[dict] = []
    done_ids: set[str] = set()
    if overwrite and output_csv.exists():
        output_csv.unlink()
        log(f"overwrite_removed_existing={output_csv}")
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        rows_out = previous.to_dict("records")
        done_ids = set(previous["row_id"].astype(str).tolist())
        log(f"resume_detected=yes existing_rows={len(rows_out)}")
    else:
        log("resume_detected=no")

    pending = []
    for idx, example in enumerate(records):
        row_id = str(example.get("id", idx))
        if row_id not in done_ids:
            pending.append((row_id, example))
    log(f"pending_rows={len(pending)}")

    model = tokenizer = None
    llm = None
    try:
        if backend == "vllm":
            llm = load_vllm_model(
                model_key,
                hf_token,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        elif backend == "hf":
            model, tokenizer = _load_hf_model_and_tokenizer(model_key, hf_token)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        total_batches = math.ceil(len(pending) / batch_size) if pending else 0
        iterator = tqdm(range(0, len(pending), batch_size), total=total_batches, desc="livecodebench", unit="batch")
        for start in iterator:
            batch = pending[start : start + batch_size]
            prompts = [
                render_livecodebench_execution_prompt(
                    _safe_text(example.get("function_name", "")),
                    _safe_text(example.get("code", "")),
                    _safe_text(example.get("input", "")),
                )
                for _, example in batch
            ]
            if llm is not None:
                texts = generate_batch_vllm(llm, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            else:
                texts = generate_batch_hf(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            for raw_text, (row_id, example) in zip(texts, batch):
                prediction = parse_exact_text_prediction(raw_text)
                gold = _normalized_multiline_text(example.get("output", ""))
                rows_out.append(
                    {
                        "row_id": row_id,
                        "function_name": example.get("function_name", ""),
                        "problem_id": example.get("problem_id", ""),
                        "numsteps": example.get("numsteps", ""),
                        "numsteps_bucket": _numsteps_bucket(example.get("numsteps", "")),
                        "gold_output": gold,
                        "predicted_output": _normalized_multiline_text(prediction),
                        "raw_prediction": raw_text,
                        "correct": _normalized_multiline_text(prediction) == gold,
                    }
                )
            pd.DataFrame(rows_out).to_csv(output_csv, index=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.read_csv(output_csv)
    overall = _summarize_binary_accuracy(df)
    by_numsteps_bucket = _summarize_binary_accuracy(df, group_cols=["numsteps_bucket"])
    outputs = write_summary_files(output_dir, overall=overall, by_group={"numsteps_bucket": by_numsteps_bucket})
    log(f"predictions_csv={output_csv}")
    for key, path in outputs.items():
        log(f"summary_{key}={path}")
    log("=== run_livecodebench_execution done ===")
    return output_csv
