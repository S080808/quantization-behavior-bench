from __future__ import annotations

from transformers import AutoTokenizer


SYSTEM_PROMPT_CHAT = (
    "You are a table question answering assistant. "
    "Answer questions based only on the provided table. "
    "Give only the final answer value, nothing else."
)


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


SIZE_DETECTION_QUESTION = (
    "Count the rows in the table one by one, then count the columns. "
    "How many rows and columns does this table have? "
    "Answer exactly as: N rows, M columns."
)

QA_PROMPT_TEMPLATE = (
    "Table:\n{table_text}\n"
    "---\n"
    "Using only the table above, answer the question.\n"
    "Give a short answer: one word, number, or phrase. No explanation.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def to_col_sep(header: list[str], rows: list[list[str]]) -> str:
    sep = " <COL> "
    lines = [sep.join(header)]
    for row in rows:
        lines.append(sep.join(row))
    return "\n".join(lines)


def build_qa_prompt_body(table_text: str, question: str) -> str:
    return QA_PROMPT_TEMPLATE.format(table_text=table_text, question=question)


def load_llama_tokenizer(hf_token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    return tokenizer


def render_prompt_with_chat_template(table_text: str, question: str, hf_token: str | None) -> str:
    tokenizer = load_llama_tokenizer(hf_token)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": build_qa_prompt_body(table_text, question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_to_token_artifact(prompt: str, hf_token: str | None) -> str:
    tokenizer = load_llama_tokenizer(hf_token)
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    return "".join(piece.replace("▁", "_") for piece in pieces)
