"""Chat-message construction for an item. Used by eval AND by training-data rendering."""
from __future__ import annotations

from eval.answer_format import DEFAULT_VARIANT, format_instruction, render_target


def render_task(item: dict) -> str:
    """Task body: optional source text, question, then options / statements / matching lists."""
    parts: list[str] = []
    if item.get("context"):
        parts.append(f"Tekst źródłowy:\n{item['context'].strip()}")
    parts.append(item["question"].strip())
    t = item["type"]
    if t in ("mc", "multi"):
        parts.append("\n".join(f"{k}. {v.strip()}" for k, v in item["options"].items()))
    elif t == "tf":
        parts.append("\n".join(f"{i}. {s.strip()}" for i, s in enumerate(item["statements"], 1)))
    elif t == "match":
        left = "\n".join(f"{i}. {s.strip()}" for i, s in enumerate(item["left"], 1))
        right = "\n".join(f"{k}. {v.strip()}" for k, v in item["right"].items())
        parts.append(f"{left}\n\n{right}")
    return "\n\n".join(parts)


def build_user_prompt(item: dict, variant: str = DEFAULT_VARIANT) -> str:
    body = render_task(item)
    instr = format_instruction(item, variant)
    return f"{body}\n\n{instr}" if instr else body


def build_messages(item: dict, variant: str = DEFAULT_VARIANT, system_prompt: str | None = None) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": build_user_prompt(item, variant)})
    return msgs


def build_training_example(item: dict, variant: str = DEFAULT_VARIANT, system_prompt: str | None = None) -> dict:
    """TRL conversational prompt-completion row: loss only on the completion (the answer)."""
    return {
        "prompt": build_messages(item, variant, system_prompt),
        "completion": [{"role": "assistant", "content": render_target(item, variant)}],
    }
