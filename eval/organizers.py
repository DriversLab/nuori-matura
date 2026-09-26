"""Scoring under the organizers' protocol: the letter the model prefers at the FIRST answer position.

Reproduces the reference exam grader (the hackathon's `scripts/prawko.py`) exactly: one user turn rendered with
the chat template, a single forward pass, the logits at the first answer position restricted to the single-token
ids of the option letters, renormalised over them, argmax. Nothing is generated and no answer format is parsed.

Why it exists: if the official grader works like that, answer-format compliance earns nothing and a target that
depresses the letter logits ("Odpowiedź: X") would cost points. This module lets us MEASURE that. It is
eval-only and additive: no training target, headline number or existing gate rule is computed from it.

Differences from `prawko.py`, all deliberate:
  * items carry the repo item schema (`eval/schema.py`): options are a dict {"A": ..., ...} of 2-8 entries and the
    answer is a letter, so the protocol applies to any option count, not only A/B/C;
  * an item's `context` (absent in their driving-licence rows) is rendered above the question, as everywhere else;
  * no 768-token prompt guard (our eval items are longer and are never truncated here).
"""
from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict

import torch

from eval.modeling import model_device

# Their exact wording. {letters} is the option-letter list of the item ("A, B albo C" for three options).
ORGANIZER_INSTRUCTION = (
    "Rozwiąż pytanie egzaminacyjne na prawo jazdy w Polsce. Wybierz jedną poprawną odpowiedź. "
    "Odpowiedz wyłącznie literą {letters}."
)
LETTERS = "ABCDEFGH"
MIN_OPTIONS, MAX_OPTIONS = 2, len(LETTERS)
# bump when organizer_prompt's assembly changes: it is hashed into the eval fingerprint (organizer_prompt_sha256)
PROMPT_BUILDER_VERSION = "organizer-prompt-v1"


# ----------------------------------------------------------------------------- prompt


def item_letters(item: dict) -> str:
    """Option letters of an item in order ("ABC"). Schema items always use consecutive letters from A."""
    return "".join(sorted(item.get("options") or {}))


def letter_list(letters: str) -> str:
    """Polish enumeration used in their instruction: 'A, B albo C' / 'A, B, C albo D' / 'A albo B'."""
    letters = list(letters)
    if not letters:
        raise ValueError("letter_list needs at least one letter")
    if len(letters) == 1:
        return letters[0]
    return ", ".join(letters[:-1]) + " albo " + letters[-1]


def organizer_prompt(item: dict, order=None, instruction: str | None = None) -> str:
    """Their user turn: instruction, blank line, question, newline, "L. option" lines.

    `order` is their option order: display position i shows the item's option number order[i] (default identity;
    (1, 2, 0) is the rotation their eval uses). Question and option texts are passed through VERBATIM -- their
    rows carry leading spaces in some options, so stripping would change the prompt bytes.
    """
    letters = item_letters(item)
    if not MIN_OPTIONS <= len(letters) <= MAX_OPTIONS:
        raise ValueError(f"[{item.get('id', '?')}] organizer prompt needs {MIN_OPTIONS}-{MAX_OPTIONS} options")
    order = list(range(len(letters))) if order is None else list(order)
    if sorted(order) != list(range(len(letters))):
        raise ValueError(f"[{item.get('id', '?')}] order {order} is not a permutation of the {len(letters)} options")
    head = (instruction or ORGANIZER_INSTRUCTION).format(letters=letter_list(letters))
    question = f"{item['context']}\n\n{item['question']}" if item.get("context") else item["question"]
    options = "\n".join(f"{LETTERS[i]}. {item['options'][letters[j]]}" for i, j in enumerate(order))
    return f"{head}\n\n{question}\n{options}"


def organizer_prompt_sha256(instruction: str | None = None) -> str:
    """sha256-16 of the instruction template + prompt-builder version: a prompt change invalidates cached evals."""
    payload = f"{PROMPT_BUILDER_VERSION}\0{instruction or ORGANIZER_INSTRUCTION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rotate_item(item: dict, shift: int = 1) -> dict:
    """Copy of an mc item with the options rotated left by `shift` and the answer letter remapped.

    shift=1 is their rotated eval (order (1, 2, 0)): A shows the old B, B the old C, C the old A. A model that
    reads the options loses nothing; a model that learned a letter position bias does.

    Items whose question or options refer to an option by letter ("tylko A") are rotated like any other: the
    rotated score is a diagnostic, never a gate number.
    """
    if item.get("type") != "mc":
        raise ValueError(f"[{item.get('id', '?')}] rotate_item needs an mc item, got type {item.get('type')!r}")
    letters = item_letters(item)
    n = len(letters)
    if n < MIN_OPTIONS:
        raise ValueError(f"[{item.get('id', '?')}] rotate_item needs >= {MIN_OPTIONS} options")
    shift %= n
    options = {letters[i]: item["options"][letters[(i + shift) % n]] for i in range(n)}
    answer = letters[(letters.index(item["answer"]) - shift) % n]
    return {**item, "options": options, "answer": answer}


# ----------------------------------------------------------------------------- scoring


def letter_token_ids(tokenizer, letters: str) -> list[int]:
    """Token id of each letter on its own (their guard: the protocol needs single-token labels)."""
    ids = [tokenizer(letter, add_special_tokens=False)["input_ids"] for letter in letters]
    bad = [letter for letter, x in zip(letters, ids) if len(x) != 1]
    if bad:
        raise ValueError(f"organizer scoring needs single-token letters; {bad} are not single tokens")
    return [x[0] for x in ids]


def render_organizer_chat(tokenizer, prompt: str) -> str:
    """Their rendering: one user turn, generation prompt, thinking off when the template understands it."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # template / tokenizer without an enable_thinking switch
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_organizer(
    model,
    tokenizer,
    items: list[dict],
    *,
    batch_size: int = 8,
    instruction: str | None = None,
    rotate: int = 0,
) -> list[dict]:
    """One record per scorable "mc" item, in input order:

    {id, subject, type, letters, answer, prediction, correct, probabilities {letter: renormalised p},
     abc_mass (the letters' share of the full next-token distribution), rotate}

    Items that are not "mc", have an unusable option count or whose letters are not single tokens are skipped
    (the caller counts them). `rotate` > 0 scores rotate_item(item, rotate) instead, answer included.
    """
    rows: list[tuple[dict, str, list[int]]] = []
    prompts: list[str] = []
    unusable: set[str] = set()
    for item in items:
        if item.get("type") != "mc":
            continue
        letters = item_letters(item)
        if not MIN_OPTIONS <= len(letters) <= MAX_OPTIONS or item.get("answer") not in letters:
            unusable.add(f"{item.get('id', '?')}: {len(letters)} options")
            continue
        try:
            ids = letter_token_ids(tokenizer, letters)
        except ValueError as exc:
            unusable.add(str(exc))
            continue
        row = rotate_item(item, rotate) if rotate else item
        prompts.append(organizer_prompt(row, instruction=instruction))
        rows.append((row, letters, ids))
    for message in sorted(unusable):
        warnings.warn(f"organizer scoring skipped an item ({message})", stacklevel=2)

    records: list[dict] = []
    if not rows:
        return records
    tokenizer.padding_side = "left"
    device = model_device(model)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            texts = [render_organizer_chat(tokenizer, p) for p in prompts[start : start + batch_size]]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            # only the final hidden state is projected (logits_to_keep=1), exactly like their logits()
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                use_cache=False,
                logits_to_keep=1,
            ).logits[:, -1, :].float()
            full = logits.softmax(-1)
            for i, (item, letters, ids) in enumerate(chunk):
                probabilities = logits[i, ids].softmax(-1).tolist()
                prediction = letters[max(range(len(letters)), key=probabilities.__getitem__)]  # ties -> first letter
                records.append({
                    "id": item["id"],
                    "subject": item.get("subject"),
                    "type": item["type"],
                    "letters": letters,
                    "answer": item["answer"],
                    "prediction": prediction,
                    "correct": prediction == item["answer"],
                    "probabilities": dict(zip(letters, probabilities)),
                    "abc_mass": full[i, ids].sum().item(),
                    "rotate": int(rotate),
                })
            del logits, full
    return records


def summarize_organizer(records: list[dict]) -> dict:
    """{n, correct, accuracy, mean_correct_probability, mean_abc_mass, per_subject {subject: {n, correct, acc}}}."""

    def stats(rows: list[dict]) -> dict:
        correct = sum(1 for r in rows if r["correct"])
        return {"n": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4) if rows else 0.0}

    def mean(values) -> float:
        values = list(values)
        return round(sum(values) / len(values), 4) if values else 0.0

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subject[str(r.get("subject"))].append(r)
    return {
        **stats(records),
        "mean_correct_probability": mean(r["probabilities"].get(r["answer"], 0.0) for r in records),
        "mean_abc_mass": mean(r["abc_mass"] for r in records),
        "per_subject": {s: stats(rows) for s, rows in sorted(by_subject.items())},
    }
