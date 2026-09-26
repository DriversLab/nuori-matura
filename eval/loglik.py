"""Loglikelihood scoring of single-choice items, lm-eval MMLU style (the LLMzSzŁ benchmark setup).

The raw prompt (no chat template) ends with "Prawidłowa odpowiedź:"; each option is scored as the summed log-prob
of the continuation " A", " B", ... and the highest wins. Independent of answer formatting, so it shows whether
fine-tuning changed knowledge rather than output style.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from eval.modeling import model_device

LLMZSZL_TEMPLATE = "Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru\n\n{question}\n{options}\nPrawidłowa odpowiedź:"

# (input_ids, positions, targets): score targets[j] under the logits at positions[j]
Request = tuple[list[int], list[int], list[int]]


def render_loglik_prompt(item: dict) -> str:
    question = item["question"].strip()
    if item.get("context"):
        question = f"{item['context'].strip()}\n\n{question}"
    options = "\n".join(f"{k}. {v.strip()}" for k, v in item["options"].items())
    return LLMZSZL_TEMPLATE.format(question=question, options=options)


def leading_special_ids(tokenizer) -> list[int]:
    """Special tokens the tokenizer puts before plain text (Llama/Mistral/Bielik: [BOS]; GPT-2 style: []).

    Trailing special tokens (an EOS some tokenizers append) are deliberately not included.
    """
    plain = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x")["input_ids"]
    for k in range(len(full) - len(plain) + 1):
        if full[k : k + len(plain)] == plain:
            return list(full[:k])
    return []


def encode_continuation(
    tokenizer, context: str, continuation: str, lead: list[int] | None = None
) -> tuple[list[int], list[int]]:
    """(context_ids, continuation_ids). The context starts with the tokenizer's leading special tokens (BOS).

    Continuation ids are the suffix of tokenize(context + continuation) when tokenize(context) is its prefix;
    encoding " A" on its own is wrong for sentencepiece tokenizers (Bielik: ['▁', '▁A'] instead of ['▁A']),
    so that is only the fallback when the boundary merges tokens.
    """
    lead = leading_special_ids(tokenizer) if lead is None else lead
    ctx = tokenizer(context, add_special_tokens=False)["input_ids"]
    whole = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    if len(whole) > len(ctx) and whole[: len(ctx)] == ctx:
        return lead + ctx, whole[len(ctx) :]
    return lead + ctx, tokenizer(continuation, add_special_tokens=False)["input_ids"]


def score_tokens(model, requests: list[Request], *, batch_size: int, pad_token_id: int = 0) -> list[torch.Tensor]:
    """Teacher-forced float32 log-probs of each request's targets, one CPU tensor per request.

    Batches are longest-first and right-padded, so real tokens keep their positions and are unaffected by padding.
    """
    device = model_device(model)
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i][0]))
    results: list[torch.Tensor] = [torch.empty(0)] * len(requests)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            width = len(requests[idx[0]][0])
            input_ids = torch.full((len(idx), width), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                ids, positions, targets = requests[i]
                if len(positions) != len(targets) or (positions and not 0 <= min(positions) <= max(positions) < len(ids)):
                    raise ValueError(f"request {i}: positions/targets do not fit its {len(ids)} input tokens")
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = 1
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), use_cache=False
            ).logits
            for row, i in enumerate(idx):
                _, positions, targets = requests[i]
                pos = torch.tensor(positions, dtype=torch.long, device=logits.device)
                tgt = torch.tensor(targets, dtype=torch.long, device=logits.device)
                logprobs = logits[row, pos].float().log_softmax(-1)
                results[i] = logprobs.gather(-1, tgt[:, None]).squeeze(-1).cpu()
            del logits
    return results


def score_mc_loglik(model, tokenizer, items: list[dict], *, batch_size: int) -> list[dict]:
    """One record per "mc" item: {id, subject, gold, pred, correct, logprobs: {letter: summed log-prob}}.

    Options whose scoring inputs are identical (single-token continuations) share one forward pass.
    """
    requests: list[Request] = []
    by_input: dict[tuple[int, ...], int] = {}
    spans: list[tuple[dict, list[tuple[str, int, int, int]]]] = []
    lead = leading_special_ids(tokenizer)
    for item in (it for it in items if it["type"] == "mc"):
        context = render_loglik_prompt(item)
        item_spans = []
        for letter in item["options"]:
            ctx, cont = encode_continuation(tokenizer, context, " " + letter, lead)
            if not cont:
                raise ValueError(f"[{item['id']}] continuation ' {letter}' encodes to no tokens")
            seq = ctx + cont
            key = tuple(seq[:-1])
            if key not in by_input:
                by_input[key] = len(requests)
                requests.append((list(key), [], []))
            req = by_input[key]
            _, positions, targets = requests[req]
            item_spans.append((letter, req, len(targets), len(cont)))
            positions.extend(range(len(ctx) - 1, len(seq) - 1))
            targets.extend(cont)
        spans.append((item, item_spans))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scores = score_tokens(model, requests, batch_size=batch_size, pad_token_id=pad_id)
    records = []
    for item, item_spans in spans:
        logprobs = {letter: scores[req][off : off + n].sum().item() for letter, req, off, n in item_spans}
        pred = max(logprobs, key=logprobs.__getitem__)
        records.append({
            "id": item["id"],
            "subject": item["subject"],
            "gold": item["answer"],
            "pred": pred,
            "correct": pred == item["answer"],
            "logprobs": logprobs,
        })
    return records


def summarize_loglik(records: list[dict]) -> dict:
    def acc(rows: list[dict]) -> float:
        return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else 0.0

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subject[r["subject"]].append(r)
    return {
        "n": len(records),
        "acc": acc(records),
        "per_subject": {s: {"n": len(rows), "acc": acc(rows)} for s, rows in sorted(by_subject.items())},
    }
