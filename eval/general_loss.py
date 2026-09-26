"""Mean NLL of reference assistant responses on held-out general Polish instruction pairs (forgetting check)."""
from __future__ import annotations

import math

from eval.loglik import Request, score_tokens


def encode_pair(tokenizer, messages: list[dict], max_length: int) -> tuple[list[int], int] | None:
    """(full_ids truncated to max_length, response_start) or None when the pair cannot be scored.

    Tokenized exactly like TRL prompt-completion rows, so the response includes <|im_end|>. Unscorable: last
    message not from the assistant, prompt not a token prefix of the full conversation, or no response token
    left within max_length.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages, return_dict=False)
    if full[: len(prompt)] != prompt or len(prompt) >= min(len(full), max_length):
        return None
    return full[:max_length], len(prompt)


def general_nll(model, tokenizer, pairs: list[dict], *, batch_size: int, max_length: int) -> dict:
    """{"nll": token-weighted mean NLL over response tokens, "tokens", "n": scored pairs, "skipped"}.

    nll is NaN when nothing could be scored.
    """
    requests: list[Request] = []
    skipped = 0
    for pair in pairs:
        enc = encode_pair(tokenizer, pair["messages"], max_length)
        if enc is None:
            skipped += 1
            continue
        ids, start = enc
        requests.append((ids[:-1], list(range(start - 1, len(ids) - 1)), ids[start:]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scores = score_tokens(model, requests, batch_size=batch_size, pad_token_id=pad_id)
    tokens = sum(len(s) for s in scores)
    total = -sum(s.double().sum().item() for s in scores)
    return {
        "nll": total / tokens if tokens else math.nan,
        "tokens": tokens,
        "n": len(requests),
        "skipped": skipped,
    }
