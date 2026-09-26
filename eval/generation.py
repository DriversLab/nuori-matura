"""Batched greedy generation for chat prompts (the answers that get graded)."""
from __future__ import annotations

import torch

from eval.modeling import model_device


def stop_token_ids(model, tokenizer) -> list[int]:
    """Union of the tokenizer eos and the model's generation_config eos ids (Bielik: [<|im_end|>, </s>])."""
    ids: list[int] = []
    gc = getattr(model, "generation_config", None)
    for value in (tokenizer.eos_token_id, getattr(gc, "eos_token_id", None)):
        for tid in value if isinstance(value, (list, tuple)) else [value]:
            if tid is not None and tid not in ids:
                ids.append(int(tid))
    return ids


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Exact prompt string the model is conditioned on (chat template + assistant header)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_responses(
    model,
    tokenizer,
    messages_list: list[list[dict]],
    *,
    max_new_tokens: int,
    batch_size: int,
    desc: str | None = None,
) -> list[str]:
    """Greedy responses (new tokens only, special tokens skipped, stripped), in input order.

    The rendered template already contains BOS, so it is tokenized with add_special_tokens=False
    (plain tokenization gives a double BOS with Bielik). Prompts are batched longest-first with left padding.
    `desc` labels an optional progress bar (none when None).
    """
    if not messages_list:
        return []
    texts = [render_prompt(tokenizer, m) for m in messages_list]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    eos_ids = stop_token_ids(model, tokenizer)
    if not eos_ids:
        raise ValueError("no eos token id on the tokenizer or model.generation_config; generation would never stop")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    tokenizer.padding_side = "left"
    device = model_device(model)
    outputs = [""] * len(texts)

    from tqdm.auto import tqdm

    with torch.inference_mode(), tqdm(total=len(texts), desc=desc, disable=desc is None, leave=False) as bar:
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
            )
            new_tokens = out[:, input_ids.shape[1] :]
            for i, row in zip(idx, new_tokens):
                outputs[i] = tokenizer.decode(row, skip_special_tokens=True).strip()
            bar.update(len(idx))
    return outputs
