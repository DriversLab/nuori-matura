"""Eval runtime: generation, loglik, general NLL, runner orchestration/caching and scripts/run_eval.py.

Fast tests use a char-level fake tokenizer (Bielik-like: the chat template renders the BOS string) and a
deterministic causal fake LM. Slow tests load HuggingFaceTB/SmolLM2-135M-Instruct and the Bielik-11B tokenizer.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eval import general_loss, generation, loglik, modeling, runner
from eval.prompts import build_messages
from eval.schema import load_items, read_jsonl, write_jsonl
from train.config import ROOT, load_config, run_dir

PAD, BOS, EOS = 0, 1, 2
OFFSET = 3
VOCAB = 512
MERGED_COLON_SPACE = VOCAB - 1
SPECIALS = {"<pad>": PAD, "<s>": BOS, "<|end|>": EOS}

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BIELIK_TOKENIZER = "speakleash/Bielik-11B-v2.3-Instruct"


# ----------------------------------------------------------------------------- fakes


class FakeTokenizer:
    """Char-level tokenizer. `merge_colon_space` merges ": " into one token (a boundary-merging tokenizer)."""

    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS

    def __init__(self, *, add_bos: bool = True, add_eos: bool = False, merge_colon_space: bool = False):
        self.add_bos, self.add_eos, self.merge = add_bos, add_eos, merge_colon_space
        self.padding_side = "right"

    def encode_text(self, text: str) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            special = next((s for s in SPECIALS if text.startswith(s, i)), None)
            if special:
                ids.append(SPECIALS[special])
                i += len(special)
            elif self.merge and text.startswith(": ", i):
                ids.append(MERGED_COLON_SPACE)
                i += 2
            else:
                assert ord(text[i]) + OFFSET < MERGED_COLON_SPACE, f"char {text[i]!r} outside fake vocab"
                ids.append(ord(text[i]) + OFFSET)
                i += 1
        return ids

    def _with_specials(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = self.encode_text(text)
        if add_special_tokens:
            ids = ([BOS] if self.add_bos else []) + ids + ([EOS] if self.add_eos else [])
        return ids

    def __call__(self, text, *, add_special_tokens: bool = True, return_tensors=None, padding: bool = False):
        seqs = [self._with_specials(t, add_special_tokens) for t in ([text] if isinstance(text, str) else text)]
        if return_tensors is None:
            return {"input_ids": seqs[0] if isinstance(text, str) else seqs}
        width = max(map(len, seqs))
        rows, masks = [], []
        for s in seqs:
            pad = width - len(s)
            left = self.padding_side == "left"
            rows.append([PAD] * pad + s if left else s + [PAD] * pad)
            masks.append([0] * pad + [1] * len(s) if left else [1] * len(s) + [0] * pad)
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        text = "<s>" + "".join(f"[{m['role']}]{m['content']}<|end|>\n" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, return_dict=True):
        text = self.render(messages, add_generation_prompt)
        if not tokenize:
            return text
        assert return_dict is False, "callers must ask for a plain id list"
        return self.encode_text(text)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = []
        for t in (ids.tolist() if isinstance(ids, torch.Tensor) else ids):
            if t in SPECIALS.values():
                out.append("" if skip_special_tokens else next(k for k, v in SPECIALS.items() if v == t))
            else:
                out.append(": " if t == MERGED_COLON_SPACE else chr(t - OFFSET))
        return "".join(out)


class GenPromptQuirkTokenizer(FakeTokenizer):
    """Generation prompt ends with a newline that the rendered full conversation does not have."""

    def render(self, messages, add_generation_prompt):
        text = super().render(messages, False)
        return text + ("[assistant]\n" if add_generation_prompt else "")


class FakeLM(torch.nn.Module):
    """Deterministic causal LM: logits at position t depend only on tokens <= t (so right padding cannot leak).

    generate() answers via `responder(prompt_text)` and records every call for inspection.
    """

    def __init__(self, tokenizer: FakeTokenizer, responder=None, generation_eos=(EOS,)):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.tok, self.responder = tokenizer, responder
        self.generation_config = SimpleNamespace(eos_token_id=list(generation_eos))
        self.forward_batches: list[tuple[int, int]] = []
        self.generate_calls: list[dict] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.forward_batches.append(tuple(input_ids.shape))
        cumulative = torch.cumsum(input_ids.to(torch.float64), dim=1)
        freqs = torch.arange(VOCAB, dtype=torch.float64) * 0.013 + 0.1
        return SimpleNamespace(logits=(3.0 * torch.sin(cumulative[..., None] * freqs)).to(torch.float32))

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample, eos_token_id, pad_token_id):
        self.generate_calls.append({
            "input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(), "max_new_tokens": max_new_tokens,
            "do_sample": do_sample, "eos_token_id": eos_token_id, "pad_token_id": pad_token_id,
        })
        new_rows = []
        for ids, mask in zip(input_ids.tolist(), attention_mask.tolist()):
            prompt = self.tok.decode([t for t, m in zip(ids, mask) if m])
            new_rows.append((self.tok.encode_text(self.responder(prompt)) + [EOS])[:max_new_tokens])
        width = max(map(len, new_rows))
        new = torch.tensor([row + [pad_token_id] * (width - len(row)) for row in new_rows], dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


def reference_logprob(model: FakeLM, ids: list[int], first_target: int) -> float:
    """sum_t log p(ids[t] | ids[:t]) for t >= first_target, one unpadded sequence."""
    logprobs = model(torch.tensor([ids])).logits[0].float().log_softmax(-1)
    return sum(logprobs[t - 1, ids[t]].item() for t in range(first_target, len(ids)))


# ----------------------------------------------------------------------------- generation


def test_stop_token_ids_union_without_duplicates():
    tok = FakeTokenizer()
    assert generation.stop_token_ids(FakeLM(tok, generation_eos=(EOS, 7)), tok) == [EOS, 7]
    model = FakeLM(tok)
    model.generation_config = SimpleNamespace(eos_token_id=5)
    assert generation.stop_token_ids(model, tok) == [EOS, 5]
    model.generation_config = None
    assert generation.stop_token_ids(model, tok) == [EOS]


def test_generate_responses_order_left_padding_single_bos_and_decoding():
    tok = FakeTokenizer()

    def responder(prompt: str) -> str:
        n = re.search(r"Q(\d+)", prompt).group(1)
        return "x" * 40 if n == "3" else f"  Odpowiedź: {n}\n "

    model = FakeLM(tok, responder, generation_eos=(EOS, 7))
    lengths = [5, 40, 12, 25, 1]
    messages = [[{"role": "user", "content": f"Q{i} " + "z" * n}] for i, n in enumerate(lengths)]
    out = generation.generate_responses(model, tok, messages, max_new_tokens=16, batch_size=2)

    assert out == ["Odpowiedź: 0", "Odpowiedź: 1", "Odpowiedź: 2", "x" * 16, "Odpowiedź: 4"]
    assert tok.padding_side == "left"
    assert [len(c["input_ids"]) for c in model.generate_calls] == [2, 2, 1]
    widths = [len(c["input_ids"][0]) for c in model.generate_calls]
    assert widths == sorted(widths, reverse=True), "batches must be longest-first"
    for call in model.generate_calls:
        assert call["do_sample"] is False and call["max_new_tokens"] == 16
        assert call["eos_token_id"] == [EOS, 7] and call["pad_token_id"] == PAD
        for ids, mask in zip(call["input_ids"], call["attention_mask"]):
            n_pad = mask.index(1)
            assert mask == [0] * n_pad + [1] * (len(mask) - n_pad), "padding must be on the left"
            assert ids[:n_pad] == [PAD] * n_pad
            assert ids[n_pad] == BOS and ids[n_pad + 1] != BOS, "exactly one BOS (add_special_tokens=False)"
    assert generation.generate_responses(model, tok, [], max_new_tokens=4, batch_size=2) == []


# ----------------------------------------------------------------------------- loglik


def _mc(item_id: str, question: str, options: dict, answer: str, subject: str = "historia") -> dict:
    return {"id": item_id, "subject": subject, "type": "mc", "question": question, "options": options, "answer": answer}


def test_leading_special_ids_and_continuation_encoding():
    assert loglik.leading_special_ids(FakeTokenizer()) == [BOS]
    assert loglik.leading_special_ids(FakeTokenizer(add_bos=False)) == []
    assert loglik.leading_special_ids(FakeTokenizer(add_eos=True)) == [BOS], "a trailing EOS must not leak into the context"

    tok = FakeTokenizer()
    ctx, cont = loglik.encode_continuation(tok, "Odpowiedź:", " A")
    assert ctx == [BOS] + tok.encode_text("Odpowiedź:") and cont == tok.encode_text(" A")

    ctx, cont = loglik.encode_continuation(FakeTokenizer(add_eos=True), "Odpowiedź:", " A")
    assert ctx[0] == BOS and EOS not in ctx

    merging = FakeTokenizer(merge_colon_space=True)
    ctx, cont = loglik.encode_continuation(merging, "Odpowiedź:", " A")  # ": " merges across the boundary
    assert ctx == [BOS] + merging.encode_text("Odpowiedź:")
    assert cont == merging.encode_text(" A") and MERGED_COLON_SPACE not in cont


def test_render_loglik_prompt_uses_llmzszl_template():
    item = _mc("h-1", "Kto?", {"A": "Anna ", "B": "Bolek"}, "B")
    item["context"] = "Tekst."
    assert loglik.render_loglik_prompt(item) == (
        "Przykładowe pytanie egzaminacyjne, test jednokrotnego wyboru\n\nTekst.\n\nKto?\nA. Anna\nB. Bolek\nPrawidłowa odpowiedź:"
    )


def test_score_mc_loglik_matches_reference_and_is_batch_invariant():
    tok = FakeTokenizer()
    items = [
        _mc("m-1", "Kto założył Kraków?", {"A": "Krak", "B": "Wanda", "C": "Lech", "D": "Popiel"}, "A"),
        {"id": "t-1", "subject": "biologia", "type": "tf", "question": "Oceń.", "statements": ["x"], "answer": ["P"]},
        _mc("m-2", "Ile?", {"A": "dwa", "B": "trzy", "C": "pięć"}, "C", subject="matematyka"),
        _mc("m-3", "Gdzie leży Gniezno w Wielkopolsce, a gdzie Poznań?", {"A": "tu", "B": "tam"}, "B"),
    ]
    model = FakeLM(tok)
    records = loglik.score_mc_loglik(model, tok, items, batch_size=2)
    assert [r["id"] for r in records] == ["m-1", "m-2", "m-3"], "only mc items, input order"
    assert sum(b for b, _ in model.forward_batches) == 3, "options of one item share a single forward row"

    for item, rec in zip([items[0], items[2], items[3]], records):
        prompt = loglik.render_loglik_prompt(item)
        expected = {}
        for letter in item["options"]:
            ctx, cont = loglik.encode_continuation(tok, prompt, " " + letter)
            expected[letter] = reference_logprob(model, ctx + cont, len(ctx))
        assert rec["logprobs"].keys() == expected.keys()
        for letter in expected:
            assert rec["logprobs"][letter] == pytest.approx(expected[letter], abs=1e-4)
        assert rec["pred"] == max(expected, key=expected.get)
        assert rec["correct"] == (rec["pred"] == item["answer"]) and rec["gold"] == item["answer"]
        assert rec["subject"] == item["subject"]

    one_by_one = loglik.score_mc_loglik(model, tok, items, batch_size=1)
    for a, b in zip(records, one_by_one):
        assert a["pred"] == b["pred"]
        assert all(a["logprobs"][k] == pytest.approx(b["logprobs"][k], abs=1e-5) for k in a["logprobs"])


def test_summarize_loglik():
    records = [
        {"subject": "historia", "correct": True}, {"subject": "historia", "correct": False},
        {"subject": "fizyka", "correct": True},
    ]
    assert loglik.summarize_loglik(records) == {
        "n": 3, "acc": 0.6667, "per_subject": {"fizyka": {"n": 1, "acc": 1.0}, "historia": {"n": 2, "acc": 0.5}},
    }
    assert loglik.summarize_loglik([]) == {"n": 0, "acc": 0.0, "per_subject": {}}


# ----------------------------------------------------------------------------- general NLL


def _pair(user: str, assistant: str, system: str | None = None) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": assistant},
    ]
    return {"id": user[:8], "source": "test", "messages": msgs}


def test_general_nll_matches_reference_truncates_and_skips():
    tok = FakeTokenizer()
    model = FakeLM(tok)
    max_length = 80
    pairs = [
        _pair("Ile to 2+2?", "Cztery."),
        _pair("Opisz Wisłę.", "Wisła to najdłuższa rzeka w Polsce. " * 3, system="Bądź zwięzły."),  # truncated
        {"id": "no-assistant", "messages": [{"role": "user", "content": "Halo?"}]},  # skipped
        _pair("z" * 100, "za długi prompt"),  # prompt >= max_length -> skipped
    ]
    result = general_loss.general_nll(model, tok, pairs, batch_size=3, max_length=max_length)

    total, tokens = 0.0, 0
    for pair in pairs[:2]:
        prompt = tok.apply_chat_template(pair["messages"][:-1], add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template(pair["messages"], return_dict=False)[:max_length]
        total -= reference_logprob(model, full, len(prompt))
        tokens += len(full) - len(prompt)
    assert result["n"] == 2 and result["skipped"] == 2 and result["tokens"] == tokens
    assert result["nll"] == pytest.approx(total / tokens, rel=1e-5)
    assert general_loss.encode_pair(tok, pairs[1]["messages"], max_length)[0][-1] != EOS, "long response was truncated"
    first = general_loss.encode_pair(tok, pairs[0]["messages"], max_length)
    assert tok.decode(first[0][first[1]:]) == "Cztery.<|end|>\n", "response tokens include the end-of-turn token"

    one_by_one = general_loss.general_nll(model, tok, pairs, batch_size=1, max_length=max_length)
    assert one_by_one["nll"] == pytest.approx(result["nll"], rel=1e-6) and one_by_one["tokens"] == tokens


def test_general_nll_skips_prefix_mismatch_and_reports_nan_when_nothing_scorable():
    tok = GenPromptQuirkTokenizer()
    pairs = [_pair("Ile to 2+2?", "Cztery.")]
    assert general_loss.encode_pair(tok, pairs[0]["messages"], 512) is None
    result = general_loss.general_nll(FakeLM(tok), tok, pairs, batch_size=2, max_length=512)
    assert result["n"] == 0 and result["tokens"] == 0 and result["skipped"] == 1 and math.isnan(result["nll"])


# ----------------------------------------------------------------------------- runner fixtures

MINI = [
    {"id": "mc-1", "subject": "historia", "type": "mc", "question": "Pytanie mc-1: kto?",
     "options": {"A": "Anna", "B": "Bolesław", "C": "Celina"}, "answer": "B", "points": 1, "source": "authored"},
    {"id": "tf-1", "subject": "biologia", "type": "tf", "question": "Pytanie tf-1: oceń.", "statements": ["s1", "s2", "s3"],
     "answer": ["P", "F", "P"], "points": 2, "partial_credit": [[2, 1]], "source": "authored"},
    {"id": "short-1", "subject": "historia", "type": "short", "question": "Pytanie short-1: kto?",
     "answer": ["Mieszko I", "Mieszko"], "points": 1, "source": "authored"},
    {"id": "num-1", "subject": "fizyka", "type": "numeric", "question": "Pytanie num-1: ile?", "answer": 3.5,
     "points": 1, "source": "authored"},
]
EXT = [
    {"id": "ext-1", "subject": "matematyka", "type": "mc", "question": "Pytanie ext-1: ile?",
     "options": {"A": "1", "B": "2"}, "answer": "A", "source": "ext"},
    {"id": "ext-2", "subject": "fizyka", "type": "mc", "question": "Pytanie ext-2: co?",
     "options": {"A": "x", "B": "y", "C": "z"}, "answer": "B", "source": "ext"},
]
CANONICAL_OUT = {
    "mc-1": "Odpowiedź: B",                 # strict pass
    "tf-1": "Odpowiedź: P, F, F",           # 2/3 parts -> partial credit 1 of 2
    "short-1": "**Odpowiedź:** Mieszko I",  # strict fail (markdown), lenient pass
    "num-1": "Odpowiedź: 3,5",              # strict pass
    "ext-1": "Odpowiedź: A",
    "ext-2": "Odpowiedź: C",
}
PAYLOAD_OUT = {"mc-1": "B", "tf-1": "P, F, P", "short-1": "Mieszko I jako pierwszy", "num-1": "3.5 m"}


class EvalHarness:
    """Monkeypatched load / generate / loglik / NLL around the real runner, grader and file IO."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp = tmp_path
        self.sets_dir = tmp_path / "sets"
        write_jsonl(MINI, self.sets_dir / "mini.jsonl")
        write_jsonl(EXT, self.sets_dir / "ext.jsonl")
        self.general = tmp_path / "general_heldout.jsonl"
        write_jsonl([_pair(f"Pytanie ogólne {i}?", f"Odpowiedź ogólna {i}.") for i in range(5)], self.general)
        self.loads: list[tuple] = []
        self.generate_calls: list[dict] = []
        self.nll_pairs: list[int] = []
        self.frees = 0
        self.load_dtype = "float32"
        monkeypatch.setattr(modeling, "detect_device", lambda: "cpu")
        monkeypatch.setattr(modeling, "load_model_and_tokenizer", self.load)
        monkeypatch.setattr(modeling, "free_model", self.free)
        monkeypatch.setattr(generation, "generate_responses", self.generate)
        monkeypatch.setattr(loglik, "score_mc_loglik", self.score_loglik)
        monkeypatch.setattr(general_loss, "general_nll", self.nll)

    def cfg(self, **overrides) -> dict:
        base = {
            "model.base": "org/Fake-Base-7B",
            "model.dtype": "float32",
            "paths.results_root": str(self.tmp / "results"),
            "data.general_heldout": str(self.general),
            "eval.sets": [
                {"path": str(self.sets_dir / "mini.jsonl"), "variants": ["canonical", "payload_only", "reason_then_answer"], "loglik": True},
                {"path": str(self.sets_dir / "ext.jsonl"), "name": "ext_mc", "variants": ["canonical"], "loglik": True, "optional": True},
            ],
            "eval.primary_set": "mini",
            "eval.primary_variant": "canonical",
            "eval.system_prompt": "Jesteś pomocnym asystentem.",
            "eval.general_loss_max_items": 3,
            "eval.batch_size": 4,
        }
        return load_config(None, {**base, **overrides})

    def load(self, source, **kwargs):
        self.loads.append((source, kwargs))
        info = {"device": "cpu", "dtype": self.load_dtype, "quantization": kwargs.get("quantization"), "attn_implementation": "sdpa"}
        return SimpleNamespace(_matura_load_info=info), FakeTokenizer()

    def free(self, model):
        self.frees += 1

    def generate(self, model, tokenizer, messages_list, *, max_new_tokens, batch_size, desc=None):
        self.generate_calls.append({"desc": desc, "n": len(messages_list), "max_new_tokens": max_new_tokens, "batch_size": batch_size})
        outputs = []
        for msgs in messages_list:
            assert msgs[0] == {"role": "system", "content": "Jesteś pomocnym asystentem."}
            user = msgs[-1]["content"]
            item_id = re.search(r"Pytanie (\S+):", user).group(1)
            if "Podaj wyłącznie odpowiedź" in user:
                outputs.append(PAYLOAD_OUT[item_id])
            elif "Najpierw krótko uzasadnij" in user:
                outputs.append(f"Krótkie uzasadnienie.\n{CANONICAL_OUT[item_id]}")
            else:
                outputs.append(CANONICAL_OUT[item_id])
        return outputs

    @staticmethod
    def score_loglik(model, tokenizer, items, *, batch_size):
        return [
            {"id": it["id"], "subject": it["subject"], "gold": it["answer"], "pred": "A", "correct": it["answer"] == "A",
             "logprobs": {k: -1.0 - i for i, k in enumerate(it["options"])}}
            for it in items if it["type"] == "mc"
        ]

    def nll(self, model, tokenizer, pairs, *, batch_size, max_length):
        self.nll_pairs.append(len(pairs))
        return {"nll": 1.25, "tokens": 10 * len(pairs), "n": len(pairs), "skipped": 0}


@pytest.fixture
def harness(tmp_path, monkeypatch) -> EvalHarness:
    return EvalHarness(tmp_path, monkeypatch)


LOAD_INFO = {"device": "cpu", "dtype": "float32", "quantization": "none", "attn_implementation": "sdpa", **runner.environment_info("cpu")}


# ----------------------------------------------------------------------------- eval sets / fingerprint


def test_eval_sets_from_cfg_normalizes_and_skips_missing_optional(harness):
    cfg = harness.cfg(**{"eval.sets": [
        str(harness.sets_dir / "mini.jsonl"),
        {"path": str(harness.sets_dir / "ext.jsonl"), "variants": "bare", "loglik": True, "optional": True},
        {"path": str(harness.sets_dir / "nope.jsonl"), "optional": True},
    ], "eval.primary_variant": "payload_only"})
    with pytest.warns(UserWarning, match="optional eval set 'nope' skipped"):
        sets = runner.eval_sets_from_cfg(cfg)
    assert sets == [
        {"name": "mini", "path": str(harness.sets_dir / "mini.jsonl"), "variants": ["payload_only"], "loglik": False, "optional": False},
        {"name": "ext", "path": str(harness.sets_dir / "ext.jsonl"), "variants": ["bare"], "loglik": True, "optional": True},
    ]


def test_base_config_eval_sets_resolve():
    sets = runner.eval_sets_from_cfg(load_config())
    assert sets[0] == {"name": "heldout", "path": "data/eval/heldout.jsonl", "variants": ["canonical", "bare", "payload_only"],
                       "loglik": True, "optional": False}


@pytest.mark.parametrize("entry, error", [
    ({"path": "data/eval/does_not_exist.jsonl"}, FileNotFoundError),
    ({"path": "data/eval/heldout.jsonl", "variants": ["canonical", "made_up"]}, ValueError),
])
def test_eval_sets_from_cfg_rejects_bad_entries(entry, error):
    with pytest.raises(error):
        runner.eval_sets_from_cfg({"eval": {"sets": [entry]}})


def test_eval_sets_from_cfg_rejects_duplicate_names():
    sets = [{"path": "data/eval/heldout.jsonl"}, {"path": "data/eval/dev.jsonl", "name": "heldout"}]
    with pytest.raises(ValueError, match="duplicate eval set name"):
        runner.eval_sets_from_cfg({"eval": {"sets": sets}})


def test_eval_fingerprint_deterministic_and_sensitive(harness):
    cfg = harness.cfg()
    fp, parts = runner.eval_fingerprint(cfg, LOAD_INFO)
    assert re.fullmatch(r"[0-9a-f]{16}", fp)
    assert runner.eval_fingerprint(copy.deepcopy(cfg), dict(LOAD_INFO)) == (fp, parts)
    assert [s["name"] for s in parts["sets"]] == ["mini", "ext_mc"]
    assert parts["sets"][0]["sha256"] == runner.file_sha256(harness.sets_dir / "mini.jsonl")
    assert parts["general_loss"]["sha256"] == runner.file_sha256(harness.general)
    assert parts["max_new_tokens_reasoning"] == cfg["eval"]["max_new_tokens_reasoning"]

    def fp_of(c: dict, info: dict = LOAD_INFO) -> str:
        return runner.eval_fingerprint(c, info)[0]

    changed = {
        "variants": harness.cfg(**{"eval.sets": [{"path": str(harness.sets_dir / "mini.jsonl"), "variants": ["canonical"], "loglik": True}]}),
        "loglik": harness.cfg(**{"eval.sets": [{**cfg["eval"]["sets"][0], "loglik": False}, *cfg["eval"]["sets"][1:]]}),
        "max_new_tokens": harness.cfg(**{"eval.max_new_tokens": 49}),
        "max_new_tokens_reasoning": harness.cfg(**{"eval.max_new_tokens_reasoning": 100}),
        "system_prompt": harness.cfg(**{"eval.system_prompt": None}),
        "limit": harness.cfg(**{"eval.limit": 2}),
        "general_loss_max_items": harness.cfg(**{"eval.general_loss_max_items": 4}),
        "general_loss_off": harness.cfg(**{"eval.general_loss": False}),
    }
    for what, other in changed.items():
        assert fp_of(other) != fp, what
    for key, value in (("dtype", "bfloat16"), ("quantization", "4bit"), ("device", "cuda"), ("attn_implementation", "flash_attention_2"),
                       ("torch", "2.13.0"), ("transformers", "5.16.0"), ("accelerator", "NVIDIA A100-SXM4-40GB")):
        assert fp_of(cfg, {**LOAD_INFO, key: value}) != fp, key
    assert fp_of(harness.cfg(**{"eval.batch_size": 1})) != fp, "left-padding batch composition changes bf16 outputs"
    assert parts["eval_code_sha256"] == runner.eval_code_sha256() and parts["batch_size"] == 4

    write_jsonl(MINI[:3], harness.sets_dir / "mini.jsonl")
    assert fp_of(cfg) != fp, "items file content"
    write_jsonl(MINI, harness.sets_dir / "mini.jsonl")
    assert fp_of(cfg) == fp
    write_jsonl([_pair("Inne?", "Tak.")], harness.general)
    assert fp_of(cfg) != fp, "general heldout file content"


def test_planned_load_info_uses_eval_quantization(harness):
    cfg = harness.cfg()
    assert runner.planned_load_info(cfg) == LOAD_INFO
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.planned_load_info(harness.cfg(**{"eval.quantization": "4bit"}))


# ----------------------------------------------------------------------------- run_eval orchestration


def test_run_eval_writes_summary_predictions_and_caches(harness):
    sets = harness.cfg()["eval"]["sets"] + [{"path": str(harness.sets_dir / "missing.jsonl"), "optional": True}]
    cfg = harness.cfg(**{"eval.sets": sets})
    with pytest.warns(UserWarning, match="missing.jsonl"):
        summary = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    out = run_dir(cfg, "base")
    assert out == harness.tmp / "results" / "fake-base-7b" / "runs" / "base"

    assert harness.loads == [("org/Fake-Base-7B", {"dtype": "float32", "quantization": "none", "attn_implementation": "auto"})]
    assert harness.frees == 1
    assert {c["desc"]: c["max_new_tokens"] for c in harness.generate_calls} == {
        "mini/canonical": 48, "mini/payload_only": 48, "mini/reason_then_answer": 384, "ext_mc/canonical": 48,
    }

    required = {"run_name", "model", "is_base", "base_model", "timestamp", "fingerprint", "fingerprint_parts", "device", "dtype",
                "quantization", "attn_implementation", "torch", "transformers", "accelerator", "primary_set", "primary_variant",
                "headline", "sets", "general_heldout", "elapsed_s"}
    assert required <= summary.keys()
    assert (summary["run_name"], summary["model"], summary["is_base"], summary["base_model"]) == ("base", "org/Fake-Base-7B", True, "org/Fake-Base-7B")
    assert {k: summary[k] for k in LOAD_INFO} == LOAD_INFO
    with pytest.warns(UserWarning, match="missing.jsonl"):
        assert summary["fingerprint"] == runner.eval_fingerprint(cfg, LOAD_INFO)[0]
    assert (summary["primary_set"], summary["primary_variant"]) == ("mini", "canonical")
    assert summary["headline"] == {"points": 3.0, "max_points": 5.0, "score_pct": 60.0, "parse_fail_rate": 0.25, "points_lenient": 4.0}

    mini = summary["sets"]["mini"]
    assert mini["path"] == str(harness.sets_dir / "mini.jsonl") and mini["n_items"] == 4
    assert list(mini["variants"]) == ["canonical", "payload_only", "reason_then_answer"]
    canonical = mini["variants"]["canonical"]
    assert canonical["per_subject"]["historia"]["points"] == 1.0 and canonical["parse_fail_reasons"] == {"no_answer_line": 1}
    assert mini["variants"]["payload_only"]["points"] == 3.0  # mc + tf; extra words in the short answer and "3.5 m" fail strictly
    assert mini["variants"]["reason_then_answer"]["points"] == canonical["points"]
    assert mini["loglik"] == {"n": 1, "acc": 0.0, "per_subject": {"historia": {"n": 1, "acc": 0.0}}}
    assert summary["sets"]["ext_mc"]["loglik"]["acc"] == 0.5
    assert summary["sets"]["ext_mc"]["variants"]["canonical"]["points"] == 1.0
    assert "missing" not in summary["sets"]
    assert summary["general_heldout"] == {"nll": 1.25, "tokens": 30, "n": 3, "skipped": 0} and harness.nll_pairs == [3]

    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    assert runner.load_summary(out) == summary
    for set_name, variant, n in [("mini", "canonical", 4), ("mini", "payload_only", 4), ("mini", "reason_then_answer", 4), ("ext_mc", "canonical", 2)]:
        rows = read_jsonl(out / f"predictions.{set_name}.{variant}.jsonl")
        assert len(rows) == n and all(r["variant"] == variant for r in rows)
        assert all(r["prompt"].startswith("<s>[system]Jesteś pomocnym asystentem.") and r["prompt"].endswith("[assistant]") for r in rows)
    short = next(r for r in read_jsonl(out / "predictions.mini.canonical.jsonl") if r["id"] == "short-1")
    assert (short["points"], short["points_lenient"], short["parse_ok"]) == (0.0, 1.0, False)
    assert [r["id"] for r in read_jsonl(out / "loglik.mini.jsonl")] == ["mc-1"]
    assert len(read_jsonl(out / "loglik.ext_mc.jsonl")) == 2

    # cache hit: same fingerprint + same model -> no load, identical summary
    with pytest.warns(UserWarning):
        cached = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
        assert cached == summary and len(harness.loads) == 1
        # force re-evaluates
        runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True, force=True)
        assert len(harness.loads) == 2
        # different model under the same run name re-evaluates
        runner.run_eval("org/Other-Model", "base", cfg, is_base=True)
        assert len(harness.loads) == 3
        # changed eval data re-evaluates
        write_jsonl(MINI[:3], harness.sets_dir / "mini.jsonl")
        changed = runner.run_eval("org/Other-Model", "base", cfg, is_base=True)
    assert len(harness.loads) == 4 and changed["fingerprint"] != summary["fingerprint"] and changed["sets"]["mini"]["n_items"] == 3


def test_run_eval_limit_variant_change_and_stale_artifacts(harness):
    cfg = harness.cfg()
    first = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True, limit=2)
    assert first["fingerprint_parts"]["limit"] == 2 and first["sets"]["mini"]["n_items"] == 2
    assert harness.nll_pairs == [2], "general NLL also honours the debug limit"
    out = run_dir(cfg, "base")
    (out / "compare.json").write_text("{}")

    narrow = harness.cfg(**{"eval.sets": [{"path": str(harness.sets_dir / "mini.jsonl"), "variants": ["canonical"]}]})
    second = runner.run_eval("org/Fake-Base-7B", "base", narrow, is_base=True, limit=2)
    assert len(harness.loads) == 2 and second["fingerprint"] != first["fingerprint"]
    assert sorted(p.name for p in out.iterdir()) == [".eval.lock", "predictions.mini.canonical.jsonl", "summary.json"], "stale artifacts removed"
    assert second["sets"]["mini"]["loglik"] is None


def test_run_eval_fingerprint_follows_actual_load_info(harness):
    harness.load_dtype = "bfloat16"
    cfg = harness.cfg()
    summary = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    assert summary["dtype"] == "bfloat16"
    assert summary["fingerprint"] == runner.eval_fingerprint(cfg, {**LOAD_INFO, "dtype": "bfloat16"})[0]


def test_run_eval_with_preloaded_model_does_not_load_or_free(harness):
    cfg = harness.cfg()
    model = SimpleNamespace(_matura_load_info=dict(LOAD_INFO))
    summary = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True, model=model, tokenizer=FakeTokenizer())
    assert harness.loads == [] and harness.frees == 0 and summary["headline"]["points"] == 3.0


def test_run_eval_adapter_only_checkpoint_uses_adapter_subdir(harness, capsys):
    ckpt = harness.tmp / "checkpoints" / "run1"
    adapter = ckpt / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/Fake-Base-7B"}))
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v1")
    cfg = harness.cfg()

    summary = runner.run_eval(str(ckpt), "run1", cfg)
    assert harness.loads[0][0] == str(adapter)
    assert "no merged weights" in capsys.readouterr().out
    assert summary["model"] == str(ckpt.resolve()) and summary["model_source"] == str(adapter.resolve())

    runner.run_eval(str(ckpt), "run1", cfg)
    assert len(harness.loads) == 1, "unchanged adapter -> cached"
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")
    runner.run_eval(str(ckpt), "run1", cfg)
    assert len(harness.loads) == 2, "retrained adapter under the same run name -> re-evaluated"

    (ckpt / "model.safetensors").write_bytes(b"merged")
    assert runner.resolve_model_source(ckpt) == str(ckpt)


def test_format_summary_table(harness):
    cfg = harness.cfg()
    text = runner.format_summary(runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True))
    assert "mini/canonical *" in text and "3/5" in text and "60.00" in text and "25.0" in text
    assert "loglik acc ext_mc: 50.0% (n=2)" in text
    assert "general NLL: 1.2500 (3 pairs, 30 tokens)" in text
    assert "per subject (mini/canonical):" in text and "historia" in text


# ----------------------------------------------------------------------------- scripts/run_eval.py


def _script():
    spec = importlib.util.spec_from_file_location("run_eval_script", ROOT / "scripts" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_help():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_eval.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "--with-base" in proc.stdout and "--base-model" in proc.stdout


def test_script_helpers(tmp_path):
    script = _script()
    ckpt = tmp_path / "run7"
    (ckpt / "adapter").mkdir(parents=True)
    (ckpt / "adapter" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/From-Adapter"}))
    assert script.infer_base_model(str(ckpt)) == "org/From-Adapter"
    assert script.infer_base_model(str(ckpt / "adapter")) == "org/From-Adapter"
    (ckpt / "resolved_config.yaml").write_text("model:\n  base: org/From-Config\n")
    assert script.infer_base_model(str(ckpt)) == "org/From-Config"
    assert script.infer_base_model("org/hub-model") is None

    assert script.default_run_name("org/Model", True) == "base"
    assert script.default_run_name(str(ckpt), False) == "run7"
    assert script.default_run_name(str(ckpt / "adapter"), False) == "run7"
    assert script.resolve_model_arg("org/hub-model") == "org/hub-model"
    with pytest.raises(FileNotFoundError):
        script.resolve_model_arg("checkpoints/definitely-not-a-run")

    cfg = load_config()
    script.apply_set_overrides(cfg, f"data/eval/heldout.jsonl,{ROOT / 'data/eval/dev.jsonl'}", "canonical,bare")
    sets = cfg["eval"]["sets"]
    assert [s["path"] for s in sets] == ["data/eval/heldout.jsonl", "data/eval/dev.jsonl"]
    assert all(s["variants"] == ["canonical", "bare"] and s["loglik"] is True for s in sets)
    assert sets[0]["organizer"] is True, "a configured set keeps its other configs/base.yaml flags"


def test_script_main_with_base_then_compare(tmp_path, monkeypatch, capsys):
    import eval.compare as compare

    script = _script()
    calls, compared = [], []

    def fake_run_eval(model_path, run_name, cfg, *, is_base=False, limit=None, force=False):
        calls.append({"model": model_path, "run_name": run_name, "is_base": is_base, "limit": limit, "force": force,
                      "base": cfg["model"]["base"],  # --variants applies to every configured set, however many there are
                      "variants": sorted({tuple(s["variants"]) for s in cfg["eval"]["sets"]})})
        return {"run_name": run_name, "fingerprint": "fp-1"}

    monkeypatch.setattr(runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(runner, "format_summary", lambda s: f"table:{s['run_name']}")
    monkeypatch.setattr(runner, "load_summary", lambda d: {"fingerprint": "fp-1"} if Path(d).name.startswith("base") else None)
    monkeypatch.setattr(compare, "write_compare", lambda run_path, base_path, **kw: compared.append((run_path, base_path)) or {})
    monkeypatch.setattr(compare, "format_compare_table", lambda cmp: "compare-table")

    ckpt = tmp_path / "run7"
    ckpt.mkdir()
    (ckpt / "resolved_config.yaml").write_text("model:\n  base: org/Some-Base\n")
    results = tmp_path / "results"
    rc = script.main(["--model", str(ckpt), "--with-base", "--limit", "2", "--variants", "canonical", "--force",
                      "--set", f"paths.results_root={results}"])
    assert rc == 0
    suffix = calls[1]["run_name"][len("run7"):]  # --variants narrows the eval: its own -dbg<fp> names, full evals untouched
    assert re.fullmatch(r"-limit2-dbg[0-9a-f]{8}", suffix)
    assert calls == [
        {"model": "org/Some-Base", "run_name": "base" + suffix, "is_base": True, "limit": 2, "force": False, "base": "org/Some-Base",
         "variants": [("canonical",)]},
        {"model": str(ckpt.resolve()), "run_name": "run7" + suffix, "is_base": False, "limit": 2, "force": True, "base": "org/Some-Base",
         "variants": [("canonical",)]},
    ]
    assert compared == [(results / "some-base" / "runs" / f"run7{suffix}", results / "some-base" / "runs" / f"base{suffix}")]
    assert "table:base" in capsys.readouterr().out

    calls.clear()
    assert script.main(["--model", "org/Some-Base", "--base", "--set", f"paths.results_root={results}"]) == 0
    assert calls[0]["run_name"] == "base" and calls[0]["is_base"] and calls[0]["base"] == "org/Some-Base"
    assert len(compared) == 1, "base evals are never compared"


def test_script_main_run_names_settings_and_guards(tmp_path, monkeypatch):
    script = _script()
    calls = []

    def fake_run_eval(model_path, run_name, cfg, *, is_base=False, limit=None, force=False):
        calls.append({"run_name": run_name, "batch_size": cfg["eval"]["batch_size"], "max_new_tokens": cfg["eval"]["max_new_tokens"]})
        return {"run_name": run_name, "fingerprint": "fp-1"}

    monkeypatch.setattr(runner, "run_eval", fake_run_eval)
    monkeypatch.setattr(runner, "format_summary", lambda s: "")
    results = tmp_path / "results"
    ckpt = tmp_path / "run7"
    ckpt.mkdir()
    (ckpt / "resolved_config.yaml").write_text("model:\n  base: org/Some-Base\neval:\n  batch_size: 8\n", encoding="utf-8")
    common = ["--no-compare", "--set", f"paths.results_root={results}"]

    assert script.main(["--model", str(ckpt), *common]) == 0
    assert calls[-1] == {"run_name": "run7", "batch_size": 8, "max_new_tokens": 48}, "the checkpoint's eval settings, plain name"
    assert script.main(["--model", str(ckpt), "--set", "eval.max_new_tokens=8", *common]) == 0
    assert re.fullmatch(r"run7-dbg[0-9a-f]{8}", calls[-1]["run_name"]) and calls[-1]["max_new_tokens"] == 8
    assert script.main(["--model", str(ckpt), "--config", "configs/base.yaml", *common]) == 0
    assert calls[-1]["batch_size"] == 16, "an explicit --config wins over the checkpoint's settings"

    n = len(calls)
    with pytest.raises(ValueError, match="reserved for the base model"):
        script.main(["--model", str(ckpt), "--run-name", "base", *common])
    with pytest.raises(ValueError, match="drop --run-name"):
        script.main(["--model", "org/Some-Base", "--base", "--run-name", "run1", *common])
    slug = results / "some-base"
    slug.mkdir(parents=True)
    (slug / "CANDIDATE.json").write_text(json.dumps({"run_name": "run7", "model_path": str(tmp_path / "elsewhere" / "run7")}))
    with pytest.raises(ValueError, match="belongs to the current candidate"):
        script.main(["--model", str(ckpt), *common])
    assert len(calls) == n, "refused before evaluating (which would overwrite the candidate's eval)"


def test_run_eval_refuses_base_names_for_non_base_models(harness):
    cfg = harness.cfg()
    for name in ("base", "base-limit5", "base-dbg0123abcd", "base-limit5-dbg0123abcd"):
        with pytest.raises(ValueError, match="reserved"):
            runner.run_eval("checkpoints/run1", name, cfg)
    assert harness.loads == []


def test_eval_code_change_invalidates_the_cached_eval(harness, monkeypatch):
    """A parser / partial-credit / prompt fix without a FORMAT_VERSION bump must not compare old base grades with new run grades."""
    cfg = harness.cfg()
    first = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    assert runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True) == first and len(harness.loads) == 1
    monkeypatch.setattr(runner, "eval_code_sha256", lambda: "0" * 16)
    second = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    assert len(harness.loads) == 2 and second["fingerprint"] != first["fingerprint"]
    assert second["fingerprint_parts"]["eval_code_sha256"] == "0" * 16


def test_model_signature_tracks_merged_weights_next_to_the_adapter(tmp_path):
    ckpt = tmp_path / "run1"
    (ckpt / "adapter").mkdir(parents=True)
    (ckpt / "adapter" / "adapter_config.json").write_text("{}")
    (ckpt / "adapter" / "adapter_model.safetensors").write_bytes(b"lora")
    adapter_only = runner.model_signature(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"merged from run1's adapter")
    merged = runner.model_signature(ckpt)
    assert merged != adapter_only
    (ckpt / "model.safetensors").write_bytes(b"merged from ANOTHER adapter onto the same dir")
    assert runner.model_signature(ckpt) != merged, "rebuilt merged weights under the same path are re-evaluated"
    assert runner.model_signature(ckpt / "adapter") == runner.model_signature(ckpt / "adapter")


def test_concurrent_evals_of_the_same_run_wait_and_reuse(harness, monkeypatch):
    """Two pipelines evaluating the base: the second must not clear the artifacts the first just wrote."""
    import threading

    cfg = harness.cfg()
    loading, release = threading.Event(), threading.Event()
    real_load = harness.load

    def slow_load(source, **kwargs):
        loading.set()
        assert release.wait(10)
        return real_load(source, **kwargs)

    monkeypatch.setattr(modeling, "load_model_and_tokenizer", slow_load)
    results: dict[str, dict] = {}
    first = threading.Thread(target=lambda: results.update(a=runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)))
    first.start()
    assert loading.wait(10)
    second = threading.Thread(target=lambda: results.update(b=runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)))
    second.start()
    second.join(0.3)
    assert second.is_alive(), "the second eval waits for the first one's lock"
    release.set()
    first.join(10)
    second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert len(harness.loads) == 1 and results["a"] == results["b"], "the second eval reused the first one's summary"


def test_script_compare_skips_on_fingerprint_mismatch(tmp_path, monkeypatch, capsys):
    script = _script()
    monkeypatch.setattr(runner, "load_summary", lambda d: {"fingerprint": "other"})
    cfg = load_config(None, {"paths.results_root": str(tmp_path)})
    script.compare_with_base(cfg, {"run_name": "run7", "fingerprint": "fp-1"})
    assert "rerun with --with-base" in capsys.readouterr().out


# ----------------------------------------------------------------------------- modeling


def test_resolve_dtype_uses_fp16_on_gpus_that_only_emulate_bf16(monkeypatch):
    """torch.cuda.is_bf16_supported() is True on every CUDA GPU (emulation); T4/V100 must get fp16."""
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "_check_bf16_tensor_supported", lambda device: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=7, minor=5))
    assert modeling.resolve_dtype("auto", "cuda") is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=8, minor=0))
    assert modeling.resolve_dtype("auto", "cuda") is torch.bfloat16
    assert modeling.resolve_dtype("bf16", "cuda") is torch.bfloat16 and modeling.resolve_dtype("fp16", "cuda") is torch.float16


def test_load_model_warns_on_cpu_offload_and_records_attention(monkeypatch):
    import transformers

    class Offloaded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hf_device_map = {"model.layers.0": 0, "model.layers.47": "cpu", "lm_head": "cpu"}
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.generation_config = None

    seen = {}
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda src, **kw: seen.update(kw) or Offloaded())
    with pytest.warns(RuntimeWarning, match="2 module\\(s\\) offloaded to CPU"):
        model = modeling.load_model("org/Big-11B", dtype="fp32", device="cpu")
    assert seen["attn_implementation"] == "sdpa" and model._matura_load_info["attn_implementation"] == "sdpa"
    assert runner.model_load_info(model)["attn_implementation"] == "sdpa"


# ----------------------------------------------------------------------------- slow: real tokenizer / tiny model


@pytest.mark.slow
def test_bielik_tokenizer_single_bos_and_boundaries():
    tok = modeling.load_tokenizer(BIELIK_TOKENIZER)
    item = load_items(ROOT / "data" / "eval" / "heldout.jsonl")[0]
    messages = build_messages(item, "canonical", "Jesteś pomocnym asystentem.")

    rendered = generation.render_prompt(tok, messages)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    assert ids == tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    assert tok(rendered)["input_ids"][:2] == [tok.bos_token_id] * 2, "plain tokenization double-BOSes (why we avoid it)"
    assert tok.eos_token == "<|im_end|>"

    assert loglik.leading_special_ids(tok) == [tok.bos_token_id]
    ctx, cont = loglik.encode_continuation(tok, loglik.render_loglik_prompt(item), " A")
    assert ctx.count(tok.bos_token_id) == 1 and tok.convert_ids_to_tokens(cont) == ["▁A"]

    pair = _pair("Jaka jest stolica Polski?", "Warszawa.")
    full, start = general_loss.encode_pair(tok, pair["messages"], 1024)
    assert tok.decode(full[start:]).startswith("Warszawa.<|im_end|>")


@pytest.mark.slow
def test_smollm2_scoring_matches_reference_and_batching(tmp_path):
    model, tok = modeling.load_model_and_tokenizer(SMOL, dtype="float32", device="cpu")
    try:
        items = load_items(ROOT / "data" / "eval" / "heldout.jsonl")[:6]
        messages = [build_messages(it, "canonical") for it in items[:3]]
        batched = generation.generate_responses(model, tok, messages, max_new_tokens=12, batch_size=3)
        single = generation.generate_responses(model, tok, messages, max_new_tokens=12, batch_size=1)
        assert batched == single and all(isinstance(o, str) and "<|im_end|>" not in o for o in batched)

        mc = [it for it in items if it["type"] == "mc"]
        records = loglik.score_mc_loglik(model, tok, mc, batch_size=4)
        assert len(records) == len(mc) >= 2
        ctx, cont = loglik.encode_continuation(tok, loglik.render_loglik_prompt(mc[0]), " B")
        with torch.inference_mode():
            logprobs = model(input_ids=torch.tensor([ctx + cont])).logits[0].float().log_softmax(-1)
        manual = sum(logprobs[len(ctx) - 1 + j, t].item() for j, t in enumerate(cont))
        assert records[0]["logprobs"]["B"] == pytest.approx(manual, abs=1e-3)
        for a, b in zip(records, loglik.score_mc_loglik(model, tok, mc, batch_size=1)):
            assert all(a["logprobs"][k] == pytest.approx(b["logprobs"][k], abs=1e-3) for k in a["logprobs"])

        pairs = [_pair("Jaka jest stolica Polski?", "Stolicą Polski jest Warszawa."),
                 _pair("Podaj dwa kolory.", "Czerwony i niebieski, czyli kolory z wielu flag.", system="Odpowiadaj krótko.")]
        prompt = tok.apply_chat_template(pairs[0]["messages"][:-1], add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template(pairs[0]["messages"], return_dict=False)
        labels = torch.tensor([full])
        labels[0, : len(prompt)] = -100
        with torch.inference_mode():
            reference = model(input_ids=torch.tensor([full]), labels=labels).loss.item()
        one = general_loss.general_nll(model, tok, pairs[:1], batch_size=1, max_length=1024)
        assert one["nll"] == pytest.approx(reference, rel=1e-4) and one["tokens"] == len(full) - len(prompt)
        both = general_loss.general_nll(model, tok, pairs, batch_size=2, max_length=1024)
        both_single = general_loss.general_nll(model, tok, pairs, batch_size=1, max_length=1024)
        assert both["n"] == 2 and both["nll"] == pytest.approx(both_single["nll"], rel=1e-4)
    finally:
        del model
        modeling.free_model(None)


@pytest.mark.slow
def test_smollm2_run_eval_end_to_end(tmp_path):
    general = tmp_path / "general_heldout.jsonl"
    write_jsonl([_pair("Jaka jest stolica Polski?", "Warszawa."), _pair("Ile to dwa plus dwa?", "Cztery."),
                 _pair("Podaj kolor nieba.", "Niebieski.")], general)
    cfg = load_config(None, {
        "model.base": SMOL,
        "paths.results_root": str(tmp_path / "results"),
        "data.general_heldout": str(general),
        "eval.sets": [{"path": "data/eval/heldout.jsonl", "variants": ["canonical", "payload_only"], "loglik": True}],
        "eval.max_new_tokens": 16,
        "eval.batch_size": 3,
    })
    summary = runner.run_eval(SMOL, "base", cfg, is_base=True, limit=3)
    out = run_dir(cfg, "base")
    assert out.is_relative_to(tmp_path)
    assert summary["headline"].keys() == set(runner.HEADLINE_KEYS)
    assert summary["fingerprint_parts"]["limit"] == 3 and summary["device"] == modeling.detect_device()
    heldout = summary["sets"]["heldout"]
    assert heldout["n_items"] == 3 and set(heldout["variants"]) == {"canonical", "payload_only"}
    for variant in heldout["variants"]:
        rows = read_jsonl(out / f"predictions.heldout.{variant}.jsonl")
        assert len(rows) == 3 and all(r["prompt"].rstrip().endswith("assistant") for r in rows)
    assert heldout["loglik"]["n"] == 1
    (record,) = read_jsonl(out / "loglik.heldout.jsonl")
    assert all(math.isfinite(v) and v < 0 for v in record["logprobs"].values())
    gen = summary["general_heldout"]
    assert gen["n"] == 3 and gen["tokens"] > 0 and math.isfinite(gen["nll"]) and gen["nll"] > 0
    assert runner.load_summary(out) == summary
    assert runner.run_eval(SMOL, "base", cfg, is_base=True, limit=3)["timestamp"] == summary["timestamp"], "cache hit"
