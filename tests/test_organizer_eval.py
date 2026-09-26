"""Organizer-protocol eval (eval/organizers.py) and its runner wiring.

Fast, no model: the char-level fake tokenizer and deterministic fake LM from test_eval_runtime, extended with the
`logits_to_keep` argument the protocol uses. The prompt is pinned byte-for-byte against the string the organizers'
own `question()` renders for prawko-v2 row 10840 (two of its options carry a leading space, so the rendered lines
have a double space after "A." and "B." -- that is their output and ours must match it exactly).
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from eval import organizers, runner
from eval.schema import read_jsonl, validate_item
from tests.test_eval_runtime import EvalHarness, FakeLM, FakeTokenizer

# prawko-v2 row 10840 as a repo item (subject/points are ours; question and option texts are verbatim, leading
# spaces included). Their row has options as a list with answer index 1 -> letter "B".
PRAWKO_ITEM = {
    "id": "10840",
    "subject": "wos",
    "type": "mc",
    "question": "W jaki sposób przewozisz dziecko o wzroście mniejszym niż 150 cm na przednim siedzeniu samochodu "
                "osobowego, który ma pięć miejsc siedzących?",
    "options": {
        "A": " Na kolanach pasażera.",
        "B": " W foteliku bezpieczeństwa lub innym urządzeniu przytrzymującym dziecko.",
        "C": "W foteliku bezpieczeństwa tyłem do kierunku jazdy, jeżeli pojazd ma aktywną poduszkę powietrzną dla pasażera.",
    },
    "answer": "B",
    "points": 1,
    "source": "prawko-v2",
}

# printed by scripts/prawko.py question(row) / question(row, (1, 2, 0)) on that row
PRAWKO_PROMPT = (
    "Rozwiąż pytanie egzaminacyjne na prawo jazdy w Polsce. Wybierz jedną poprawną odpowiedź. "
    "Odpowiedz wyłącznie literą A, B albo C.\n"
    "\n"
    "W jaki sposób przewozisz dziecko o wzroście mniejszym niż 150 cm na przednim siedzeniu samochodu osobowego, "
    "który ma pięć miejsc siedzących?\n"
    "A.  Na kolanach pasażera.\n"
    "B.  W foteliku bezpieczeństwa lub innym urządzeniu przytrzymującym dziecko.\n"
    "C. W foteliku bezpieczeństwa tyłem do kierunku jazdy, jeżeli pojazd ma aktywną poduszkę powietrzną dla pasażera."
)
PRAWKO_PROMPT_ROTATED = (
    "Rozwiąż pytanie egzaminacyjne na prawo jazdy w Polsce. Wybierz jedną poprawną odpowiedź. "
    "Odpowiedz wyłącznie literą A, B albo C.\n"
    "\n"
    "W jaki sposób przewozisz dziecko o wzroście mniejszym niż 150 cm na przednim siedzeniu samochodu osobowego, "
    "który ma pięć miejsc siedzących?\n"
    "A.  W foteliku bezpieczeństwa lub innym urządzeniu przytrzymującym dziecko.\n"
    "B. W foteliku bezpieczeństwa tyłem do kierunku jazdy, jeżeli pojazd ma aktywną poduszkę powietrzną dla pasażera.\n"
    "C.  Na kolanach pasażera."
)


def _mc(item_id: str, question: str, options: dict, answer: str, subject: str = "historia") -> dict:
    return {"id": item_id, "subject": subject, "type": "mc", "question": question, "options": options,
            "answer": answer, "points": 1, "source": "authored"}


# ----------------------------------------------------------------------------- fakes


class OrganizerLM(FakeLM):
    """FakeLM plus the `logits_to_keep` argument of the organizers' forward pass."""

    def forward(self, input_ids, attention_mask=None, use_cache=None, logits_to_keep=None):
        out = super().forward(input_ids, attention_mask, use_cache)
        return out if not logits_to_keep else SimpleNamespace(logits=out.logits[:, -int(logits_to_keep) :, :])


class ThinkingTokenizer(FakeTokenizer):
    """Chat template that accepts enable_thinking (records what it was called with)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.thinking: list = []

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, return_dict=True,
                            enable_thinking=None):
        self.thinking.append(enable_thinking)
        return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
                                           return_dict=return_dict)


class SplitLetterTokenizer(FakeTokenizer):
    """A tokenizer whose letter labels are NOT single tokens ("C" splits in two)."""

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            ids += super().encode_text(ch) * (2 if ch == "C" else 1)
        return ids


def reference_scores(model, tokenizer, prompt: str, letters: str) -> tuple[list[float], float]:
    """(renormalised letter probabilities, letters' share of the full distribution) for one unpadded prompt."""
    text = organizers.render_organizer_chat(tokenizer, prompt)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    logits = model(torch.tensor([ids])).logits[0, -1].float()
    label_ids = organizers.letter_token_ids(tokenizer, letters)
    return logits[label_ids].softmax(-1).tolist(), logits.softmax(-1)[label_ids].sum().item()


# ----------------------------------------------------------------------------- prompt


def test_organizer_prompt_matches_their_question_byte_for_byte():
    assert organizers.organizer_prompt(PRAWKO_ITEM) == PRAWKO_PROMPT
    assert organizers.organizer_prompt(PRAWKO_ITEM, order=(1, 2, 0)) == PRAWKO_PROMPT_ROTATED
    assert organizers.organizer_prompt(organizers.rotate_item(PRAWKO_ITEM)) == PRAWKO_PROMPT_ROTATED
    assert organizers.ORGANIZER_INSTRUCTION.format(letters="A, B albo C") in PRAWKO_PROMPT


def test_letter_list_and_option_counts():
    assert organizers.letter_list("ABC") == "A, B albo C"
    assert organizers.letter_list("AB") == "A albo B"
    assert organizers.letter_list("ABCD") == "A, B, C albo D"

    four = _mc("m-4", "Ile?", {"A": "1", "B": "2", "C": "3", "D": "4"}, "D")
    prompt = organizers.organizer_prompt(four)
    assert prompt.startswith("Rozwiąż") and "literą A, B, C albo D." in prompt
    assert prompt.endswith("\nA. 1\nB. 2\nC. 3\nD. 4")

    with_context = {**_mc("m-c", "Co wynika z tekstu?", {"A": "x", "B": "y"}, "A"), "context": "Tekst źródłowy."}
    assert organizers.organizer_prompt(with_context).endswith("Tekst źródłowy.\n\nCo wynika z tekstu?\nA. x\nB. y")

    with pytest.raises(ValueError, match="options"):
        organizers.organizer_prompt({"id": "x", "type": "mc", "question": "?", "options": {"A": "one"}, "answer": "A"})
    with pytest.raises(ValueError, match="permutation"):
        organizers.organizer_prompt(PRAWKO_ITEM, order=(0, 1))


def test_organizer_prompt_sha256_is_stable_and_instruction_sensitive():
    sha = organizers.organizer_prompt_sha256()
    assert len(sha) == 16 and sha == organizers.organizer_prompt_sha256(organizers.ORGANIZER_INSTRUCTION)
    expected = hashlib.sha256(
        f"{organizers.PROMPT_BUILDER_VERSION}\0{organizers.ORGANIZER_INSTRUCTION}".encode()
    ).hexdigest()[:16]
    assert sha == expected
    assert organizers.organizer_prompt_sha256("Odpowiedz literą {letters}.") != sha


# ----------------------------------------------------------------------------- letters / rotation


def test_letter_token_ids_requires_single_token_letters():
    tok = FakeTokenizer()
    assert organizers.letter_token_ids(tok, "ABC") == [ord(c) + 3 for c in "ABC"]
    with pytest.raises(ValueError, match=r"single-token letters; \['C'\]"):
        organizers.letter_token_ids(SplitLetterTokenizer(), "ABC")


def test_rotate_item_rotates_options_remaps_the_answer_and_stays_schema_valid():
    rotated = organizers.rotate_item(PRAWKO_ITEM)
    assert rotated["options"] == {"A": PRAWKO_ITEM["options"]["B"], "B": PRAWKO_ITEM["options"]["C"],
                                 "C": PRAWKO_ITEM["options"]["A"]}
    assert rotated["answer"] == "A" and rotated["id"] == PRAWKO_ITEM["id"]
    assert validate_item(copy.deepcopy(rotated)) == rotated
    assert PRAWKO_ITEM["options"]["A"] == " Na kolanach pasażera.", "rotate_item must not mutate its input"

    # their order (1, 2, 0) == shift 1; every shift keeps the gold option text on the gold letter
    for shift in range(1, 5):
        other = organizers.rotate_item(PRAWKO_ITEM, shift)
        assert other["options"][other["answer"]] == PRAWKO_ITEM["options"]["B"]
    assert organizers.rotate_item(PRAWKO_ITEM, 3) == PRAWKO_ITEM
    four = _mc("m-4", "Ile?", {"A": "1", "B": "2", "C": "3", "D": "4"}, "B")
    assert organizers.rotate_item(four, 2)["options"] == {"A": "3", "B": "4", "C": "1", "D": "2"}
    assert organizers.rotate_item(four, 2)["answer"] == "D"
    with pytest.raises(ValueError, match="needs an mc item"):
        organizers.rotate_item({"id": "t-1", "type": "tf", "statements": ["a"], "answer": ["P"]})


# ----------------------------------------------------------------------------- scoring


def test_score_organizer_argmax_probabilities_and_abc_mass():
    tok = FakeTokenizer()
    model = OrganizerLM(tok)
    items = [
        PRAWKO_ITEM,
        {"id": "t-1", "subject": "biologia", "type": "tf", "question": "Oceń.", "statements": ["x"], "answer": ["P"]},
        _mc("m-2", "Ile to dwa plus dwa?", {"A": "trzy", "B": "cztery"}, "B", subject="matematyka"),
        _mc("m-3", "Kto?", {"A": "Krak", "B": "Wanda", "C": "Lech", "D": "Popiel"}, "C"),
    ]
    records = organizers.score_organizer(model, tok, items, batch_size=2)

    assert [r["id"] for r in records] == ["10840", "m-2", "m-3"], "mc items only, input order"
    assert tok.padding_side == "left"
    for record, item in zip(records, [items[0], items[2], items[3]]):
        letters = "".join(sorted(item["options"]))
        probabilities, mass = reference_scores(model, tok, organizers.organizer_prompt(item), letters)
        assert record["letters"] == letters and record["rotate"] == 0 and record["type"] == "mc"
        assert record["subject"] == item["subject"] and record["answer"] == item["answer"]
        assert list(record["probabilities"]) == list(letters)
        for letter, p in zip(letters, probabilities):
            assert record["probabilities"][letter] == pytest.approx(p, abs=1e-6)
        assert sum(record["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
        assert record["abc_mass"] == pytest.approx(mass, abs=1e-6)
        assert 0.0 <= record["abc_mass"] <= 1.0
        best = max(letters, key=lambda letter: record["probabilities"][letter])
        assert record["prediction"] == best and record["correct"] == (best == item["answer"])

    one_by_one = organizers.score_organizer(model, tok, items, batch_size=1)
    assert [r["prediction"] for r in one_by_one] == [r["prediction"] for r in records], "batching must not matter"
    for a, b in zip(records, one_by_one):
        assert a["abc_mass"] == pytest.approx(b["abc_mass"], abs=1e-6)
    assert organizers.score_organizer(model, tok, items[1:2], batch_size=2) == [], "no mc items -> nothing scored"


def test_score_organizer_rotates_and_skips_unusable_letters():
    tok = FakeTokenizer()
    model = OrganizerLM(tok)
    rotated = organizers.score_organizer(model, tok, [PRAWKO_ITEM], batch_size=4, rotate=1)
    assert len(rotated) == 1 and rotated[0]["rotate"] == 1 and rotated[0]["answer"] == "A"
    probabilities, _ = reference_scores(model, tok, PRAWKO_PROMPT_ROTATED, "ABC")
    assert [rotated[0]["probabilities"][letter] for letter in "ABC"] == pytest.approx(probabilities, abs=1e-6)
    assert rotated[0]["id"] == PRAWKO_ITEM["id"], "same id as the unrotated row; distinguished by 'rotate'"

    split = SplitLetterTokenizer()
    with pytest.warns(UserWarning, match="organizer scoring skipped an item"):
        assert organizers.score_organizer(OrganizerLM(split), split, [PRAWKO_ITEM], batch_size=4) == []


def test_score_organizer_uses_the_chat_template_without_generating():
    thinking = ThinkingTokenizer()
    model = OrganizerLM(thinking)
    organizers.score_organizer(model, thinking, [PRAWKO_ITEM], batch_size=4)
    assert thinking.thinking == [False], "enable_thinking=False is passed when the template accepts it"
    assert model.generate_calls == [] and model.forward_batches == [(1, len(thinking.encode_text(
        organizers.render_organizer_chat(thinking, PRAWKO_PROMPT))))]

    plain = FakeTokenizer()  # its template has no enable_thinking parameter: the TypeError fallback renders anyway
    assert organizers.render_organizer_chat(plain, "Q") == plain.render([{"role": "user", "content": "Q"}], True)
    assert len(organizers.score_organizer(OrganizerLM(plain), plain, [PRAWKO_ITEM], batch_size=4)) == 1


def test_summarize_organizer():
    records = [
        {"subject": "historia", "answer": "A", "correct": True, "probabilities": {"A": 0.6, "B": 0.4}, "abc_mass": 0.5},
        {"subject": "historia", "answer": "B", "correct": False, "probabilities": {"A": 0.8, "B": 0.2}, "abc_mass": 0.3},
        {"subject": "fizyka", "answer": "A", "correct": True, "probabilities": {"A": 1.0, "B": 0.0}, "abc_mass": 0.4},
    ]
    assert organizers.summarize_organizer(records) == {
        "n": 3, "correct": 2, "accuracy": 0.6667, "mean_correct_probability": 0.6, "mean_abc_mass": 0.4,
        "per_subject": {"fizyka": {"n": 1, "correct": 1, "accuracy": 1.0},
                        "historia": {"n": 2, "correct": 1, "accuracy": 0.5}},
    }
    assert organizers.summarize_organizer([]) == {
        "n": 0, "correct": 0, "accuracy": 0.0, "mean_correct_probability": 0.0, "mean_abc_mass": 0.0, "per_subject": {},
    }


# ----------------------------------------------------------------------------- fingerprint


LOAD_INFO = {"device": "cpu", "dtype": "float32", "quantization": "none", "attn_implementation": "sdpa",
             **runner.environment_info("cpu")}


def _pre_feature_fingerprint(cfg: dict, load_info: dict) -> tuple[str, dict]:
    """The fingerprint exactly as it was computed before the organizer feature (no organizers.py in the code hash,
    no organizer keys in the per-set parts): what a set without `organizer` must still produce, byte for byte."""
    from eval.answer_format import FORMAT_VERSION
    from eval.loglik import LLMZSZL_TEMPLATE
    from train.config import get_dotted, resolve_path

    ev = cfg.get("eval") or {}
    sets = runner.eval_sets_from_cfg(cfg)
    h = hashlib.sha256()
    here = runner.Path(runner.__file__).resolve().parent
    for name in ("answer_format.py", "grader.py", "prompts.py", "schema.py", "generation.py", "loglik.py", "general_loss.py"):
        h.update(name.encode() + b"\0" + (here / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    parts: dict = {
        "format_version": FORMAT_VERSION,
        "eval_code_sha256": h.hexdigest()[:16],
        "sets": [{"name": s["name"], "sha256": runner.file_sha256(s["path"]), "variants": s["variants"],
                  "loglik": s["loglik"]} for s in sets],
        "max_new_tokens": ev.get("max_new_tokens"),
        "system_prompt": ev.get("system_prompt"),
        "limit": ev.get("limit"),
        "batch_size": int(ev.get("batch_size") or 8),
        "general_loss": None,
        **{k: load_info.get(k) for k in ("dtype", "quantization", "device", "attn_implementation", "torch",
                                         "transformers", "accelerator")},
    }
    if any(runner.REASONING_VARIANT in s["variants"] for s in sets):
        parts["max_new_tokens_reasoning"] = ev.get("max_new_tokens_reasoning")
    if any(s["loglik"] for s in sets):
        parts["loglik_template_sha256"] = hashlib.sha256(LLMZSZL_TEMPLATE.encode()).hexdigest()[:16]
    if ev.get("general_loss"):
        heldout = get_dotted(cfg, "data.general_heldout")
        parts["general_loss"] = {
            "sha256": runner.file_sha256(heldout) if heldout and resolve_path(heldout).exists() else None,
            "max_items": ev.get("general_loss_max_items"),
            "max_length": get_dotted(cfg, "model.max_length"),
        }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16], parts


def test_fingerprint_of_a_set_without_organizer_is_unchanged(tmp_path, monkeypatch):
    harness = EvalHarness(tmp_path, monkeypatch)
    plain = harness.cfg()
    assert runner.eval_fingerprint(plain, LOAD_INFO) == _pre_feature_fingerprint(plain, LOAD_INFO)
    assert all("organizer" not in key for part in runner.eval_fingerprint(plain, LOAD_INFO)[1]["sets"] for key in part)
    assert runner.eval_fingerprint(plain, LOAD_INFO)[1]["eval_code_sha256"] == runner.eval_code_sha256()


def test_fingerprint_tracks_organizer_settings_and_code(tmp_path, monkeypatch):
    harness = EvalHarness(tmp_path, monkeypatch)
    plain = harness.cfg()
    plain_fp = runner.eval_fingerprint(plain, LOAD_INFO)[0]
    sets = copy.deepcopy(plain["eval"]["sets"])

    enabled = harness.cfg(**{"eval.sets": [{**sets[0], "organizer": True}, sets[1]]})
    fp, parts = runner.eval_fingerprint(enabled, LOAD_INFO)
    assert fp != plain_fp, "enabling the organizer protocol invalidates the cached eval"
    assert parts["sets"][0] == {"name": "mini", "sha256": parts["sets"][0]["sha256"], "variants": sets[0]["variants"],
                               "loglik": True, "organizer": True, "organizer_rotate": False,
                               "organizer_prompt_sha256": organizers.organizer_prompt_sha256()}
    assert parts["sets"][1] == _pre_feature_fingerprint(plain, LOAD_INFO)[1]["sets"][1], "the other set is untouched"
    assert parts["eval_code_sha256"] != runner.eval_code_sha256(), "organizers.py is hashed in for organizer evals"
    assert parts["eval_code_sha256"] == runner.eval_code_sha256(runner.ORGANIZER_CODE_FILES)

    rotate = harness.cfg(**{"eval.sets": [{**sets[0], "organizer": True, "organizer_rotate": True}, sets[1]]})
    assert runner.eval_fingerprint(rotate, LOAD_INFO)[0] != fp, "the rotated pass changes what the summary holds"
    ignored = harness.cfg(**{"eval.sets": [{**sets[0], "organizer_rotate": True}, sets[1]]})
    assert runner.eval_fingerprint(ignored, LOAD_INFO)[0] == plain_fp, "organizer_rotate alone does nothing"

    monkeypatch.setattr(organizers, "ORGANIZER_INSTRUCTION", "Odpowiedz literą {letters}.")
    assert runner.eval_fingerprint(enabled, LOAD_INFO)[0] != fp, "a prompt change invalidates the cached eval"


def test_organizer_sets_from_cfg_normalizes(tmp_path, monkeypatch):
    harness = EvalHarness(tmp_path, monkeypatch)
    sets = copy.deepcopy(harness.cfg()["eval"]["sets"])
    cfg = harness.cfg(**{"eval.sets": [{**sets[0], "organizer": True, "organizer_rotate": True}, sets[1]]})
    assert runner.organizer_sets_from_cfg(cfg) == {
        "mini": {"organizer": True, "organizer_rotate": True},
        "ext_mc": {"organizer": False, "organizer_rotate": False},
    }
    assert runner.eval_sets_from_cfg(cfg)[0].keys() == {"name", "path", "variants", "loglik", "optional"}


# ----------------------------------------------------------------------------- runner wiring


def fake_score_organizer(model, tokenizer, items, *, batch_size=8, instruction=None, rotate=0):
    """Deterministic stand-in for score_organizer: predicts "A" for every mc item, skips everything else."""
    records = []
    for item in items:
        if item["type"] != "mc":
            continue
        row = organizers.rotate_item(item, rotate) if rotate else item
        letters = "".join(sorted(row["options"]))
        records.append({
            "id": row["id"], "subject": row["subject"], "type": "mc", "letters": letters, "answer": row["answer"],
            "prediction": "A", "correct": row["answer"] == "A",
            "probabilities": {letter: (0.5 if letter == "A" else 0.5 / (len(letters) - 1)) for letter in letters},
            "abc_mass": 0.25, "rotate": rotate,
        })
    return records


def organizer_cfg(harness: EvalHarness, **overrides) -> dict:
    sets = copy.deepcopy(harness.cfg()["eval"]["sets"])
    return harness.cfg(**{"eval.sets": [
        {**sets[0], "variants": ["canonical"], "loglik": False, "organizer": True, "organizer_rotate": True},
        {**sets[1], "variants": [], "loglik": False, "organizer": True},  # organizer-only: no generation, no grading
    ], **overrides})


def test_run_eval_scores_the_organizer_protocol_and_writes_its_predictions(tmp_path, monkeypatch, capsys):
    harness = EvalHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(organizers, "score_organizer", fake_score_organizer)
    cfg = organizer_cfg(harness)
    summary = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    out = runner.run_dir(cfg, "base")

    mini = summary["sets"]["mini"]["organizer"]
    assert mini == {"n": 1, "correct": 0, "accuracy": 0.0, "mean_correct_probability": 0.25, "mean_abc_mass": 0.25,
                    "per_subject": {"historia": {"n": 1, "correct": 0, "accuracy": 0.0}},
                    "skipped": 3,  # tf / short / numeric items are not scorable under the organizers' protocol
                    "rotated": {"n": 1, "correct": 1, "accuracy": 1.0, "mean_correct_probability": 0.5,
                                "mean_abc_mass": 0.25,
                                "per_subject": {"historia": {"n": 1, "correct": 1, "accuracy": 1.0}}}}
    ext = summary["sets"]["ext_mc"]["organizer"]
    assert (ext["n"], ext["correct"], ext["skipped"], ext["rotated"]) == (2, 1, 0, None)

    rows = read_jsonl(out / "organizer.mini.jsonl")
    assert [(r["id"], r["rotate"], r["answer"]) for r in rows] == [("mc-1", 0, "B"), ("mc-1", 1, "A")]
    assert rows[0].keys() == {"id", "subject", "type", "letters", "answer", "prediction", "correct", "probabilities",
                              "abc_mass", "rotate"}
    assert [r["rotate"] for r in read_jsonl(out / "organizer.ext_mc.jsonl")] == [0, 0], "no rotated pass for ext_mc"

    assert summary["sets"]["ext_mc"]["variants"] == {}, "variants: [] -> the organizer path only"
    assert [c["desc"] for c in harness.generate_calls] == ["mini/canonical"], "nothing is generated for ext_mc"
    assert (summary["primary_set"], summary["primary_variant"]) == ("mini", "canonical")
    assert summary["headline"]["points"] == 3.0
    printed = capsys.readouterr().out
    assert "mini organizer acc 0.0% (n=1, ABC mass 25.0%, rotated acc 100.0%)" in printed
    assert "ext_mc organizer acc 50.0% (n=2, ABC mass 25.0%)" in printed

    text = runner.format_summary(summary)
    assert "organizer acc mini: 0.0% (n=1, skipped 3, ABC mass 25.0%, rotated 100.0%)" in text

    # the organizer block is part of the cached eval; flipping the rotated pass re-evaluates
    assert runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True) == summary and len(harness.loads) == 1
    sets = copy.deepcopy(cfg["eval"]["sets"])
    sets[0]["organizer_rotate"] = False
    again = runner.run_eval("org/Fake-Base-7B", "base", harness.cfg(**{"eval.sets": sets}), is_base=True)
    assert len(harness.loads) == 2 and again["sets"]["mini"]["organizer"]["rotated"] is None


def test_run_eval_without_organizer_writes_no_organizer_block(tmp_path, monkeypatch):
    harness = EvalHarness(tmp_path, monkeypatch)
    summary = runner.run_eval("org/Fake-Base-7B", "base", harness.cfg(), is_base=True)
    assert all("organizer" not in s for s in summary["sets"].values())
    assert not list(runner.run_dir(harness.cfg(), "base").glob("organizer.*.jsonl"))


def test_run_eval_with_only_organizer_sets_has_no_headline(tmp_path, monkeypatch):
    harness = EvalHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(organizers, "score_organizer", fake_score_organizer)
    sets = copy.deepcopy(harness.cfg()["eval"]["sets"])
    cfg = harness.cfg(**{"eval.sets": [{**s, "variants": [], "loglik": False, "organizer": True} for s in sets]})

    with pytest.warns(UserWarning, match="organizer-only eval"):
        summary = runner.run_eval("org/Fake-Base-7B", "base", cfg, is_base=True)
    assert summary["headline"] is None and summary["primary_set"] is None and summary["primary_variant"] is None
    assert harness.generate_calls == [] and summary["sets"]["mini"]["organizer"]["n"] == 1
    assert json.loads((runner.run_dir(cfg, "base") / "summary.json").read_text(encoding="utf-8")) == summary
    text = runner.format_summary(summary)
    assert "no prompt variants were graded" in text and "organizer acc mini:" in text
