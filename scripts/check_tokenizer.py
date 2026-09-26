#!/usr/bin/env python
"""Check that the GGUF we serve tokenizes chat prompts exactly like the HF tokenizer we train with.

Training (train/sft.py) renders prompts with the HF chat template and HF tokenizer. Serving runs llama.cpp or QVAC on
the official GGUF, which has its own chat template copy and its own tokenizer settings. The known risk for
Bielik-4.5B-v3.0: the GGUF has add_space_prefix=False, while the HF normalizer prepends "▁" to every text segment after
a special token ("<|im_start|>" + "user" -> "▁user"). If the ids differ, the LoRA sees slightly different prompt tokens
at serve time than in training.

  python scripts/check_tokenizer.py --gguf /workspace/models/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf
  python scripts/check_tokenizer.py --gguf M.gguf --llama-bin /workspace/llama.cpp/build/bin   # llama-tokenize binary
  python scripts/check_tokenizer.py --gguf M.gguf --exam exams/mock/exam.json --n 6              # real items via harness.prompts
  python scripts/check_tokenizer.py --gguf M.gguf --server http://127.0.0.1:8080                  # + running llama-server
  python scripts/check_tokenizer.py --gguf-meta M.gguf [--kv-ctx 32768 --kv-types q8_0,q8_0]      # header only, no tokenizer

GGUF tokenization uses llama-cpp-python (vocab_only=True) when importable, else the llama-tokenize binary from
--llama-bin / PATH. Both load only the vocabulary, not the weights. --server runs one 1-token chat completion per
conversation to read usage.prompt_tokens (shows a double BOS or a template difference inside the server).

Exit code: 0 = identical everywhere, 1 = mismatches found (read the report), 2 = could not run.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import difflib
import json
import shutil
import struct
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

HF_DEFAULT = "speakleash/Bielik-4.5B-v3.0-Instruct"
ORGANIZER_SYSTEM_PROMPT_FALLBACK = (
    "Rozwiąż zadanie z historii po polsku. Otrzymujesz tekst źródeł, a obrazy zastąpiono opisami. "
    "Wykorzystaj źródła i własną wiedzę zgodnie z poleceniem. Udziel tylko odpowiedzi na podane zadanie. "
    "Nie dopisuj innych zadań. Nie masz dostępu do narzędzi ani internetu."
)

# ----------------------------------------------------------------------------- GGUF header (stdlib only)

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
# bits per element of KV-cache types (ggml block sizes; tbq*/pq* from qvac-fabric ggml-tbq-types.h)
KV_BITS = {"f32": 32.0, "f16": 16.0, "bf16": 16.0, "q8_0": 8.5, "q4_0": 4.5, "q4_1": 5.0, "q5_0": 5.5, "q5_1": 6.0,
           "iq4_nl": 4.5, "tbq4_0": 5.25, "tbq3_0": 4.25, "pq4_0": 4.125, "pq3_0": 3.125}


def read_gguf_metadata(path: str | Path, keep_arrays: tuple[str, ...] = ()) -> dict[str, Any]:
    """Key/value header of a GGUF file, streamed (tensor data is never read). Arrays are summarised as
    {"array_len": n} unless their key is in keep_arrays."""
    meta: dict[str, Any] = {}
    with open(path, "rb") as fh:

        def rd(fmt: str):
            size = struct.calcsize(fmt)
            buf = fh.read(size)
            if len(buf) != size:
                raise ValueError(f"{path}: truncated GGUF header")
            return struct.unpack(fmt, buf)[0]

        def rstr() -> str:
            return fh.read(rd("<Q")).decode("utf-8", "replace")

        def value(vtype: int, keep: bool):
            if vtype == _STRING:
                return rstr()
            if vtype == _ARRAY:
                etype, count = rd("<I"), rd("<Q")
                if keep:
                    return [value(etype, True) for _ in range(count)]
                if etype == _STRING:
                    for _ in range(count):
                        fh.seek(rd("<Q"), 1)
                else:
                    fh.seek(struct.calcsize(_SCALAR[etype]) * count, 1)
                return {"array_len": count}
            return rd(_SCALAR[vtype])

        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        meta["gguf.version"] = rd("<I")
        meta["gguf.tensor_count"] = rd("<Q")
        n_kv = rd("<Q")
        for _ in range(n_kv):
            key = rstr()
            meta[key] = value(rd("<I"), key in keep_arrays)
    return meta


def kv_estimate(meta: dict, ctx: int, ktype: str, vtype: str) -> dict:
    """KV-cache bytes for ctx tokens (all slots together) from the GGUF attention shape."""
    arch = meta.get("general.architecture", "llama")
    layers = int(meta[f"{arch}.block_count"])
    heads = int(meta.get(f"{arch}.attention.head_count") or 1)
    kv_heads = int(meta.get(f"{arch}.attention.head_count_kv") or heads)
    emb = int(meta.get(f"{arch}.embedding_length") or 0)
    head_k = int(meta.get(f"{arch}.attention.key_length") or (emb // heads if heads else 0))
    head_v = int(meta.get(f"{arch}.attention.value_length") or head_k)
    for t in (ktype, vtype):
        if t not in KV_BITS:
            raise ValueError(f"unknown KV type {t!r} (known: {', '.join(KV_BITS)})")
    per_token = layers * kv_heads * (head_k * KV_BITS[ktype] + head_v * KV_BITS[vtype]) / 8
    return {"arch": arch, "layers": layers, "kv_heads": kv_heads, "head_dim_k": head_k, "head_dim_v": head_v,
            "ctx": ctx, "k": ktype, "v": vtype, "bytes_per_token": per_token, "kv_bytes": int(per_token * ctx)}


# ----------------------------------------------------------------------------- conversations


def organizer_system_prompt() -> str:
    try:
        from harness.prompts import ORGANIZER_SYSTEM_PROMPT  # builder A's contract
        return ORGANIZER_SYSTEM_PROMPT
    except Exception:
        return ORGANIZER_SYSTEM_PROMPT_FALLBACK


def sample_conversations() -> list[tuple[str, list[dict], bool]]:
    """(name, messages, add_generation_prompt). Our own short texts (no exam content), covering the prompt shapes the
    harness sends: closed true/false with a format line, an open item with an image description, the essay, and one
    training-style conversation that ends with the assistant answer."""
    sp = organizer_system_prompt()
    closed = ("Oceń prawdziwość informacji dotyczących unii lubelskiej (1569). Wpisz P, jeśli informacja jest "
              "prawdziwa, albo F – jeśli jest fałszywa.\n1. Unię zawarto za panowania Zygmunta II Augusta.\n"
              "2. Na jej mocy utworzono wspólny sejm.\n\nOdpowiedz wyłącznie w formacie:\n1: P\n2: F")
    open_item = ("Źródło 1.\n„Król winien jest słuchać rady senatu” – fragment traktatu z XVI wieku.\n"
                 "[Opis obrazu images/Z01.png: Mapa Rzeczypospolitej około 1600 roku; zaznaczono granice województw "
                 "i Inflanty.]\n\nNa podstawie źródła 1. i mapy wyjaśnij, dlaczego szlachta domagała się "
                 "egzekucji dóbr. W odpowiedzi uwzględnij dwa argumenty.\n\nTekst po polsku. Podaj wszystkie "
                 "wymagane elementy odpowiedzi.")
    essay = ("Wybierz jeden temat i napisz wypracowanie.\nTemat 1. Oceń politykę wewnętrzną Stefana Batorego.\n"
             "Temat 2. Scharakteryzuj przemiany gospodarcze ziem polskich w latach 1815–1830.\n\n"
             "Jeden tekst: numer wybranego tematu i całe wypracowanie. Minimum 300 wyrazów zgodnie z poleceniem.")
    return [
        ("closed_tf (generation prompt)", [{"role": "system", "content": sp}, {"role": "user", "content": closed}], True),
        ("open + image description (generation prompt)", [{"role": "system", "content": sp}, {"role": "user", "content": open_item}], True),
        ("essay (generation prompt)", [{"role": "system", "content": sp}, {"role": "user", "content": essay}], True),
        ("no system prompt (generation prompt)", [{"role": "user", "content": " Kto zwołał sejm w 1505 roku?"}], True),
        ("training row (prompt + answer)", [{"role": "system", "content": sp}, {"role": "user", "content": closed},
                                            {"role": "assistant", "content": "1: P\n2: F"}], False),
    ]


def exam_conversations(exam_path: str, n: int) -> list[tuple[str, list[dict], bool]]:
    """Real items rendered with the harness (train/serve use the same build_messages). Needs harness.prompts."""
    from harness.prompts import build_messages

    path = Path(exam_path)
    if path.is_dir():  # an unzipped pack folder (exams/mock) works as well as exams/mock/exam.json
        path = path / "exam.json"
    exam = json.loads(path.read_text(encoding="utf-8"))
    items = exam["items"][:n]
    essay = [it for it in exam["items"] if str(it.get("id")) == "26" and it not in items]
    return [(f"exam item {it['id']}", build_messages(it), True) for it in items + essay]


# ----------------------------------------------------------------------------- tokenizers


class GGUFTokenizer:
    def __init__(self, gguf: str, llama_bin: str | None):
        self.gguf = gguf
        self.llm = None
        self.bin = None
        if llama_bin is None:
            try:
                from llama_cpp import Llama  # optional dependency

                self.llm = Llama(model_path=gguf, vocab_only=True, verbose=False)
                self.backend = "llama-cpp-python (vocab_only)"
                return
            except ImportError:
                pass
        cand = Path(llama_bin) / "llama-tokenize" if llama_bin else None
        found = str(cand) if cand and cand.exists() else shutil.which("llama-tokenize")
        if not found:
            raise RuntimeError("no GGUF tokenizer: pip install llama-cpp-python, or pass --llama-bin <llama.cpp/build/bin>")
        self.bin = found
        self.backend = f"{found} (vocab only)"

    def ids(self, text: str) -> list[int]:
        """Token ids of text: special tokens parsed, no BOS added (the template text carries its own <s>)."""
        if self.llm is not None:
            return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        proc = subprocess.run([self.bin, "-m", self.gguf, "--stdin", "--ids", "--no-bos", "--no-escape", "--log-disable"],
                              input=text.encode("utf-8"), capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-tokenize failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-500:]}")
        lines = [ln for ln in proc.stdout.decode().splitlines() if ln.strip().startswith("[")]
        if not lines:
            raise RuntimeError(f"llama-tokenize printed no id list: {proc.stdout.decode(errors='replace')[-300:]}")
        return json.loads(lines[-1])


def load_hf_tokenizer(hf: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf)


def render_gguf_template(template: str, messages: list[dict], add_generation_prompt: bool, bos: str, eos: str) -> str:
    """Render the GGUF's own chat template the way HF does (sandboxed jinja2, trim_blocks, lstrip_blocks)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                            bos_token=bos, eos_token=eos)


# ----------------------------------------------------------------------------- comparison


def diff_ids(a: list[int], b: list[int], pieces) -> list[str]:
    """Up to 3 differing blocks, each shown with token pieces (a = HF, b = GGUF)."""
    out = []
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    for tag, i1, i2, j1, j2 in ops[:3]:
        out.append(f"    {tag} at HF[{i1}:{i2}] / GGUF[{j1}:{j2}]: HF {pieces(a[max(0, i1 - 2):i2 + 2])} "
                   f"vs GGUF {pieces(b[max(0, j1 - 2):j2 + 2])}")
    if len(ops) > 3:
        out.append(f"    ... {len(ops) - 3} more differing blocks")
    return out


def first_text_diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"first difference at char {i}: HF {a[max(0, i - 30):i + 30]!r} vs GGUF {b[max(0, i - 30):i + 30]!r}"


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_check(args) -> int:
    report: dict[str, Any] = {"gguf": args.gguf, "hf": args.hf, "conversations": []}
    problems: list[str] = []

    meta = read_gguf_metadata(args.gguf)
    gg = {k: meta.get(k) for k in ("general.architecture", "tokenizer.ggml.model", "tokenizer.ggml.pre",
                                   "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                                   "tokenizer.ggml.add_space_prefix", "tokenizer.ggml.bos_token_id",
                                   "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id")}
    gg["vocab_size"] = (meta.get("tokenizer.ggml.tokens") or {}).get("array_len")
    template = meta.get("tokenizer.chat_template")
    gg["chat_template_chars"] = len(template) if isinstance(template, str) else None
    report["gguf_meta"] = gg
    print("GGUF:", json.dumps(gg, ensure_ascii=False))

    tok = load_hf_tokenizer(args.hf)
    hf_info = {"class": type(tok).__name__, "bos": tok.bos_token, "eos": tok.eos_token, "vocab_size": len(tok),
               "add_bos_token": getattr(tok, "add_bos_token", None)}
    report["hf"] = hf_info
    print("HF:  ", json.dumps(hf_info, ensure_ascii=False))
    if gg["vocab_size"] and gg["vocab_size"] != len(tok):
        problems.append(f"vocab size differs: GGUF {gg['vocab_size']} vs HF {len(tok)}")

    gtok = GGUFTokenizer(args.gguf, args.llama_bin)
    print(f"GGUF tokenizer backend: {gtok.backend}")

    def pieces(ids: list[int]) -> list[str]:
        return tok.convert_ids_to_tokens(ids)

    convs = exam_conversations(args.exam, args.n) if args.exam else sample_conversations()
    template_mismatch = 0
    for name, messages, gen in convs:
        hf_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)
        hf_ids = list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen, return_dict=False))
        hf_ids_of_text = tok(hf_text, add_special_tokens=False)["input_ids"]
        gguf_ids = gtok.ids(hf_text)
        rec: dict[str, Any] = {"name": name, "hf_len": len(hf_ids), "gguf_len": len(gguf_ids),
                               "tokens_identical": gguf_ids == hf_ids}
        lines = []
        if hf_ids_of_text != hf_ids:
            lines.append("    note: HF tokenizer(text) != HF apply_chat_template(tokenize=True) (BOS handling inside HF)")
        if gguf_ids != hf_ids:
            lines += diff_ids(hf_ids, gguf_ids, pieces)
            problems.append(f"{name}: token ids differ")
        if isinstance(template, str):
            try:
                g_text = render_gguf_template(template, messages, gen, tok.bos_token or "<s>", tok.eos_token or "</s>")
                rec["template_identical"] = g_text == hf_text
                if g_text != hf_text:
                    template_mismatch += 1
                    lines.append("    GGUF chat_template renders differently: " + first_text_diff(hf_text, g_text))
            except Exception as exc:  # llama.cpp uses its own jinja engine (minja); jinja2 may reject some templates
                rec["template_identical"] = None
                lines.append(f"    could not render the GGUF chat_template with jinja2: {exc}")
        if args.server and gen:
            try:
                srv_text = post_json(args.server.rstrip("/") + "/apply-template", {"messages": messages})["prompt"]
                rec["server_template_identical"] = srv_text == hf_text
                if srv_text != hf_text:
                    lines.append("    server /apply-template text differs: " + first_text_diff(hf_text, srv_text))
                    problems.append(f"{name}: server renders a different prompt")
                usage = post_json(args.server.rstrip("/") + "/v1/chat/completions",
                                  {"messages": messages, "max_tokens": 1, "temperature": 0, "seed": 42,
                                   "cache_prompt": False}).get("usage", {})
                rec["server_prompt_tokens"] = usage.get("prompt_tokens")
                if usage.get("prompt_tokens") != len(hf_ids):
                    lines.append(f"    server prompt_tokens={usage.get('prompt_tokens')} vs HF {len(hf_ids)} "
                                 "(+1 usually = double BOS: template <s> plus add_bos_token)")
                    problems.append(f"{name}: server prompt length {usage.get('prompt_tokens')} != HF {len(hf_ids)}")
            except Exception as exc:
                lines.append(f"    server check failed: {exc}")
                problems.append(f"{name}: server check failed")
        status = "OK " if rec["tokens_identical"] else "DIFF"
        print(f"[{status}] {name}: HF {len(hf_ids)} ids, GGUF {len(gguf_ids)} ids")
        for ln in lines:
            print(ln)
        report["conversations"].append(rec)

    if template_mismatch:
        problems.append(f"GGUF chat_template differs from the HF template in {template_mismatch} conversations")
    if not isinstance(template, str):
        problems.append("GGUF has no tokenizer.chat_template: pass --chat-template-file to llama-server or use the HF template")
    starts_with_bos = isinstance(template, str) and "bos_token" in template
    if starts_with_bos and gg.get("tokenizer.ggml.add_bos_token"):
        print("NOTE: the template renders <s> itself and add_bos_token=True. llama-server must not add a second BOS; "
              "check it with --server (prompt_tokens must equal the HF length).")
    if gg.get("tokenizer.ggml.add_space_prefix") is False:
        print("NOTE: GGUF add_space_prefix=False. HF prepends '▁' after special tokens; any DIFF blocks above at "
              "'▁user' / '▁assistant' / after '<|im_end|>' come from this.")

    report["problems"] = problems
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)  # e.g. runs/ on a fresh checkout
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if problems:
        print(f"MISMATCHES ({len(problems)}):")
        for p in problems:
            print("  - " + p)
        print("Options: (1) send HF-tokenized prompts as token ids to llama-server /completion (exact train/serve parity; "
              "QVAC does not accept token-id prompts); (2) keep chat requests: base and tuned see the same GGUF "
              "tokenization, so the comparison stays fair, but the LoRA saw HF ids in training. Decide on the mock.")
        return 1
    print("All checked conversations tokenize identically (HF vs GGUF).")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="GGUF model file (only its vocabulary is loaded)")
    ap.add_argument("--hf", default=HF_DEFAULT, help=f"HF tokenizer id or local dir (default {HF_DEFAULT})")
    ap.add_argument("--llama-bin", default=None, help="dir with llama-tokenize (skips llama-cpp-python)")
    ap.add_argument("--exam", default=None,
                    help="exam.json or its pack folder: render its first --n items (+ essay 26) with harness.prompts")
    ap.add_argument("--n", type=int, default=5, help="items to check with --exam (default 5)")
    ap.add_argument("--server", default=None, help="running llama-server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--json-out", default=None, help="write the full report as JSON")
    ap.add_argument("--gguf-meta", default=None, metavar="GGUF", help="print the GGUF header metadata as JSON and exit")
    ap.add_argument("--kv-ctx", type=int, default=None, help="with --gguf-meta: print a KV-cache estimate for this many tokens")
    ap.add_argument("--kv-types", default="q8_0,q8_0", help="with --kv-ctx: K,V cache types (default q8_0,q8_0)")
    args = ap.parse_args(argv)
    if not args.gguf and not args.gguf_meta:
        ap.error("pass --gguf (tokenizer check) or --gguf-meta (header dump)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gguf_meta:
        meta = read_gguf_metadata(args.gguf_meta)
        if args.kv_ctx:
            k, _, v = args.kv_types.partition(",")
            print(json.dumps(kv_estimate(meta, args.kv_ctx, k, v or k)))
        else:
            print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        return 0
    return run_check(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(2)
    except Exception as exc:
        print(f"check_tokenizer.py: FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
