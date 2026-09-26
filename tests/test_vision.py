"""harness/vision.py and scripts/describe_images.py without any model: a fake VLM replaces the HTTP call."""
import base64
import importlib.util
import io
import json
import os
import pathlib
import struct
import subprocess
import threading
import zlib
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness import vision

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _png(rgb: tuple[int, int, int], w: int = 2, h: int = 2) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


RED, GREEN, BLUE = _png((255, 0, 0)), _png((0, 255, 0)), _png((0, 0, 255))


def _img(path: str, data: bytes) -> dict:
    import hashlib
    return {"path": path, "source_page": 1, "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def exam(tmp_path):
    """A.png and C.png are byte-identical; B.png is shared by 2.1/2.2; D.png has no marker (like items 7/8/15)."""
    d = tmp_path / "exam"
    (d / "images").mkdir(parents=True)
    for name, data in {"A": RED, "B": GREEN, "C": RED, "D": BLUE}.items():
        (d / "images" / f"{name}.png").write_bytes(data)
    items = [
        {"id": "1", "question": "Rozstrzygnij, czy mapa dotyczy XVI wieku.",
         "source_text": "Źródło 1. Mapa Rzeczypospolitej\n[Obraz: images/A.png]\n\nNa podstawie: atlas.",
         "images": [_img("images/A.png", RED)], "answer_format": "Tekst po polsku."},
        {"id": "2.1", "question": "Podaj nazwę budowli.", "source_text": "Fotografia\n[Obraz: images/B.png]",
         "images": [_img("images/B.png", GREEN)], "answer_format": "Tekst po polsku."},
        {"id": "2.2", "question": "Podaj styl budowli.", "source_text": "Fotografia\n[Obraz: images/B.png]",
         "images": [_img("images/B.png", GREEN)], "answer_format": "A"},
        {"id": "3", "question": "Kto jest autorem rysunku?", "source_text": "Rysunek\n[Obraz: images/C.png]",
         "images": [_img("images/C.png", RED)], "answer_format": "Tekst po polsku."},
        {"id": "4", "question": "Podaj nazwę stylu obrazu.", "source_text": "Józef Chełmoński, Bociany.",
         "images": [_img("images/D.png", BLUE)], "answer_format": "Tekst po polsku."},
        {"id": "5", "question": "Pytanie bez obrazu.", "source_text": "Sam tekst.", "images": [],
         "answer_format": "A"},
    ]
    (d / "exam.json").write_text(json.dumps({"exam_id": "test-exam", "items": items}, ensure_ascii=False),
                                 encoding="utf-8")
    return d, items


class FakeVLM:
    """Stands in for vision._http_json. Answers by image colour so each image gets a recognisable text."""

    def __init__(self, replies=None, fail_on=None, model="fake-vlm.gguf"):
        self.calls = []
        self.replies = replies or {}
        self.fail_on = fail_on
        self.model = model
        self.lock = threading.Lock()

    def colour(self, payload):
        uri = payload["messages"][0]["content"][0]["image_url"]["url"]
        data = base64.b64decode(uri.split(",", 1)[1])
        return {RED: "red", GREEN: "green", BLUE: "blue"}[data]

    def __call__(self, url, payload=None, timeout=None, api_key=None):
        if payload is None:
            return {"object": "list", "data": []}
        with self.lock:
            self.calls.append((url, payload))
        colour = self.colour(payload)
        if colour == self.fail_on:
            raise vision.VisionError("HTTP 500 from fake")
        reply = self.replies.get(colour, f"Ilustracja w kolorze {colour}. Napis „{colour.upper()}”.")
        if callable(reply):
            reply = reply(payload)
        return {"model": self.model, "choices": [{"message": {"role": "assistant", "content": reply},
                                                  "finish_reason": "stop"}]}


@pytest.fixture
def no_tesseract(monkeypatch):
    monkeypatch.setattr(vision.shutil, "which", lambda name: None)


# ------------------------------------------------------------------------------------------ parsing


def test_markers_and_item_paths(exam):
    _, items = exam
    assert vision.find_markers("a [Obraz: images/X.png] b [Obraz:  ./images/Y.png ]") == ["images/X.png", "images/Y.png"]
    assert vision.item_image_paths(items[0]) == ["images/A.png"]
    assert vision.item_image_paths(items[4]) == ["images/D.png"]  # listed in images, no marker
    marker_only = {"id": "9", "source_text": "[Obraz: images/B.png]", "images": []}
    assert vision.item_image_paths(marker_only) == ["images/B.png"]


def test_collect_refs_dedupes_by_sha_and_reports_missing(exam):
    d, items = exam
    missing = []
    refs = vision.collect_image_refs(d, items + [{"id": "8", "source_text": "[Obraz: images/NOPE.png]"}],
                                     missing=missing)
    assert missing == ["images/NOPE.png"]
    by_path = {r.path: r for r in refs}
    assert set(by_path) == {"images/A.png", "images/B.png", "images/D.png"}
    a = by_path["images/A.png"]
    assert a.paths == ["images/A.png", "images/C.png"] and a.item_ids == ["1", "3"]
    assert a.sha256 == vision.sha256_file(d / "images" / "C.png") == a.declared_sha256
    b = by_path["images/B.png"]
    assert b.item_ids == ["2.1", "2.2"] and len(b.questions) == 2 and len(b.source_contexts) == 1


def test_source_context_marks_this_image():
    text = "Źródło 1. Plany\n[Obraz: images/A.png]\n\n[Obraz: images/B.png]\n\nŹródło 2. Tekst"
    ctx = vision.source_context(text, "images/B.png")
    assert vision.THIS_IMAGE in ctx and vision.OTHER_IMAGE in ctx and "[Obraz:" not in ctx
    assert ctx.index(vision.OTHER_IMAGE) < ctx.index(vision.THIS_IMAGE)
    assert vision.source_context("Józef Chełmoński, Bociany.", "images/D.png") == "Józef Chełmoński, Bociany."


def test_data_uri_encoding(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(RED)
    uri = vision.image_data_uri(p)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == RED
    j = tmp_path / "y.bin"
    j.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)
    assert vision.image_data_uri(j).startswith("data:image/jpeg;base64,")


def test_prompt_and_payload_contract(exam):
    d, items = exam
    ref = next(r for r in vision.collect_image_refs(d, items) if r.path == "images/B.png")
    msgs = vision.build_vlm_messages(ref, "data:image/png;base64,AAAA")
    parts = msgs[0]["content"]
    assert msgs[0]["role"] == "user" and [p["type"] for p in parts] == ["image_url", "text"]  # image first
    text = parts[1]["text"]
    assert "Podaj nazwę budowli." in text and "Podaj styl budowli." in text and vision.THIS_IMAGE in text
    assert "Nie rozwiązuj zadania" in text and "Skan tekstu" in text
    for kind in vision.IMAGE_KINDS:
        assert kind in text.lower()
    body = vision.request_payload("vlm", msgs)
    assert body["temperature"] == 0 and body["seed"] == 42 and body["max_tokens"] == 700 and body["top_k"] == 1
    assert body["repeat_penalty"] == 1.0 and body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["reasoning_budget"] == 0
    retry = vision.request_payload("vlm", msgs, retry=True)
    assert retry["temperature"] == 0 and retry["presence_penalty"] > 0 and retry["repeat_penalty"] > 1


def test_api_url():
    assert vision.api_url("http://127.0.0.1:8081", "chat/completions") == "http://127.0.0.1:8081/v1/chat/completions"
    assert vision.api_url("http://h:11434/v1/", "/models") == "http://h:11434/v1/models"


# ------------------------------------------------------------------------------------------ describe + cache


def test_describe_images_calls_once_per_unique_image(exam, monkeypatch, no_tesseract):
    d, items = exam
    fake = FakeVLM()
    monkeypatch.setattr(vision, "_http_json", fake)
    out = vision.describe_images(d, items, base_url="http://vlm:1", model="vlm")
    assert len(fake.calls) == 3
    assert all(url == "http://vlm:1/v1/chat/completions" for url, _ in fake.calls)
    assert set(out) == {"images/A.png", "images/B.png", "images/C.png", "images/D.png"}
    assert out["images/A.png"] == out["images/C.png"] == "Ilustracja w kolorze red. Napis „RED”."
    cache = json.loads((d / "descriptions.json").read_text(encoding="utf-8"))
    assert set(cache) == set(out)
    for path, entry in cache.items():
        assert set(entry) == {"sha256", "description", "model", "ocr"}
        assert entry["sha256"] == vision.sha256_file(d / path)
        assert entry["model"] == "fake-vlm.gguf" and entry["ocr"] is None
    assert vision.load_descriptions(d / "descriptions.json") == out


def test_cache_reused_by_sha_and_force(exam, monkeypatch, no_tesseract):
    d, items = exam
    cache_path = d / "cache" / "desc.json"
    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    first = vision.describe_images(d, items, cache_path=cache_path)

    def boom(*a, **k):
        raise AssertionError("no request expected")

    monkeypatch.setattr(vision, "_http_json", boom)
    again, results = vision.describe_images_detailed(d, items, cache_path=cache_path)
    assert again == first and {r.status for r in results} == {"cached"}

    (d / "images" / "D.png").write_bytes(_png((1, 2, 3)))  # new bytes -> only D is described again
    fake = FakeVLM(replies={})
    fake.colour = lambda payload: "blue" if base64.b64decode(
        payload["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1]) not in (RED, GREEN) else "x"
    monkeypatch.setattr(vision, "_http_json", fake)
    vision.describe_images(d, items, cache_path=cache_path)
    assert len(fake.calls) == 1

    fake2 = FakeVLM()
    fake2.colour = lambda payload: "red"
    monkeypatch.setattr(vision, "_http_json", fake2)
    vision.describe_images(d, items, cache_path=cache_path, force=True, parallel=1)
    assert len(fake2.calls) == 3


def test_cache_entry_copied_to_new_path_with_same_bytes(exam, monkeypatch, no_tesseract):
    d, items = exam
    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    vision.describe_images(d, items)
    (d / "images" / "E.png").write_bytes(GREEN)
    extra = {"id": "6", "question": "Q", "source_text": "[Obraz: images/E.png]", "images": []}
    fake = FakeVLM()
    monkeypatch.setattr(vision, "_http_json", fake)
    out = vision.describe_images(d, items + [extra])
    assert fake.calls == [] and out["images/E.png"] == out["images/B.png"]
    assert "images/E.png" in json.loads((d / "descriptions.json").read_text(encoding="utf-8"))


def test_stale_cache_from_another_exam_is_not_used(exam, monkeypatch, no_tesseract):
    d, items = exam
    stale = {"images/A.png": {"sha256": "0" * 64, "description": "opis innego obrazu", "model": "m", "ocr": None}}
    vision.save_cache(d / "descriptions.json", stale)
    assert vision.load_descriptions(d / "descriptions.json", d, items) == {}
    fake = FakeVLM()
    monkeypatch.setattr(vision, "_http_json", fake)
    out = vision.describe_images(d, items)
    assert len(fake.calls) == 3 and out["images/A.png"].startswith("Ilustracja w kolorze red")
    assert vision.load_descriptions(d / "descriptions.json", d, items)["images/A.png"] == out["images/A.png"]


def test_failed_image_is_not_cached_and_rerun_retries_it(exam, monkeypatch, no_tesseract):
    d, items = exam
    monkeypatch.setattr(vision, "_http_json", FakeVLM(fail_on="green"))
    out, results = vision.describe_images_detailed(d, items)
    failed = [r for r in results if r.status == "failed"]
    assert [r.path for r in failed] == ["images/B.png"] and "HTTP 500" in failed[0].error
    assert "images/B.png" not in out
    assert "images/B.png" not in json.loads((d / "descriptions.json").read_text(encoding="utf-8"))
    fake = FakeVLM()
    monkeypatch.setattr(vision, "_http_json", fake)
    out2 = vision.describe_images(d, items)
    assert len(fake.calls) == 1 and "images/B.png" in out2


def test_think_block_stripped_and_loop_retried(exam, monkeypatch, no_tesseract):
    d, items = exam
    looping = "Mapa. " + "Napis „X”. " * 30
    fake = FakeVLM(replies={
        "red": "<think>rozważam</think>\n**Mapa** Polski.",
        "green": lambda p: "Fotografia kościoła." if p["presence_penalty"] > 0 else looping,
        "blue": "<think>bez końca",
    })
    monkeypatch.setattr(vision, "_http_json", fake)
    out, results = vision.describe_images_detailed(d, items)
    by = {r.path: r for r in results}
    assert out["images/A.png"] == "Mapa Polski."
    assert by["images/B.png"].retried and out["images/B.png"] == "Fotografia kościoła."
    assert by["images/D.png"].status == "failed" and "empty" in by["images/D.png"].error
    assert len(fake.calls) == 4  # red, green x2, blue


def test_degenerate_detection():
    assert vision.is_degenerate("Tekst. " + "abc def " * 10)
    assert vision.is_degenerate("\n".join(["- Kraków"] * 4))
    assert not vision.is_degenerate("Tablica genealogiczna.\n- Zygmunt I Stary, 1506–1548.\n- Zygmunt II August.")
    assert vision.collapse_repetition("A\nA\nB") == "A\nB"


# ------------------------------------------------------------------------------------------ OCR (optional)


def test_tesseract_missing_means_no_ocr(exam, monkeypatch, no_tesseract):
    d, items = exam

    def no_run(*a, **k):
        raise AssertionError("tesseract must not run")

    monkeypatch.setattr(vision.subprocess, "run", no_run)
    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    _, results = vision.describe_images_detailed(d, items, ocr="auto")
    assert all(r.ocr is None for r in results)
    _, results = vision.describe_images_detailed(d, items, ocr="on", force=True)
    assert all("tesseract" in (r.error or "") for r in results if r.status == "described")


def test_tesseract_present_fills_ocr(exam, monkeypatch):
    d, items = exam
    seen = []

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        seen.append(cmd)
        if "--list-langs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="List of available languages (2):\neng\npol\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="NAPIS NA MAPIE\n\f", stderr="")

    monkeypatch.setattr(vision.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    vision.describe_images(d, items)
    cache = json.loads((d / "descriptions.json").read_text(encoding="utf-8"))
    assert {e["ocr"] for e in cache.values()} == {"NAPIS NA MAPIE"}
    ocr_cmds = [c for c in seen if "--list-langs" not in c]
    assert len(ocr_cmds) == 3 and all(c[-2:] == ["-l", "pol"] and c[2] == "stdout" for c in ocr_cmds)


def test_tesseract_without_polish_is_skipped(exam, monkeypatch):
    d, items = exam
    monkeypatch.setattr(vision.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(vision.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(
        cmd, 0, stdout="List of available languages (1):\neng\n", stderr=""))
    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    _, results = vision.describe_images_detailed(d, items)
    assert all(r.ocr is None for r in results)


def test_ocr_coverage_and_reference_scores():
    assert vision.ocr_coverage("Mapa. Napis „Warszawa” i „Kraków”.", "Warszawa Kraków Gdańsk Poznań") == 0.5
    assert vision.ocr_coverage("x", "ab") is None and vision.ocr_coverage("x", None) is None
    ref = "Plan. Podpis „Łomża” oraz nazwiska „BEM” i „DYBICZ”. Rzeka z przeprawami."
    sc = vision.reference_scores("Plan z napisami „ŁOMŻA”, „Bem”. Widać rzekę i przeprawy.", ref)
    assert sc["label_recall"] == pytest.approx(2 / 3, abs=1e-3) and 0 < sc["word_recall"] < 1


# ------------------------------------------------------------------------------------------ real HTTP (local fake)


def test_http_roundtrip_against_local_fake_server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen["get"] = self.path
            self._send({"object": "list", "data": [{"id": "vlm"}]})

        def do_POST(self):
            seen["post"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send({"choices": [{"message": {"content": "Mapa."}, "finish_reason": "stop"}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert vision.check_server(base) is None
        resp = vision._http_json(vision.api_url(base, "chat/completions"), {"model": "vlm", "ą": "ę"}, api_key="k")
        assert resp["choices"][0]["message"]["content"] == "Mapa."
        assert seen == {"get": "/v1/models", "post": "/v1/chat/completions", "auth": "Bearer k",
                        "body": {"model": "vlm", "ą": "ę"}}
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)
    assert vision.check_server(base, timeout=1) is not None  # closed now


# ------------------------------------------------------------------------------------------ script


def _script():
    spec = importlib.util.spec_from_file_location("describe_images", ROOT / "scripts" / "describe_images.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_dry_run_and_run(exam, monkeypatch, no_tesseract, tmp_path):
    d, _ = exam
    script = _script()
    cache = tmp_path / "runs" / "desc.json"

    def boom(*a, **k):
        raise AssertionError("dry run must not send requests")

    monkeypatch.setattr(vision, "_http_json", boom)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert script.main(["--exam-dir", str(d), "--cache", str(cache), "--dry-run"]) == 0
    out = buf.getvalue()
    assert "3 unique images, 0 cached, 3 to describe" in out and "planned" in out and "Nie rozwiązuj" in out
    assert not cache.exists()

    monkeypatch.setattr(vision, "_http_json", FakeVLM())
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"images/A.png": "Ilustracja. Napis „RED”."}, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert script.main(["--exam-dir", str(d), "--cache", str(cache), "--parallel", "2",
                            "--reference", str(ref)]) == 0
    out = buf.getvalue()
    assert "3 described" in out and "(= images/C.png)" in out and "labels 1.00" in out
    assert set(vision.load_descriptions(cache, d, _)) == {"images/A.png", "images/B.png", "images/C.png", "images/D.png"}

    monkeypatch.setattr(vision, "_http_json", boom)  # everything cached: no server needed
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert script.main(["--exam-dir", str(d), "--cache", str(cache)]) == 0
    assert "3 cached" in buf.getvalue()


def test_script_exit_codes(exam, monkeypatch, no_tesseract, tmp_path):
    d, _ = exam
    script = _script()
    cache = str(tmp_path / "desc.json")

    def down(url, payload=None, timeout=None, api_key=None):
        raise vision.VisionError("cannot reach")

    monkeypatch.setattr(vision, "_http_json", down)
    with redirect_stdout(io.StringIO()):
        assert script.main(["--exam-dir", str(d), "--cache", cache]) == 2
    monkeypatch.setattr(vision, "_http_json", FakeVLM(fail_on="blue"))
    with redirect_stdout(io.StringIO()):
        assert script.main(["--exam-dir", str(d), "--cache", cache]) == 1
    with redirect_stdout(io.StringIO()):
        assert script.main(["--exam-dir", str(tmp_path / "nope"), "--cache", cache]) == 2


@pytest.mark.skipif(not os.environ.get("MATURA_MOCK_DIR"), reason="set MATURA_MOCK_DIR to the unzipped mock package")
def test_real_mock_package_images():
    d = pathlib.Path(os.environ["MATURA_MOCK_DIR"])
    items = json.loads((d / "exam.json").read_text(encoding="utf-8"))["items"]
    missing = []
    refs = vision.collect_image_refs(d, items, missing=missing)
    assert missing == [] and all(r.declared_sha256 == r.sha256 for r in refs)
    listed = {vision.normalize_image_path(i["path"]) for it in items for i in it["images"]}
    assert {p for r in refs for p in r.paths} == listed
    for r in refs:
        assert len(vision.build_vlm_prompt(r)) < 6000
