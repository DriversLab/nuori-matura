# Polish data, Wikipedia fetching and near-duplicate detection: research report (Tasks A–C)

**Summary:**
- **A:** train on `openeurollm/EU-Instruct-Synthetic` (config `pl`, filtered). Use `NASK-PIB/PLLuM-Align` for the held-out check. Both download anonymously, and I found 0 exact prompt overlaps between them.
- **B:** use MediaWiki TextExtracts with `explaintext=1`, one title per request. It returns the whole article; the 157k-char `Polska` page came back complete. Use `prop=pageprops` in batches of 50 to resolve titles, and `list=categorymembers` to find titles.
- **C:** no model separates paraphrases from same-topic questions on question text alone. `BAAI/bge-m3` works on "question stem (options removed) + gold answer", with the rule `cos_QA ≥ 0.80 OR cos_Q ≥ 0.90`. On my test pairs it flagged 10/10 paraphrases and 0/6 same-topic questions.

All scratch scripts are in `<research scratch>/abc/` (local research workspace, not committed). Other agents also write to `research/`, and my first `meta.py` there has since been overwritten by someone else.

---

## TASK A: Polish instruction data (regression guard + held-out check)

### Candidates (all checked on the Hub API, first-rows API and local download; no token)

| Dataset id | Gated | License | Rows | Fields | Origin / quality notes |
|---|---|---|---|---|---|
| `openeurollm/EU-Instruct-Synthetic` (config `pl`) | no | apache-2.0 | 146,247 (pl) | `messages` [user, assistant] (always 2 turns), `language` | **Synthetic, written directly in Polish, not translated.** The card only names the `synthgen-if` pipeline; the generator LLM is UNVERIFIED. Modern, fluent answers (median: prompt 300 chars, answer 854). Heavy on format constraints: ~7% of prompts mention "litera", 6% "JSON". Artifacts: 2.6% of answers contain `Final thought`, and some have English bold headers (`**Caveats**`, `**Details**`). My filter keeps 141,924 rows. Single file `pl/train.parquet`, 157 MB. |
| `NASK-PIB/PLLuM-Align` | no | cc-by-sa-4.0 | `dialogs.jsonl` 1,989 · `ranking.jsonl` 1,818 · `rating.jsonl` 500 | `id`, `chosen` [msgs], `rejected` [msgs]; `rating.jsonl` adds `chosen_rating`, `rejected_rating`, `chosen_rating_details`, `rejected_rating_details` | **Human-annotated native Polish** (EMNLP 2025 paper). ranking + rating give **1,019 unique single-turn prompts**; median chosen answer 344 chars. Categories: neut 195, neut-gen 195, long 114, contr 102, neut-nonpl 96, ethic 52, ~120 `polqa_*` (from PolQA), antropic 28 (HH-style), red_teaming 24, toxic 17, reasoning 11. `dialogs.jsonl` covers only 100 multi-turn dialogues. The three files have different schemas, so datasets-server can't convert them and `load_dataset(id)` fails; load each file with `data_files=`. |
| `mmosiolek/pl_alpaca_data_cleaned` | no | cc-by-4.0 (the source Alpaca data is CC BY-NC and OpenAI-generated) | 51,715 (`pl_alpaca_data_cleaned.json`) + 252 (`pl_user_oriented_instructions.json`: `id, motivation_app, instruction, input, expected_output`) | `instruction`, `input`, `output` | **Machine-translated** by GPT-3.5-Turbo from yahma/alpaca-cleaned. Short answers (median 200 chars); 32,060 rows have empty `input`. Keeps Alpaca's errors, e.g. "Znajdź całkowity dochód … 100 długopisów" answered "100 dolarów". No datasets-server parquet (conversion failed). |
| `chrisociepa/raw-self-generated-instructions-pl` | no | cc-by-4.0 | 55,125 (`data_pl.json`, which is actually JSONL) + `seed_data_pl.json` | `instruction`, `input`, `output` | **Generated in Polish by GPT-3.5-Turbo (self-instruct), not translated.** Median output 211 chars, simple. Only 62 exact instruction matches with mmosiolek. datasets-server parquet exists. |
| `chrisociepa/self-generated-instructions-pl` | no | cc-by-4.0 | 104,527 | `instruction, input, output, prompt` | **Union of the two rows above.** Never pair it with either one. |
| `CohereLabs/aya_dataset`, filtered on `language_code=="pol"` | no | apache-2.0 | 1,483 train / 0 test | `inputs, targets, language, language_code, annotation_type, user_id` | Human-written: 448 original annotations, 1,035 re-annotations of templated data, 19 annotators. Some targets are tiny (e.g. `"3)."`). |
| `openeurollm/Dolci-Instruct-SFT-translated` (config `pl`) | no | apache-2.0 | 494,773 | `id`, `messages` | **Machine-translated.** Visible mistakes, e.g. thymus ("grasica") rendered as "Oznaczenie czasu". 50 shards, 673 MB. |
| `JohnTdi/polish-llm-sft-pl` / `JohnTdi/bielik-distill-polish-10k` | no | apache-2.0 | 38,781 / 10,304 | `messages` | Synthetic. The Bielik-distilled set has clear hallucinations (the PBS-1 row). Being Bielik self-distillation, it is a poor independent check. |
| `OpenAssistant/oasst2`, filtered on `lang=="pl"` | no | apache-2.0 | 431 messages, 54 trees | tree format | Human-written but too small. |

**Excluded:**
- `s3nh/alpaca-dolly-instruction-only-polish` (23,687) and `Lajonbot/alpaca-dolly-chrisociepa-instruction-only-polish` (48,637): no license. 97.8% of s3nh instructions are wrapped in literal quotes, and some inputs are junk ("brak tłumaczenia…").
- `saillab/alpaca-polish-cleaned`: card says CC BY-NC and Google Translate; inputs contain the string `"nan"`.
- `grappeq/alpaca-polish-gemma3-translation`: cc-by-nc-4.0.
- `pelcra/PLLuMIC` (1,278 hand-written instructions, cc-by-sa): `gated=auto`, so the anonymous download of `pllumic.json` returns **HTTP 401**.
- `sdadas/gpt-exams`: cc-by-nc-sa and exam-style, so it risks overlapping the matura eval. I used it only to calibrate Task C.
- `speakleash` and `CYFRAGOVPL` publish no Polish instruction datasets on the Hub (`author=` API search; speakleash only has `PES-2018-2022`).

### Recommended disjoint pair
- **Training mix:** `openeurollm/EU-Instruct-Synthetic`, config `pl`, with the artifact filter below.
  - Fallback: `chrisociepa/raw-self-generated-instructions-pl`. It is native Polish but has weaker, shorter answers.
- **Held-out general-capability loss:** `NASK-PIB/PLLuM-Align`, taking `chosen` from `ranking.jsonl` and `rating.jsonl`, deduplicated by prompt (1,019 items). Compute NLL on assistant tokens only.
  - Fallback: the Polish subset of `CohereLabs/aya_dataset`.
- **Why this pair:** different producers (synthetic pipeline vs. human annotators), both native Polish, and 0 exact normalized prompt overlaps (verified).

### Verified download code (ran anonymously, HF_TOKEN unset, `get_token() is None`)
```python
import re
from datasets import load_dataset

train_src = load_dataset("openeurollm/EU-Instruct-Synthetic", "pl", split="train")        # 146,247 rows, 9.3s
ARTIFACT = re.compile(r"Final thought|\*\*(Caveats|Details|Summary|Overview|Answer)\*\*")
def ok(ex):
    a = ex["messages"][1]["content"]
    return not ARTIFACT.search(a) and 20 <= len(a) <= 6000
train_clean = train_src.filter(ok)                                                          # 141,924 rows

# per-file: load_dataset("NASK-PIB/PLLuM-Align", split="train") -> DatasetGenerationError (mixed schemas)
rank = load_dataset("NASK-PIB/PLLuM-Align", data_files="ranking.jsonl", split="train")      # 1,818
rate = load_dataset("NASK-PIB/PLLuM-Align", data_files="rating.jsonl",  split="train")      # 500
pairs = {}
for ds in (rank, rate):
    for ex in ds:
        m = ex["chosen"]
        pairs.setdefault(m[0]["content"], [{"role": x["role"], "content": x["content"]} for x in m])
heldout = [{"messages": m} for m in pairs.values()]                                         # 1,019

# also verified:
load_dataset("mmosiolek/pl_alpaca_data_cleaned", data_files="pl_alpaca_data_cleaned.json", split="train")  # 51,715
load_dataset("chrisociepa/raw-self-generated-instructions-pl", data_files="data_pl.json", split="train")    # 55,125
load_dataset("CohereLabs/aya_dataset", split="train").filter(lambda x: x["language_code"] == "pol")         # 1,483
```
Other verified download routes:
- **Raw files:** `hf_hub_download(repo_id, filename, repo_type="dataset", token=False)` worked for all 12 files I tried, including `NASK-PIB/PLLuM-Align/{dialogs,ranking,rating}.jsonl`, `openeurollm/EU-Instruct-Synthetic/pl/train.parquet` and the oasst2 and aya parquets.
- **datasets-server parquet:** `https://huggingface.co/datasets/openeurollm/EU-Instruct-Synthetic/resolve/refs%2Fconvert%2Fparquet/pl/train/0000.parquet` (156,987,535 bytes); `https://huggingface.co/api/datasets/openeurollm/EU-Instruct-Synthetic/parquet/pl/train/0.parquet` returns HTTP 200.
- **Not available via `/parquet`:** PLLuM-Align and mmosiolek (conversion "failed").

---

## TASK B: Full plain text from Polish Wikipedia (live-tested from this machine)

### Findings
- **User-Agent is enforced:**
  - The default `python-requests/…` UA gets **HTTP 403** with the body *"Please set a user-agent and respect our robot policy https://w.wiki/4wJS"*. That link redirects to `https://wikitech.wikimedia.org/wiki/Robot_policy`.
  - An empty UA also gets 403.
  - A descriptive UA (`name/version (contact) python-requests/x`) gets 200.
  - The API:Etiquette page (fetched live) asks for the format `clientname/version (contact information e.g. username, email) framework/version`.
- **Which method returns full text:**

| Method | Batching | Result |
|---|---|---|
| `prop=extracts&explaintext=1` (TextExtracts, no `exintro`/`exchars`/`exsentences`) | **1 title per request.** With several titles you get the warning `"exlimit" was too large for a whole article extracts request, lowered to 1.`; only one page gets `extract` and the response carries `continue.excontinue` | **Full article, not truncated.** `Polska` = 157,061 chars. 267 of the 273 `<p>` paragraphs from `action=parse` are present; the 6 missing are infobox, navbox or quote-block content. `[n]` reference markers are removed. Headings (default `exsectionformat=wiki`) look like `\n\n\n== Życiorys ==\n`; `plain` gives a bare heading line; `raw` gives `\ufffd\ufffd2\ufffd\ufffdHeading`. The `Przypisy` body is empty, but `Bibliografia` and `Linki zewnętrzne` text is kept, so drop those sections by heading. About 0.25 s per request (10 titles took 2.46 s). **Recommended.** |
| `prop=cirrusdoc` | 50 titles/request (4.08 s) | `cirrusdoc[0].source.text` is full text **without headings**, but it includes reference text (ISBN appears 16× vs 0× in the extract) and authority-control junk at the end. Also has `opening_text`, `heading`, `auxiliary_text`, `category`, `wikibase_item`. It is a CirrusSearch internal API, so long-term stability is UNVERIFIED. |
| `prop=revisions&rvprop=content&rvslots=main` | 50/50 in one request | Returns wikitext only, which needs a parser (`mwparserfromhell` is not installed). |
| `action=parse&prop=text` / REST `/api/rest_v1/page/html/<Title>` | 1 page | HTML only (278,607 chars and 482,649 bytes for Adam Mickiewicz), so you would have to strip it yourself. |
| `generator=categorymembers` + `prop=extracts&exintro=1&exlimit=20` | 20/request | Returns 20 lead sections (intros only). |

- **Title handling** (`formatversion=2`, `redirects=1`):
  - **Redirects:** `query.redirects=[{"from":"Konstytucja trzeciego maja","to":"Konstytucja 3 maja"}]`.
  - **Normalization:** `query.normalized=[{"from":"kopernik","to":"Kopernik"}]`.
  - **Missing pages:** `{"title":"Konstytucja 3 Maja","missing":true}`. That capital-M title does not exist; the real one is `Konstytucja 3 maja`.
  - **Invalid titles:** `{"title":"Foo[bar]","invalid":true,"invalidreason":…}`.
  - **Disambiguation pages:** `prop=pageprops&ppprop=disambiguation` gives `"pageprops":{"disambiguation":""}` (e.g. `Kopernik`).
- **maxlag:** forcing it with `maxlag=-1` returns **HTTP 200**, header `Retry-After: 5`, body `{"error":{"code":"maxlag",…}}`. Send `maxlag=5` on every request, send requests one at a time (not in parallel), use `Accept-Encoding: gzip`, and on `maxlag`, 429 or 503 wait the `Retry-After` value.
- **categorymembers works:** `cmtitle=Kategoria:Polscy poeci&cmnamespace=0&cmlimit=500` returns 500 titles plus `continue.cmcontinue`. The full walk below returned 1,872 titles in 2.5 s; the category has 8 subcategories.

### Verified snippet (`abc/plwiki.py`; its test run printed the results listed after it)
```python
import re, time, requests
API = "https://pl.wikipedia.org/w/api.php"
S = requests.Session()
S.headers.update({"User-Agent": "MaturaFT-data/0.1 (https://github.com/<you>/<repo>; <you>@example.org) python-requests/" + requests.__version__,
                  "Accept-Encoding": "gzip"})

def api(**params):
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for attempt in range(6):
        r = S.get(API, params=params, timeout=60)
        if r.status_code in (429, 503) or (r.ok and r.json().get("error", {}).get("code") == "maxlag"):
            time.sleep(int(r.headers.get("Retry-After", 5)) * (attempt + 1)); continue
        r.raise_for_status(); j = r.json()
        if "error" in j: raise RuntimeError(j["error"])
        return j
    raise RuntimeError("gave up after retries")

def category_titles(category, recursive_depth=0, _seen=None):
    _seen = _seen if _seen is not None else set(); out, cont = [], {}
    while True:
        j = api(action="query", list="categorymembers", cmtitle=category, cmtype="page|subcat", cmlimit="max", **cont)
        for m in j["query"]["categorymembers"]:
            if m["ns"] == 0: out.append(m["title"])
            elif m["ns"] == 14 and recursive_depth > 0 and m["title"] not in _seen:
                _seen.add(m["title"]); out += category_titles(m["title"], recursive_depth - 1, _seen)
        if "continue" not in j: return list(dict.fromkeys(out))
        cont = {"cmcontinue": j["continue"]["cmcontinue"]}

def resolve(titles):  # <=50 titles per call
    q = api(action="query", titles="|".join(titles), redirects=1, prop="pageprops", ppprop="disambiguation")["query"]
    mapping = {t: t for t in titles}
    for n in q.get("normalized", []): mapping = {k: (n["to"] if v == n["from"] else v) for k, v in mapping.items()}
    for rd in q.get("redirects", []): mapping = {k: (rd["to"] if v == rd["from"] else v) for k, v in mapping.items()}
    pages = {p["title"]: p for p in q["pages"]}
    return {o: f for o, f in mapping.items()
            if not (pages.get(f, {}).get("missing") or pages.get(f, {}).get("invalid")
                    or "disambiguation" in pages.get(f, {}).get("pageprops", {}))}

DROP_SECTIONS = {"Przypisy", "Bibliografia", "Linki zewnętrzne", "Zobacz też", "Uwagi", "Galeria", "Literatura", "Źródła", "Bibliografia uzupełniająca"}
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)

def full_plaintext(title):  # whole-article extracts: 1 title/request
    p = api(action="query", prop="extracts", explaintext=1, exsectionformat="wiki", titles=title, redirects=1)["query"]["pages"][0]
    return None if p.get("missing") or p.get("invalid") else p.get("extract", "")

def clean(extract, keep_headings=False):
    parts, pos = [], 0
    for m in HEADING.finditer(extract):
        parts += [(None, extract[pos:m.start()]), ((len(m.group(1)), m.group(2)), None)]; pos = m.end()
    parts.append((None, extract[pos:]))
    out, skip = [], None
    for head, body in parts:
        if head:
            lvl, name = head
            if skip is not None and lvl <= skip: skip = None
            if skip is None and name in DROP_SECTIONS: skip = lvl
            if skip is None and keep_headings: out.append(f"\n{name}\n")
        elif skip is None: out.append(body)
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).replace("\xa0", " ").strip()
```
**Test run output:**
- `resolve([...])` returned `{'Konstytucja trzeciego maja': 'Konstytucja 3 maja', 'adam Mickiewicz': 'Adam Mickiewicz', 'Pan Tadeusz': …, 'Polska': …}`. `Kopernik` (disambiguation) and `Konstytucja 3 Maja` (missing) were dropped.
- Clean lengths: `Polska` 157,061 → 144,542 chars; `Konstytucja 3 maja` 34,584 → 31,602.
- The cleaned text contains no `== ` and no "Przypisy".

---

## TASK C: Embedding model for near-duplicate detection

**Setup:**
- sentence-transformers 6.0.1, `SentenceTransformer(name, device="mps")`, `normalize_embeddings=True`.
- Test set: 10 paraphrase pairs (PARA), 6 same-topic pairs (SAME), 4 template variants (HARD: pole vs obwód, r=5 vs r=7, different equation, the same 4 options with a different author) and 4 unrelated pairs (UNREL). All are hand-written and include MC variants.
- Calibration sets:
  - 900 `sdadas/gpt-exams` questions: 60 domains × 15 questions, giving 6,300 same-domain pairs and 398,250 cross-domain pairs.
  - `sdadas/ppc` test split: exact paraphrase (label 1) vs non-paraphrase (label 3).

### Size and load time (M3 Max)
| Model | Params | Dim / max_seq | HF cache on disk | First load incl. download | Load from cache (MPS) |
|---|---|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | 384 / 128 | 458 MB | 23.7 s | 2.3 s |
| `intfloat/multilingual-e5-small` | 118M | 384 / 512 | 470 MB | 19.0 s | 1.8 s |
| `intfloat/multilingual-e5-base` | 278M | 768 / 512 | 1.1 GB | 29.0 s | 2.7 s |
| `sdadas/mmlw-roberta-base` | 124M | 768 / 512 (CLS pooling) | 478 MB | 17.5 s | 0.8 s |
| `BAAI/bge-m3` | 568M | 1024 / 8192 | **4.3 GB** (`pytorch_model.bin` 2.1 GB plus a separately fetched 2.1 GB `model.safetensors` snapshot) | 61.0 s | 7.8 s (3.1 s on a later run); encodes 2,000 short texts in 6.9 s |

### Cosine similarity on the question text only
Column key:
- **para:** "W którym roku uchwalono Konstytucję 3 maja?" vs "Kiedy uchwalono Konstytucję 3 Maja? Podaj rok."
- **same:** the same question vs "Który organ uchwalił Konstytucję 3 maja?"
- **unrel:** "Ile wynosi pH czystej wody?" vs "Kto napisał Pana Tadeusza?"
- **MC para:** the MC question vs a reworded MC with options reordered
- **MC vs stem:** the MC question vs the same question without options
- **MC same-topic:** the MC question vs the "which body passed it" MC
- **MC same options:** Mickiewicz vs Słowacki, with identical options

| Model / prefix | para | same | unrel | MC para | MC vs stem | MC same-topic | MC same options | PARA min | SAME max | margin | PPC AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM-L12 (none) | .980 | .930 | -.002 | .963 | .887 | .791 | .791 | .770 | .930 | **-.160** | .929 |
| e5-small `query: ` | .971 | .966 | .689 | .956 | .948 | .902 | .956 | .865 | .966 | -.101 | .850 |
| e5-small (none) | .970 | .968 | .729 | .978 | .956 | .926 | .959 | .919 | .968 | -.049 | .872 |
| e5-base `query: ` | .972 | .962 | .699 | .968 | .904 | .899 | .943 | .871 | .962 | -.091 | .888 |
| e5-base (none) | .956 | .959 | .730 | .962 | .902 | .908 | .939 | .888 | .959 | -.071 | .882 |
| mmlw-roberta-base `zapytanie: ` | .992 | .995 | .917 | .988 | .978 | .961 | .992 | .953 | .995 | -.042 | .733 |
| mmlw-roberta-base (none) | .989 | .993 | .886 | .981 | .968 | .943 | .987 | .918 | .993 | -.075 | .738 |
| bge-m3 (none) | .910 | .895 | .338 | .919 | .667 | .667 | .869 | .667 | .895 | -.228 | **.940** |

**No question-only threshold works for any model.** "W którym roku…" vs "Który organ uchwalił…" scores at least as high as several true paraphrases in every model.
- **mmlw-roberta-base:** even unrelated pairs land at 0.86–0.93, so it is unusable here.
- **e5 models:** unrelated pairs are already ~0.69–0.86.
- **bge-m3:** widest spread (unrelated 0.23–0.38) and the best PPC AUC.

### Fix: embed "question + gold answer" (`f"{q}\nOdpowiedź: {a}"`)
Here the answer is the option text, not the letter. Margins (PARA min − SAME max):

| Model | Margin |
|---|---|
| **bge-m3** | **+.146** (.842 vs .696) |
| e5-small `query:` | +.056 (.963 vs .907) |
| e5-base `query:` | +.047 |
| e5-small (none) | +.037 |
| MiniLM | -.017 |
| mmlw | -.022 / -.017 |

Removing the option list before embedding also helps. With the options still in, the "same options, different author" pair scores 0.871; stripped, it drops to 0.508.

### Recommendation
- **Model:** `BAAI/bge-m3`. Dense embeddings, **no prefix**, normalized.
- **What to embed:** the question stem with MC options removed, plus the gold answer text.
- **Rule:** flag if `cos_QA ≥ 0.80` **or** `cos_Q(stem) ≥ 0.90`.
- **Result on the test set:** PARA 10/10, SAME 0/6, UNREL 0/4. HARD 1/4 flagged ("pole" vs "obwód" koła, r=5: QA .844, Q .908).
- **At scale (open-ended gpt-exams questions, answer cut to its first sentence):**
  - QA ≥ 0.80 flags 4.0% of same-domain pairs and 20 of 398k cross-domain pairs.
  - QA ≥ 0.85 flags 1.2%; Q ≥ 0.90 flags 0.35%.
  - Pairs ≥ 0.93 are real duplicates ("podstawowe/główne metody wyceny zapasów").
  - The 0.80–0.90 QA band is mostly same-topic ("patologia układu hormonalnego" vs "rozrodczego", .865). That over-flagging is fine when you are dropping synthetic items; raise the QA threshold to 0.85 if you want to keep more.
- **Lighter option:** `intfloat/multilingual-e5-small` with `"query: "` on both sides (its model card says to use `query:` for symmetric tasks) on the same Q+A text. The threshold is ≈0.935 and the margin is narrow (0.056), so treat it as UNVERIFIED at scale.
- **`sdadas/mmlw-roberta-base`** needs the `"zapytanie: "` prefix (per its card) but performed worst here.

### Verified snippet (`abc/dedup_final.py`)
```python
import re, torch
from sentence_transformers import SentenceTransformer

def strip_options(q: str) -> str:
    m = re.search(r"\s(?:\(?A[.)])\s", q)            # option list starts at first "A." / "A)" / "(A)"
    stem = q[:m.start()] if m else q
    stem = re.sub(r"\s*(Wybierz|Wskaż) (poprawną|prawidłową) odpowiedź:?\s*$", "", stem.strip())
    return stem.strip().rstrip(":")

def qa_text(q, a): return f"{strip_options(q)}\nOdpowiedź: {a}"   # a = answer TEXT, not the letter

model = SentenceTransformer("BAAI/bge-m3", device="mps" if torch.backends.mps.is_available() else "cpu")
def embed(texts): return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_tensor=True)

def flag_near_duplicates(synth, evalset, thr_qa=0.80, thr_q=0.90):
    """synth/evalset: lists of (question, answer_text). Returns (max_qa, max_q, nearest_eval_idx, flagged) per synth item."""
    Sq, Eq = embed([strip_options(q) for q, _ in synth]), embed([strip_options(q) for q, _ in evalset])
    Sqa, Eqa = embed([qa_text(q, a) for q, a in synth]), embed([qa_text(q, a) for q, a in evalset])
    best_qa, idx = (Sqa @ Eqa.T).max(dim=1)
    best_q = (Sq @ Eq.T).max(dim=1).values
    flagged = (best_qa >= thr_qa) | (best_q >= thr_q)
    return best_qa.cpu().numpy(), best_q.cpu().numpy(), idx.cpu().numpy(), flagged.cpu().numpy()
```
**Caveats:**
- `strip_options` expects a space before `A.` or `A)`. A stem with an initial like "Jan A. Kowalski" would be cut early.
- In sentence-transformers 6.0.1, `get_sentence_embedding_dimension()` still works but warns that it is now `get_embedding_dimension()`.
- The HF Hub prints "You are sending unauthenticated requests" on every anonymous download; it is harmless.