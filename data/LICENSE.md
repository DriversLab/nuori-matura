# Data licences

Licence and source of every data folder. The code is MIT ([LICENSE](../LICENSE)); models, tools and exam links are
listed in [SOURCE.md](../SOURCE.md). Copyrighted exam material (CKE papers and answer keys, the organizers' exam
packages, page scans) and everything derived from it is **not** in this repository; SOURCE.md credits and links it.
No script here downloads CKE papers. The history training rows that come from exam papers are rebuilt from the
teammates' **private** nuori-ai dataset, which only team members can access, so they cannot be rebuilt from this
repository alone. The organizers' packages come from the organizers. Only the public third-party datasets and the
Wikipedia text are fetched by scripts. All of these go into gitignored folders (see "Not in this repository" below).

Licence texts:
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ (legal code:
  https://creativecommons.org/licenses/by-sa/4.0/legalcode)
- Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 (full text at the end of this file)

## Committed data

| Path | Contents | Source | Licence | Rebuild |
|---|---|---|---|---|
| `data/wiki/articles.jsonl`, `titles.yaml`, `fetch_report.json` | 453 Polish Wikipedia articles (plain text), the title list, the fetch report | https://pl.wikipedia.org/ | CC BY-SA 4.0 | `python scripts/fetch_wiki.py` |
| `data/rag/titles_history.yaml` | titles of the Polish Wikipedia articles in the history knowledge base | https://pl.wikipedia.org/ | CC BY-SA 4.0 | the articles and the index: `python scripts/build_rag.py` |
| `data/history/synthetic/*.jsonl` | 560 history exam-style items (80 essays, 480 open and closed) | written by Claude (Anthropic) agents during the hackathon, grounded on Polish Wikipedia and on public-domain documents from Polish Wikisource (https://pl.wikisource.org/); each item lists its `source_urls` | CC BY-SA 4.0 | none (generated once); format and checks in `docs/DATA.md`, "Synthetic items" |
| `data/synthetic/` (`raw/`, `verify/`, `clean.jsonl`, `dedup_report.json`, `key_audit.json`) | 2,744 matura-style items for 8 subjects (2,513 after cleaning), the blind-solver checks and the build reports (earlier 8-subject pipeline) | written by Claude (Anthropic) agents from `data/wiki/articles.jsonl`; each item has `source: "wikipedia:<title>"` | CC BY-SA 4.0 | `scripts/generate_synthetic.py`, then `scripts/build_data.py` |
| `data/eval/heldout.jsonl`, `dev.jsonl`, `verification_log.jsonl` | 60 + 20 matura-style eval items and their solver/critic log (legacy 8-subject eval, earlier pipeline) | written by Claude (Anthropic) agents from Polish Wikipedia text on 16.09.2026, before the hackathon | CC BY-SA 4.0 | none (generated once) |
| `data/eval/general_heldout.jsonl` | 300 Polish instruction/response pairs | NASK-PIB/PLLuM-Align, https://huggingface.co/datasets/NASK-PIB/PLLuM-Align | CC BY-SA 4.0 (upstream) | `python scripts/fetch_external.py` |
| `data/general/train_pool.jsonl` | 6,000 Polish instruction/response pairs | openeurollm/EU-Instruct-Synthetic, config `pl`, https://huggingface.co/datasets/openeurollm/EU-Instruct-Synthetic | Apache-2.0 (upstream) | `python scripts/fetch_external.py` |
| `data/blocklist/README.md` | the list of third-party exam datasets used for leakage filtering, with their licences (no exam text) | our own text, written by `scripts/fetch_external.py` | MIT, like the code | `python scripts/fetch_external.py` |

## Wikipedia-derived data (CC BY-SA 4.0)

- **Attribution.** Text from the Polish Wikipedia (https://pl.wikipedia.org/), by Wikipedia contributors; the authors of
  each article are listed in its page history. Per row:
  - `url`, `pageid` and `revid` in `data/wiki/articles.jsonl` and `tests/fixtures/wiki/*.json`;
  - `source_urls` in `data/history/synthetic/*.jsonl`;
  - `source: "wikipedia:<title>"` in `data/synthetic/` (article `https://pl.wikipedia.org/wiki/<title>`).
  - The eval items in `data/eval/heldout.jsonl` and `dev.jsonl` carry `source: "authored"` and no per-item article.
- **Changes.** Articles are fetched through the MediaWiki API as plain text (TextExtracts, `train/wiki.py`), so markup,
  tables and images are dropped. 40 of the 453 articles in `data/wiki/articles.jsonl` are cut to at most 40,000
  characters and end with `[…]` (`scripts/fetch_wiki.py --max-chars`). The synthetic and eval items are new exam-style
  questions, answers and essays written from these texts; they quote or paraphrase passages of them.
- **Our adaptations** (the synthetic history items, the 8-subject synthetic items, the eval items and their logs) are
  shared under CC BY-SA 4.0 as well (ShareAlike). The Wikisource documents quoted by some history items (the 1791
  constitution, the 1792 Targowica act, the 1794 insurrection act and Połaniec proclamation, the 1863 manifesto,
  Gloger's encyclopaedia) are in the public domain.
- The items were written with a closed LLM at build time only, which the hackathon rules allow; no closed API is used
  in the exam harness.

## Third-party datasets (upstream licences)

- `data/general/train_pool.jsonl`: **openeurollm/EU-Instruct-Synthetic** (config `pl`), **Apache-2.0**. Changes: only
  single-turn rows without generator artefacts and with sane lengths are kept, prompts that overlap the general
  held-out set are dropped, 6,000 rows are sampled (seed 42) and converted to `{id, messages, source}`
  (`train/external.py`). Used as general replay (about 10% of the training rows) to limit forgetting.
- `data/eval/general_heldout.jsonl`: **NASK-PIB/PLLuM-Align**, **CC BY-SA 4.0**. Changes: the `chosen` responses of
  single-turn prompts from `ranking.jsonl` and `rating.jsonl`, 300 rows sampled (seed 42) and converted to
  `{id, messages, source}`. Used only as a loss check for forgetting, never trained.

## Not in this repository (gitignored)

| Path | What | Why it is not committed | How to get it |
|---|---|---|---|
| `data/processed/` | history SFT rows (`data/processed/history/`) and rendered rows of earlier runs | derived from CKE papers and answer keys (© Centralna Komisja Egzaminacyjna) and two Nowa Era trial papers (© Nowa Era) | `python scripts/build_history_data.py` from a checkout of the teammates' private nuori-ai dataset at commit d86d555, plus the gitignored LLM-filled targets (next row). Needs team access to nuori-ai; steps in SOURCE.md, "Team contributions", and `docs/DATA.md` |
| `data/history/llm_filled_answers*.jsonl`, `data/history/filled_*/` | LLM-written targets for CKE tasks | keyed by CKE task ids, many follow the CKE answer key's model answer | written by Claude (Anthropic) agents during the hackathon; kept locally and copied to the GPU box with the built rows (`docs/RUNBOOK.md` 6.1) |
| `data/rag/*` except `titles_history.yaml` | Wikipedia articles, title resolution and BM25 index of the history knowledge base (CC BY-SA 4.0 once built) | size; rebuilt from the title list | `python scripts/build_rag.py` |
| `data/blocklist/*.jsonl`, `data/eval/ext_llmzszl_matura.jsonl` | third-party datasets that redistribute CKE exam text (leakage filtering and secondary eval only, never trained) | © CKE; dataset licences missing or non-commercial | `python scripts/fetch_external.py` (sources and licences in `data/blocklist/README.md`) |
| `data/eval/ext_prawko_*.jsonl` | 65 driving-test questions (prawko-v2 dev and test splits, evaluation only, never trained) | third-party rows; the licence of the question text is not verified by us | `python scripts/import_prawko.py` (source: the Ministry of Infrastructure question bank, https://www.gov.pl/web/infrastruktura/jak-uzyskac-prawo-jazdy; split by https://github.com/stared/train-llm-from-scratch, `datasets/prawko-v2`) |
| `data/synthetic/.emb_cache/` | embedding cache of the dedup step | build cache | `python scripts/build_data.py` |

## Model weights (LoRA adapters)

- Weights are not in this repository (`checkpoints/` and `*.gguf` are gitignored). The base weights are downloaded
  from the links in SOURCE.md.
- Our LoRA adapters for **speakleash/Bielik-4.5B-v3.0-Instruct** (https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct)
  and **speakleash/Bielik-11B-v3.0-Instruct** (https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) are
  **Apache-2.0**, following the Bielik base model's licence. The Bielik GGUF model cards also refer to the Bielik terms
  of use (bielik.ai/terms). A LoRA adapter for google/gemma-4-12b-it, if one is trained, follows the Gemma terms of
  that model instead.
- The adapters are trained on rows derived from CKE exam papers and answer keys (© Centralna Komisja Egzaminacyjna)
  and two Nowa Era trial papers (© Nowa Era), together with the data above. Under the hackathon rules, models built
  from third-party content may be used for **educational and research purposes only**; no commercial use.

## Data outside `data/`

- `tests/fixtures/wiki/Kula.json`, `Prawo_Coulomba.json`: Polish Wikipedia extracts with `url` and `revid`,
  CC BY-SA 4.0, attribution as above.
- `tests/test_organizer_eval.py` and `tests/test_compare_registry.py` quote two prawko-v2 driving-test questions
  (rows 10840 and 4367, in full or shortened) from the Ministry of Infrastructure question bank on gov.pl, taken via
  https://github.com/stared/train-llm-from-scratch (`datasets/prawko-v2`). We have not verified the licence of the
  question text. The quotes are short test fixtures of the organizers' prompt format, credited here, and are not
  covered by the MIT licence of the code.
- The other fixtures (`tests/fixtures/general_sample.jsonl`, `synthetic_sample.jsonl`, `history_exam/`) were written by
  the team for the tests (no CKE material) and are MIT, like the code.

## Apache License 2.0 (full text, for `data/general/train_pool.jsonl`)

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
