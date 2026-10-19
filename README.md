# Trading Research Papers

A local library of academic papers on algorithmic, high-frequency and AI-driven trading,
built from Paper Digest's
[Recent Papers on Algorithmic Trading & High-Frequency Trading](https://resources.paperdigest.org/2020/04/recent-papers-on-algorithmic-trading-high-frequency-trading/)
list (1,000 papers, ranked, dated 2022 – 2026).

The scripts fetch the list, find a legal open-access PDF for each paper where one exists, and
keep a manifest recording what was downloaded, from where, and why the rest weren't.
Nothing here bypasses paywalls or CAPTCHAs; papers without an open copy are left for manual
retrieval (e.g. through a library).

## Status (2026-10-15)

| Status       | Count | Meaning                                        |
| ------------ | ----: | ---------------------------------------------- |
| `downloaded` |   655 | PDF saved in `papers/`                         |
| `not_found`  |   343 | No accessible PDF; `note` says why (see below) |
| `error`      |     1 | Downloader crashed on this row (rank 277)      |

The `not_found` notes break down roughly as: paywalled publishers (Springer, Elsevier,
Wiley, T&F, SAGE, INFORMS, IEEE…), 32 behind a CAPTCHA even in a normal browser (all
ScienceDirect entries plus ranks 90 and 718), and a few dead or retitled entries.

## Repository layout

```
main.py  Step 1: parse the Paper Digest list and download open-access PDFs
browser-download.py        Step 2: retry `not_found` entries in a real (Playwright) browser
paperlib.py                Shared by both: PDF title check and manifest writer (JSON + CSV)
download-papers.py         Legacy arXiv-only scraper (superseded by paperdigest-downloader.py)
main.py                    Placeholder entry point (unused)
docs/research/             Research notes drawing on the corpus (cited)
docs/plans/                Build plans for systems based on that research
papers/                    PDFs + manifest.json / manifest.csv (git-ignored, ~1.1 GB)
trader/                    The LLM-assisted trading system (see "Trading system" below)
tests/                     pytest suite for trader/ (offline; no keys needed)
data/                      trader/'s point-in-time parquet store (git-ignored)
.env                       Keys and contact emails (not committed; see Setup)
```

The research note [What it takes to build LLM-based algorithmic trading](docs/research/llm-algo-trading.md)
draws on this corpus. Its implementation plan is
[LLM trading build plan](docs/plans/llm-trading-build-plan.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # requests, beautifulsoup4, playwright, pymupdf
uv run playwright install chromium       # browser for step 2
```

Set `UNPAYWALL_EMAIL` in `.env`. Unpaywall finds many more open-access copies and requires
a contact email. Scripts don't read `.env` themselves, so pass it to uv with `--env-file`.

The trading system also reads `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` and `APCA_API_BASE_URL`
(Alpaca **paper** account only; anything else is refused), `SEC_USER_AGENT` (`"Name email"`, which
EDGAR requires), and optionally `LLM_BASE_URL` (the local model server; default `http://localhost:1234/v1`).
`ANTHROPIC_API_KEY` is only needed if the tagger moves to a hosted model, or for the Phase 4 reviewer.

## Trading system (`trader/`)

Built phase by phase from the [build plan](docs/plans/llm-trading-build-plan.md). Done so far:

- **Phase 0:** `trader/config.py` holds the frozen 50-name universe (SPY is the regime reference), the
  price-only core strategy, hard risk limits, pinned model IDs and their training cutoffs.
- **Phase 1:** point-in-time data in `data/<table>/*.parquet`. Every row has an `available_at` (UTC), and
  features read only through `trader.data.store.as_of(table, t)`, which never returns a later row.
- **Phase 2:** a local LLM tags documents into schema-validated records (`trader/llm/`). It is cached by
  `(doc_id, model, prompt_version)` and labelled pre- or post-cutoff; there is also an anonymised arm.

```bash
uv run pytest                                                  # offline test suite
uv run --env-file .env python -m trader.data filings           # EDGAR 8-K/10-Q/10-K since 2016
uv run --env-file .env python -m trader.data prices            # Alpaca daily bars (needs Alpaca keys)
uv run --env-file .env python -m trader.data news              # Alpaca/Benzinga news (needs Alpaca keys)
uv run --env-file .env python -m trader.data count-news 2026-10-15   # sanity check: ~130+/day expected
```

Every refresh is incremental and can be re-run safely. Re-runs add nothing new, and an interrupted news
backfill resumes from the newest stored article.

Timestamp sources (`available_at`):

| Table     | `available_at`                                      | Note                                                                    |
| --------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| `prices`  | 16:00 New York on the session date                  | Bars are split- and dividend-adjusted; history re-fetched when restated |
| `news`    | Benzinga `created_at`                               | Edited articles keep `created_at` but store the latest text             |
| `filings` | EDGAR acceptance time from the filing's `.hdr.sgml` | See below                                                               |

**EDGAR's JSON acceptance time is unreliable.** In `data.sec.gov/submissions`, `acceptanceDateTime` ends
in `Z`, but for about two thirds of the universe (AAPL, MSFT, …) the value is UTC *plus* the Eastern
offset. For the rest (NFLX, GE, TSLA, …) it is true UTC. This was checked on 2026-10-19 against the
filings' index pages. Each filing's SGML header gives the true time (US/Eastern), so `filings.py` fetches
it once per new filing. If the header is missing, it falls back to the later of the two JSON readings,
flagged `accepted_source = json-latest`.

### Local LLM (Phase 2)

The tagger runs on a local OpenAI-compatible server before any hosted model is paid for. The default is
llama.cpp's `llama serve` (as installed with Unsloth) inside WSL, serving Unsloth's GGUF of Gemma 4 E4B
(`unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`, 5 GB, CPU). Its pre-training data ends in January 2025, so news from
February 2025 on counts as post-cutoff evidence: about 21 months, against about 4 for Claude 5.5. The server's
model ID must equal `config.TAGGER_MODEL` and have a cutoff in `config.MODEL_CUTOFFS`, or the run is refused.

```bash
llama serve -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_M -c 16384 -np 1 --cache-ram 0 --no-mmproj --port 9931
# .env: LLM_BASE_URL=http://localhost:9931/v1
uv run --env-file .env python -m trader.llm check     # pinned model served + valid JSON on a sample
uv run --env-file .env python -m trader.llm papers    # shakedown -> docs/research/corpus-llm-papers.csv
uv run --env-file .env python -m trader.llm news      # tags stored news after the cutoff; --since for more
```

`-c 16384` fits the papers (about 8k tokens each). `--cache-ram 0` turns off llama.cpp's RAM cache of past
prompts, 8 GB by default, which grew until the machine swapped. `--no-mmproj` skips the image and audio
encoders. The server then uses about 7 GB.

Pick a model that fits in memory. WSL has 15 GB of RAM and no GPU access (the GPU is AMD). A 29 GB Q8_0 27B
model pages from disk at well under one token a second and starves the whole machine.

For GPU speed, serve from Windows instead: LM Studio, or a Vulkan build of llama.cpp. WSL can't reach a server
listening on Windows' `localhost` by default. Either add `networkingMode=mirrored` under `[wsl2]` in
`C:\Users\<you>\.wslconfig` and run `wsl --shutdown`, or let the server listen on the local network and
point `LLM_BASE_URL` at the WSL default gateway (`ip route`).

`docs/research/corpus-llm-papers.csv` is the shakedown's output: 65 papers on CPU, all schema-valid on the
first attempt, ~40 s each. It is machine-extracted and unverified. A spot check found right test windows
(TradingAgents, FinMem, Trading-R1) but wrong `evaluated_models` for RL papers (algorithms, not language
models) and some guessed cutoff answers, so check a row against its paper before citing it.

Invalid output gets up to two repair turns that quote the validation error, then is quarantined in
`data/<table>_quarantine/`. Delete a quarantine file to retry those documents. Server errors abort the run
and keep what finished. Every run appends tokens and seconds per 1,000 documents to `data/llm_runs.jsonl`.
`--workers N` sends N requests at once, which helps only if the server is set to serve requests in parallel.

## Pipeline

### 1. List and download: `main.py`

```bash
uv run --env-file .env python paperdigest-downloader.py            # full run
uv run --env-file .env python paperdigest-downloader.py --since 2026-01-01 --limit 50
uv run python paperdigest-downloader.py --dry-run                   # list only, write manifest
uv run python paperdigest-downloader.py --html saved_page.html      # if the site blocks scripts
```

For each row it tries a chain of resolvers until one returns a real PDF:

1. Source-specific rules: arXiv, OpenReview, ACL Anthology, ACM DL, PubMed → PMC, NeurIPS, GitHub README → arXiv
2. [Unpaywall](https://unpaywall.org/) open-access locations (needs the email)
3. [Semantic Scholar](https://api.semanticscholar.org/) `openAccessPdf` / arXiv id
4. Landing-page scrape (`citation_pdf_url` meta tag, PDF links)
5. arXiv title search for a preprint of a paywalled paper

A PDF is accepted only if the paper's title appears on its first two pages (see step 2 for why).
Requests are rate-limited per host.

Re-running resumes: `downloaded`/`recovered` rows are left exactly as they are and everything
else is retried. Results are **merged** into the existing manifest, so rows outside
`--since`/`--limit` are kept, and an entry that still isn't found keeps its `browser:` note
and accumulated `tried` list. `--dry-run` only adds rows that are new on the page.

### 2. Browser retry: `browser-download.py`

Many publisher pages need JavaScript or a Cloudflare check before they serve a PDF. This
script opens each `not_found` entry's `view_url` in Chromium, collects candidate PDF links,
and downloads through the browser session.

```bash
xvfb-run -a uv run python browser-download.py                # all not_found entries not yet tried
xvfb-run -a uv run python browser-download.py --retry        # include ones a previous run tried
xvfb-run -a uv run python browser-download.py --ranks 18,20 --workers 1
```

- Runs headed under `xvfb-run` by default; headless Chromium gets blocked more often (`--headless` to force it).
- **Accepts a PDF only if the paper's title appears on its first two pages.** Without this
  check, small journal sites served author templates, copyright forms, or papers from the
  reference list. Page 2 is included because some publishers prepend a cover sheet.
- Entries stuck on a bot check are recorded with the note `browser: blocked by bot check (…)`.
- Saves the manifest after every entry, so it can be interrupted safely.
- Keeps a persistent browser profile in `.browser-profile/` (git-ignored).

Entries the bundled Chromium couldn't get past were then retried by hand in a regular Chrome
session (ACM, SSRN and the open-access titles at Wiley and T&F all work that way) and run
through the same title check.

## Manifest

`papers/manifest.json` is the source of truth: one object per paper, keyed by `rank`.

| Field                                | Description                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| `rank`                               | Position on the Paper Digest list (1–1000)                                     |
| `title`, `authors`, `source`, `date` | As listed by Paper Digest                                                      |
| `view_url`                           | Paper Digest's "View" link (usually a DOI)                                     |
| `status`                             | `downloaded`, `not_found`, `error`, `recovered`                                |
| `pdf_path`                           | File name inside `papers/`                                                     |
| `pdf_url`                            | Where the PDF was actually fetched from                                        |
| `note`                               | Why an entry isn't downloaded; notes starting with `browser:` come from step 2 |
| `tried`                              | Every candidate URL attempted                                                  |

PDFs are named `{rank:04d}_{Title}.pdf`. The title is ASCII-folded, runs of anything
non-alphanumeric become `_`, and the result is cut to 80 characters.

## Caveats

- `manifest.json` is the source of truth; `manifest.csv` is a flattened copy (without
  `tried`) that both scripts rewrite on every save.
- The title check was added on 2026-10-15 and has since been run over every PDF; 9 wrong
  documents (user guides, forms, an ethics policy, cited or unrelated papers) were removed and
  their rows set back to `not_found` with a `rejected wrong PDF` note. Rank 812 was kept: it is
  the right arXiv entry, but the authors later retitled it from "ATMS" to "INTAGS".
- A few accepted files are worth knowing about: ranks 93 and 928 open with a publisher cover
  sheet, and rank 740 may be a one-page preview. Rank 376's SSRN paper has since been
  retitled ("Sleep Disruption and Retail Investor Trading"), so it was not accepted.
