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
.env                       UNPAYWALL_EMAIL=… for the Unpaywall API (not committed)
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
