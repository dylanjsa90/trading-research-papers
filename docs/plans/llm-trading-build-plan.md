# Build Plan: LLM-Assisted Daily Trading System (Paper First)

_Created 2026-10-16. The evidence and rationale are in
[`docs/research/llm-algo-trading.md`](../research/llm-algo-trading.md); section numbers
below (§N) refer to that document._

## Goal

Build a daily/swing trading system for US equities and ETFs.

- An LLM turns news and filings into structured features.
- A deterministic engine turns features plus prices into target weights.
- A frontier LLM reviews the day's trades, with veto/shrink powers only.
- Hard risk limits and paper execution sit downstream of every LLM.

Prove whether each LLM component adds value using ablations, post-cutoff data and
forward paper trading before any real capital is involved.

## Design rules

These come from the research and apply in every phase.

1. **Point-in-time or nothing.** Every stored row carries `available_at`. Every
   feature lookup is `as_of(t)` (§4.2).
2. **LLMs never touch the broker.** They return schema-validated JSON. Code decides and
   code executes (§5.3, §9).
3. **Pinned models, labelled contamination.** Model IDs are fixed in config. Every
   result is tagged pre-cutoff (an upper bound) or post-cutoff (evidence) (§4.2).
4. **Fail safe.** Any LLM, data or API failure means "no new risk" (§9).
5. **Cache everything the LLM produces,** keyed by `(doc_id, model, prompt_version)`,
   so backtests are reproducible and re-runs are free.
6. **Ablate.** Each LLM layer must beat the configuration without it, after costs
   (§8.2).

## Repository layout

The new package is added beside the existing downloader scripts, which stay untouched.

```
trader/
  __init__.py Alpaca paper account
  config.py            # universe, risk limits, model IDs, cutoffs, paths; reads env vars
  data/
    prices.py          # Alpaca daily bars → parquet
    news.py            # Alpaca/Benzinga news (v1beta1 REST) → parquet
    filings.py         # EDGAR submissions/filings index → parquet
    store.py           # DuckDB over parquet; as_of(ts) queries
  llm/
    schemas.py         # pydantic models for extraction + review outputs
    tagger.py          # Haiku 5.5 Batch extraction, cache, anonymised mode
    finbert.py         # local FinBERT baseline (optional dependency group)
    reviewer.py        # Opus 5.5 daily review (approve / veto / shrink)
  strategy/
    features.py        # price features + aggregated LLM features per symbol-day
    signals.py         # deterministic target weights
  backtest.py          # daily target-weight engine, costs, walk-forward, ablations
  risk.py              # hard limits + kill switches
  execution.py         # alpaca-py paper orders, reconciliation, client_order_id
  run_daily.py         # orchestrates one after-close run (--dry-run)
tests/
data/                  # parquet + caches (git-ignored)
```

**Dependencies.** Add with `uv add`: `anthropic`, `alpaca-py`, `pandas`, `pyarrow`,
`duckdb`, `pydantic`, plus `pytest` (dev). FinBERT's `transformers` and `torch` go in an
optional group (about 2 GB), installed only when needed.

**Environment** (`.env`, already git-ignored; run with `uv run --env-file .env`, as the
README does):

- `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` and
  `APCA_API_BASE_URL=https://paper-api.alpaca.markets` (paper keys only)
- `ANTHROPIC_API_KEY`
- `SEC_USER_AGENT="Name contact@email"`. EDGAR requires a declared User-Agent (§7).

## Phase 0: Setup and decisions

- Create an Alpaca **paper** account and generate paper API keys. Add the Anthropic key
  and the EDGAR User-Agent to `.env`.
- Confirm the v1 universe. Proposed: about 50 heavily covered large caps plus SPY as the
  regime reference, frozen at the start of each evaluation window (§12.1).
- Confirm the price-only core. Proposed: a trend and regime-filtered long-only
  portfolio with volatility scaling (§12.2).

**Accept when:** `.env` has the keys, and the universe list and core strategy are
written into `trader/config.py`.

## Phase 1: Foundations and point-in-time data

- `config.py` holds:
  - the universe;
  - risk limits;
  - `TAGGER_MODEL = "claude-haiku-5-5"` and `REVIEW_MODEL = "claude-opus-5-5"`;
  - `MODEL_TRAINING_CUTOFF = 2026-06-30`, the conservative end of "Jun 2026" per the
    models overview;
  - data paths.
- `data/prices.py` stores daily bars for the universe from 2016 onward (Alpaca history
  starts there), with `available_at` = that session's close.
- `data/news.py` pages through `v1beta1/news` for the universe, keeping headline,
  summary, body, symbols and `created_at`/`updated_at`. Set `available_at` =
  `created_at`.
- `data/filings.py` stores EDGAR filing metadata (form, accession, acceptance datetime)
  for 8-K, 10-Q and 10-K. Set `available_at` = acceptance time. It stays within the
  10 requests/s cap.
- `data/store.py` provides `as_of(table, ts)` over the parquet files via DuckDB.

**Accept when:**

- A property test shows that `as_of(t)` never returns a row with `available_at > t`.
- Incremental re-runs append without creating duplicates.
- A sample day's news count is in line with Alpaca's documented volume.

## Phase 2: LLM extraction tier

- `llm/schemas.py` defines the extraction record.
  - Fields: `symbols`, `event_type` (a fixed enum: earnings, guidance, M&A, legal or
    regulatory, product, management, macro, analyst action, other), `sentiment`
    (−1..1), `surprise` (none/low/med/high), `confidence` (0..1), `horizon_days`, and
    `relevant` (bool).
  - The prompt asks what the text _says_, never what the price _will do_ (§4.2).
- `llm/tagger.py` submits Message Batches with structured output (`output_config.format`)
  and polls until done. It writes results to the cache, keyed by `custom_id` and never
  by position.
  - Thinking is kept minimal: low effort or thinking off, since extraction doesn't
    benefit and thinking tokens bill as output.
  - Token usage is logged per batch.
  - **Anonymised mode** replaces company names, tickers and dates with placeholders
    before sending, for the contamination control arm.
- `llm/finbert.py` scores the same documents locally as the free baseline.
- **Shakedown on this repo's corpus.** Run the same batch-plus-schema pipeline over the
  roughly 70 LLM, sentiment and agent PDFs in `papers/`, reusing the pymupdf
  text-extraction approach from `paperlib.py`. Produce
  `docs/research/corpus-llm-papers.csv` with these columns:
  - method;
  - market;
  - test window;
  - whether the test window is after the evaluated model's cutoff;
  - whether costs are modelled;
  - whether it beats buy-and-hold.
    This exercises the pipeline end to end for pennies, and the table becomes an appendix
    to the research doc.

**Accept when:**

- 100% of outputs validate against the schema; failures are retried, then quarantined.
- Re-running is a pure cache hit at zero cost.
- Cost per 1k documents is logged.
- The corpus CSV exists.

## Phase 3: Strategy, backtester and ablations

- `strategy/features.py` builds:
  - price features (trend, volatility, SPY regime);
  - per-symbol-day aggregates of LLM features as of the prior close, such as decayed
    sentiment, count of high-surprise events, and negative-event flags.
- `strategy/signals.py` computes the price-only core weights. LLM features enter as
  filters or tilts (§3.3), for example: skip or trim names with fresh high-surprise
  negative events, and tilt toward positive ones.
- `backtest.py` is a daily target-weight engine.
  - Signals use data up to close(t) and execute at open(t+1).
  - Costs: commission, half-spread and slippage.
  - Walk-forward windows.
  - Results are labelled pre- or post-cutoff.
  - Output is a metrics table: CAGR, Sharpe, Sortino, max drawdown, turnover, large-move
    hit rate, and LLM cost.
- The ablation runner covers: price-only, +FinBERT, +Haiku, +Haiku-anonymised, and later
  +Opus review. Benchmarks are SPY buy-and-hold and equal-weight universe buy-and-hold.
- Log every configuration tried, for multiple-testing honesty (§8.2).

**Accept when:**

- A planted-future-data test fails loudly.
- The ablation table is generated with pre- and post-cutoff columns.
- Results reproduce exactly from the cache.

## Phase 4: Opus reviewer, risk layer and paper execution

- `llm/reviewer.py`: once per run, Opus 5.5 receives the proposed target weights, the
  feature evidence and the relevant article excerpts.
  - Excerpts are quoted as **untrusted data**.
  - It returns per-symbol `approve | veto | shrink(factor ∈ [0,1])` with a short
    rationale, via structured output.
  - Effort is set explicitly; the model's default is `medium`.
  - It may never add a symbol or increase a weight.
  - On an error, timeout, schema failure or `refusal`, the review is treated as "no
    change".
- `risk.py` enforces hard limits _after_ the review: per-name cap, gross cap, long-only,
  allowlist, price and liquidity floor, maximum order notional, and maximum orders per
  run. Kill switches cover daily loss, drawdown, stale data, reconciliation mismatch
  and repeated API errors.
- `execution.py` uses `alpaca-py` against the paper endpoint.
  - Market-on-open orders.
  - A deterministic `client_order_id` from (date, symbol, side).
  - Pre-run reconciliation of broker positions against expected positions.
  - Post-run fill capture.
- `run_daily.py` runs the full sequence: refresh data → tag new documents → features →
  signals → review → risk → orders. A `--dry-run` flag logs orders without submitting
  them.

**Accept when:**

- The dry-run order list for a given date equals the backtester's for that date.
- Tests prove the reviewer and malformed LLM output cannot breach risk caps.
- A rerun on the same day submits nothing new, which proves idempotency.

## Phase 5: Forward paper trading

- Schedule `run_daily.py` after the US close. Use cron or a systemd timer if the WSL
  machine is reliably on; otherwise move to a small always-on VM.
- Write a weekly report: performance against benchmarks and the price-only shadow
  portfolio, the ablation deltas so far, LLM spend, and any risk events.
- Write down the **go-live gate** before starting (§12.5), for example:
  - at least three months of paper trading;
  - post-cutoff Sharpe above the price-only baseline after costs;
  - drawdown within limits;
  - zero limit breaches;
  - clean reconciliation.

**Accept when:** the system has run unattended for 20+ trading days with complete logs
and weekly reports.

## Later options (not scheduled)

- A TradingAgents multi-agent arm on the same harness (§5.3).
- A local open-weight model arm, for example with FinCAD-style de-biasing (§4.1).
- LLM-assisted strategy search (§5.4), only with strict multiple-testing accounting.
- Intraday: move to NautilusTrader (§8.1).
- Prediction-market experiment (§6).

## Risks to the plan

| Risk                                                | Mitigation                                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Too little post-cutoff data to judge the LLM layers | Forward paper trading is the main test. Report pre-cutoff results only as upper bounds.       |
| Survivorship bias in a large-cap universe           | Freeze the universe at each window start. Note the bias explicitly in the pre-cutoff results. |
| Over-tuning on backtests                            | Walk-forward evaluation, a log of configurations tried, and simple rules.                     |
| Paper fills flatter results                         | The backtester's cost model is the conservative reference (§7).                               |
| A model upgrade silently changes behaviour          | Pinned IDs. A model change is treated as a new evaluation.                                    |
| Prompt injection via news                           | Schema-only output, no broker tools for the LLM, hard limits downstream (§9).                 |
