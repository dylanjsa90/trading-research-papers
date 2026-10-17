# LLM-Based Algorithmic Trading: What It Takes

*Research note, 2026-10-16. Scope: daily/swing trading, a hybrid LLM setup (a cheap
high-volume model plus a frontier decision model), and paper trading first. The phased
implementation plan is in [`docs/plans/llm-trading-build-plan.md`](../plans/llm-trading-build-plan.md).*

**How to read citations.** `[#NNN]` is a paper in this repo's corpus, named
`papers/NNNN_*.pdf` (rank on the Paper Digest list). `papers/` is git-ignored, so those
links only resolve locally; see [Local corpus sources](#local-corpus-sources). Web
sources are linked inline. Claims marked *(secondary)* come from a third-party write-up,
not the primary source, and should be re-checked before anything depends on them.

---

## 1. Executive summary

- **Treat the LLM as a text-to-features engine inside a systematic strategy, not as an
  autonomous trader.** Results that look strong on short windows weaken under long,
  broad backtests ([FINSABER](https://arxiv.org/abs/2505.07078)). In a live,
  uncontaminated benchmark most LLM agents traded badly [#202]. Text signals add most
  value as a risk filter [#8].
- **The defining LLM-specific failure is memorised look-ahead.** A model can "know"
  prices and news in any window before its training cutoff
  ([Gao, Jiang & Yan](https://arxiv.org/abs/2512.23847);
  [Look-Ahead-Bench](https://arxiv.org/abs/2601.13770)). The current Claude models
  have a **June 2026** training data cutoff
  ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)).
  As of this note, only about 3.5 months of history is clean. Forward paper trading is
  therefore the primary evidence, and earlier backtests are upper bounds.
- **Market: US equities and ETFs.** They have the richest timestamped text (Benzinga
  news via Alpaca from 2015, SEC EDGAR filings), free paper trading, the most comparable
  literature, and the liquidity in which LLM agents did best [#202].
- **Proposed architecture.**
  - A point-in-time data store.
  - Claude Haiku 5.5 tags news and filings via the Batch API into a strict schema.
  - A local FinBERT runs as a free control.
  - A deterministic signal engine sets target weights.
  - Claude Opus 5.5 reviews once a day, and may only veto or shrink trades.
  - Hard-coded risk limits sit outside every LLM.
  - Execution goes to an Alpaca paper account with reconciliation.
- **Cost is not the constraint.** Tagging about ten years of news with Haiku 5.5 via
  the Batch API is roughly $50–100. Daily operation is under about $15/month in LLM
  spend. Free Alpaca data is enough for daily bars ([§11](#11-costs)).
- **Every LLM component must earn its place.** Run an ablation ladder: price-only →
  +FinBERT → +Haiku features → +anonymised features → +Opus review. Judge each rung on
  post-cutoff and forward results.

## 2. Scope and constraints

| Decision | Choice | Why it matters |
|---|---|---|
| Horizon | Daily / swing (decide after the close, trade at the next open) | LLM latency (seconds) and cost are irrelevant at this horizon. Most LLM-agent research operates here [#554]. |
| LLM split | Hybrid: Haiku 5.5 for bulk extraction, Opus 5.5 for the daily review | Bulk extraction is a high-volume, low-judgment job. The review is low-volume and high-judgment. |
| Market | US equities and ETFs ([§6](#6-market-choice)) | The data, broker, and literature all fit. |
| Capital | Paper only until a written go-live gate is met | Alpaca paper fills are optimistic ([§7](#7-data-and-broker-stack)). |

Out of scope for v1:
- **HFT.** Even "HFT" LLM frameworks reason over bar windows [#249], and an LLM cannot
  sit in a sub-second loop.
- **Options.** See [§6](#6-market-choice).
- **Fine-tuning or RL of the LLM itself.** See [§5.4](#54-later-stage-options).

## 3. The evidence: what LLMs add, and what they don't

### 3.1 Headline results vs. rigorous evaluation

- **Flagship agent papers evaluate on short windows.**
  - TradingAgents ran its main backtest from 1 Jan to 29 Mar 2024 on Apple, Nvidia,
    Microsoft, Meta and Google [#413].
  - FinMem trained on 17 Aug 2021 – 5 Oct 2022 and tested on 6 Oct 2022 – 10 Apr 2023
    [#748].
  - HedgeAgents reports a 70% annualised return and 400% total over three years [#373].
    Claims of this size need the test-window and cutoff checks in [§4](#4-the-llm-specific-pitfall-memorised-look-ahead).
- **Broader tests erase much of the edge.**
  [FINSABER](https://arxiv.org/abs/2505.07078) (Li, Kim, Cucuringu & Ma; KDD 2026
  Datasets & Benchmarks oral; v6, Jun 2026) ran rolling-window backtests over two
  decades and 100+ symbols.
  - Previously reported LLM advantages "deteriorate significantly".
  - LLM strategies were **too conservative in bull markets**, trailing passive
    benchmarks, and **too aggressive in bear markets**, taking heavy losses.
  - The authors call for trend detection and regime-aware risk control rather than more
    framework complexity.

### 3.2 Live, uncontaminated benchmarking

AI-Trader [#202] is a fully automated live benchmark across US stocks, A-shares and
crypto.
- "General intelligence does not automatically translate to effective trading
  capability, with most agents exhibiting poor returns and weak risk management."
- Risk-control capability determined cross-market robustness.
- Excess returns came "more readily in highly liquid markets".

### 3.3 Text as a signal

- **Best used as a filter.** A systematic review of 80 empirical studies from
  2012–2026 [#8] found that adding news and social sentiment to price data improved
  Sharpe ratios by 20–60% and cut maximum drawdown by up to 49.6% versus price-only
  baselines.
  - The value is strongly state-dependent, appearing mainly in high-volatility regimes
    and crises.
  - Domain transformer models (FinBERT, LLaMA) beat lexicons.
  - The review recommends sentiment "as a complementary risk filter, rather than a
    primary signal amplifier".
- **Accuracy is the wrong metric.** In [#50], tweet sentiment from BERTweet, RoBERTa
  and FinBERT had "near-random directional accuracy". It still improved risk-adjusted
  performance because it caught large moves. Evaluate on P&L and risk, not on
  classification accuracy.
- **Fine-tuned small models work.** FinLlama [#465] (a fine-tuned Llama 2 7B) and
  sentiment-augmented RL [#462] report gains in portfolio simulations. These are
  single-study results on limited windows.

### 3.4 Where text hurt

- **SOTA [#5]** (options agents, six-month out-of-sample run): keeping news during
  reinforcement learning cut out-of-sample return from **18.3% to −2.7%**. News did help
  the frontier-teacher trajectories.
- **CryptoTrade [#472]** beat time-series baselines "but not compared to traditional
  trading signals".

### 3.5 What this implies

1. Start from a sound price-based core, such as trend or regime filters, which
   FINSABER recommends. Add LLM text features as filters or tilts, and measure each
   one's marginal value.
2. Put risk management in deterministic code, since it is what made the difference in
   [#202].
3. Prefer long, broad, walk-forward evaluation over short-window showcase results.

## 4. The LLM-specific pitfall: memorised look-ahead

### 4.1 Mechanism and evidence

An LLM trained on data up to its cutoff may have absorbed the outcomes of events in a
historical backtest. Even with strictly point-in-time inputs, its outputs can encode
the future.

- **[Gao, Jiang & Yan, "Detecting Lookahead Bias in LLM Forecasts"](https://arxiv.org/abs/2512.23847)**
  (v2, Jun 2026) introduce a statistical test built on *Lookahead Propensity* (LAP).
  - LAP is the estimated probability that the model already knows a firm-date outcome.
  - It is materially positive before the cutoff and near zero after.
  - In news-to-return and earnings-call-to-capex tasks, predictive power was larger
    for high-LAP observations. The effect was insignificant after the cutoff.
- **[Look-Ahead-Bench](https://arxiv.org/abs/2601.13770)** (Benhenda, Jan 2026) compares
  stock selection on five large-cap tech names in an in-sample window (Apr–Sep 2021)
  with an out-of-sample window (Jul–Dec 2024).
  - Standard LLMs (Llama 3.1 8B/70B, DeepSeek 3.2) show significant alpha decay.
  - Purpose-built point-in-time models do not.
- **[Glasserman & Lin](https://arxiv.org/abs/2309.17322)** (GPT-3.5, headline sentiment)
  compared named headlines with headlines stripped of company identifiers.
  - In-sample, the anonymised headlines did *better*. The authors read this as the
    model's knowledge of the company distracting it more than look-ahead helped.
  - The advantage lost significance out of sample.
  - They recommend anonymisation for de-biased backtests.
- **[FinCAD](https://arxiv.org/abs/2605.24564)** (Li, Wang & Ma, v2 Aug 2026) suppresses
  memorised outcomes at inference time using context-aware decoding.
  - It cut in-sample backtest returns by up to 67.1% at the model level.
  - Out-of-sample results for the larger models stayed close to baseline.
  - In-sample vs. out-of-sample rank agreement rose from 0.779 to 0.846.
  - It needs token-level control of decoding, which hosted APIs do not expose. So it
    applies only to a local open-weight model.

### 4.2 What this means for this project

- **Contamination window.** Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 all have a
  *Jun 2026* training data cutoff
  ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)).
  Every LLM-feature backtest before July 2026 is potentially contaminated.
- **Pin the model.** Model IDs are pinned snapshots, so the cutoff stays fixed. A future
  model with a later cutoff resets the clean window, so switching models is an
  evaluation event, not a config tweak.
- **Mitigations, in order of strength.**
  1. **Forward paper trading** (genuinely out of sample).
  2. **Post-cutoff historical windows**, short for now and growing daily.
  3. **Ask extraction questions, not prediction questions.** "What event does this
     article describe, and is it positive for the named company?" leaks less than
     "Will this stock go up?". It is not immune, because sentiment judgements can still
     be coloured by hindsight.
  4. **Anonymised control arm.** Replace company names, tickers and dates with
     placeholders. A large in-sample gap between the named and anonymised arms, which
     shrinks after the cutoff, is a contamination signature (Glasserman & Lin;
     Gao et al.).
  5. **FinCAD-style decoding** on a local open model, as an optional research arm.
- **Classical biases still apply.**
  - *Survivorship:* today's large caps backtested from 2015 are winners by
    construction. Freeze the universe at the start of each evaluation window.
  - *Data snooping:* many tried variants inflate the best result. Track the number of
    configurations tried. MadEvolve [#74] explicitly estimates p-hacking probabilities.
  - *Timestamp leakage:* use when a document became available, such as the EDGAR
    acceptance time or the news `created_at`, not the event date.
  - *Costs:* include commissions, spread and slippage.

## 5. Recommended architecture

### 5.1 Shape

```
 ┌─────────────────────── point-in-time store (every row has available_at) ──────────────────────┐
 │  daily bars (Alpaca)      news (Alpaca/Benzinga)      filings (SEC EDGAR)      [macro, later]  │
 └───────┬───────────────────────────┬────────────────────────────┬──────────────────────────────┘
         │                           │                            │
         │              ┌────────────┴─────────────┐              │
         │              │ Tier 1 extraction        │◄─────────────┘
         │              │  Claude Haiku 5.5, Batch │  strict JSON schema, cached by
         │              │  + FinBERT (local ctrl)  │  (doc_id, model, prompt_version)
         │              │  + anonymised variant    │
         │              └────────────┬─────────────┘
         ▼                           ▼
 ┌─────────────────────────────────────────────────┐
 │ Deterministic signal engine → target weights    │  price trend/regime + LLM features
 └───────────────────────┬─────────────────────────┘
                         ▼
 ┌─────────────────────────────────────────────────┐
 │ Tier 2 daily review — Claude Opus 5.5           │  may APPROVE / VETO / SHRINK only
 └───────────────────────┬─────────────────────────┘
                         ▼
 ┌─────────────────────────────────────────────────┐
 │ Hard risk layer (code, not LLM)                 │  caps, kill switch, allowlist, sanity
 └───────────────────────┬─────────────────────────┘
                         ▼
 ┌─────────────────────────────────────────────────┐
 │ Execution — Alpaca paper; reconcile every run   │  idempotent client_order_id
 └─────────────────────────────────────────────────┘
   Audit log of every LLM input/output + token cost  ◄── all layers
```

### 5.2 Roles

| Role | Implementation | Why this way |
|---|---|---|
| Event, sentiment and surprise extraction from news and filings | `claude-haiku-5-5` via the Message Batches API (50% off), structured output via `output_config.format`, low effort | High volume and a narrow task. Batch latency is fine for an after-close run. |
| Control and baseline | FinBERT, local | Free and deterministic. Tells you whether the paid LLM adds anything [#8], [#50]. |
| Signal → target weights | Plain Python | Testable, reproducible, and auditable. |
| Daily review | `claude-opus-5-5`, structured output, effort set explicitly (its default is `medium`) | Low volume, high judgment. Its powers are deliberately limited. |
| Risk limits and execution | Plain Python plus `alpaca-py` | The LLM never holds broker credentials or order tools. |

Model IDs, pricing and API behaviour are from the Claude API reference and the
[models overview](https://platform.claude.com/docs/en/about-claude/models/overview),
checked 2026-10-16.

### 5.3 How this relates to the literature's designs, and why it is more constrained

- **The common pattern.** Published multi-agent systems map trading-desk roles onto
  agents:
  - analysts, bull/bear researchers, a trader and risk managers [#413];
  - layered memory with character profiles [#748];
  - self-reflection and data synthesis [#256];
  - a fund manager with asset-class experts [#373];
  - a full planner/alpha/risk/portfolio/backtest/execution/audit/memory decomposition
    [#201].
  The survey [#554] summarises these designs and their data inputs.
- **Why this design is more constrained.**
  1. The evidence in [§3](#3-the-evidence-what-llms-add-and-what-they-dont) favours
     risk control and trend awareness over more agent complexity.
  2. **Poisoning Agentic Alpha** [#27] attacked four pipeline roles (analyst,
     researcher, trader, risk manager) across four communication topologies, using only
     the source data and prompts the agents consume. It found that "no architecture is
     inherently robust". News text is attacker-reachable input, so the LLM's output must
     pass through a schema and deterministic limits before it can affect an order.
- **Keep a multi-agent arm as an experiment.** TradingAgents
  ([repo](https://github.com/TauricResearch/TradingAgents), Apache-2.0, v0.3.1, about
  111k stars) supports Anthropic models. It can be run as an experimental arm against
  the same data and evaluation harness, rather than as the core.

### 5.4 Later-stage options

- **Train the LLM on trading outcomes.**
  - Trading-R1 [#248] uses SFT plus RL with an easy-to-hard curriculum on a 100k-sample
    corpus.
  - FLAG-Trader [#372] uses a partially fine-tuned LLM as the RL policy.
  - Janus-Q [#135] fine-tunes on 62,400 event-annotated articles with
    cumulative-abnormal-return labels.
  - SOTA [#5] post-trains an open model for option-strategy selection.
  All of these need GPU training budgets and careful leakage control.
- **LLMs as strategy researchers or coders.**
  - QuantCode-Bench [#96]: the best frontier models generate correct Backtrader
    strategies about 70–76% of the time single-turn, and 95–98% with agentic iteration.
  - MadEvolve [#74] applies AlphaEvolve-style evolutionary search to trading strategies
    and compares against Claude Code.
  This is a good fit once the backtester exists, but it amplifies data-snooping risk.
- **Multimodal inputs** such as price charts [#535]. The survey notes visual data is
  underexplored [#554].

## 6. Market choice

| Market | Text and data for LLMs | Paper trading and API | Literature comparability | Risks | Verdict |
|---|---|---|---|---|---|
| **US equities and ETFs** | Benzinga news from 2015 via Alpaca; SEC filings with acceptance times | Free Alpaca paper account ($100k) | FINSABER, TradingAgents, FinMem, Trading-R1 | Survivorship in the universe; costs | **Chosen** |
| Crypto | Less fundamental text; on-chain data | Many exchange APIs, 24/7 | CryptoTrade [#472] | Manipulation and wash trading [#531], [#61]; 24/7 operations | Later, maybe |
| Options | Text plus implied distributions | Alpaca options exist | SOTA [#5] | Harder risk management; news hurt out of sample [#5] | Not v1 |
| Prediction markets | Natural fit for LLM forecasting | Kalshi data | PredictionMarketBench [#153] | Fees and settlement; separate infrastructure | Good later experiment |

US equities also match AI-Trader's finding that agents earn excess returns more
readily in highly liquid markets [#202].

## 7. Data and broker stack

| Need | Source | Cost | Notes |
|---|---|---|---|
| Daily bars | [Alpaca Market Data](https://docs.alpaca.markets/docs/about-market-data-api) | Free Basic plan | See the Basic plan limits below. Daily bars work on the free plan. |
| News | [Alpaca historical news](https://docs.alpaca.markets/us/docs/historical-news-data) | Included with an account | See the news notes below. |
| Filings | [SEC EDGAR](https://www.sec.gov/os/accessing-edgar-data) (`data.sec.gov`) | Free | Maximum 10 requests/s. Declare a User-Agent with a contact email; unmanaged bots are throttled. |
| Alternative market data | Massive (formerly Polygon.io) | Free tier about 5 calls/min, end-of-day only *(secondary)* | Rebranded Oct 2025 *(secondary: [apicostcalc](https://apicostcalc.com/blog/polygon-massive-rebrand-api-pricing.html))*. Check massive.com/pricing. |
| Broker (paper, then live) | Alpaca: [`alpaca-py`](https://github.com/alpacahq/alpaca-py) (Apache-2.0) | Free paper account | Official [MCP server](https://github.com/alpacahq/alpaca-mcp-server) exists. Use it read-only for exploration at most; never give the LLM order tools ([§9](#9-risk-management-and-security)). |
| Alternative broker | Interactive Brokers via [`ib_async`](https://github.com/ib-api-reloaded/ib_async) (BSD-2, replaces ib_insync) | IBKR account | Broader markets, more operational complexity. |

**Alpaca Basic plan limits.**
- Real-time equities come from IEX only.
- Historical data goes back to 2016, minus the latest 15 minutes.
- 200 historical calls/min and 30 websocket symbols.
- Algo Trader Plus is $99/month for all US exchanges and 10,000 calls/min.

**Alpaca news notes.**
- Benzinga is the sole provider, averaging 130+ articles a day.
- One endpoint, `data.alpaca.markets/v1beta1/news`, serves stocks and crypto.
- `alpaca-py`'s built-in data clients target v2, so news needs a generic REST call
  *(secondary: [Alpaca forum](https://forum.alpaca.markets/t/historical-news-on-python-3-11/11727))*.

**Paper-trading realism**
([Alpaca paper trading docs](https://docs.alpaca.markets/docs/paper-trading)).
- Paper fills simulate against the real-time NBBO. They fill only when marketable, and
  get random partial fills 10% of the time.
- Paper does **not** simulate market impact, latency slippage, queue position, price
  improvement, regulatory fees or dividends. Borrow fees are "coming soon".
- Order size is not checked against displayed liquidity.
- Paper P&L is therefore optimistic. The backtester's cost model, not paper fills, is
  the conservative estimate.

## 8. Backtesting and evaluation

### 8.1 Framework landscape

Repository activity checked via the GitHub API on 2026-10-16.

| Framework | Status | Licence | Fit |
|---|---|---|---|
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | Active (pushed 2026-10-16), about 30k stars | LGPL-3.0 | Event-driven, Rust core, the same code for backtest and live. The best upgrade path for intraday or production. |
| [LEAN](https://github.com/QuantConnect/Lean) | Active, about 22k stars | Apache-2.0 | Multi-asset, C# core with Python. Heavier setup. |
| [vectorbt](https://github.com/polakowo/vectorbt) | Active, about 9.4k stars | Apache-2.0 + Commons Clause | Fast vectorised parameter sweeps. |
| [zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Active, about 2k stars | Apache-2.0 | Equity factor research with the Pipeline API. Needs data bundles. |
| [backtesting.py](https://github.com/kernc/backtesting.py) | Active | **AGPL-3.0** | Single-asset focus. Watch the licence. |
| [backtrader](https://github.com/mementum/backtrader) | **Stale**: last push Aug 2024 | GPL-3.0 | Avoid for new work, even though QuantCode-Bench [#96] targets it. |

**v1 recommendation: a small in-house daily target-weight backtester.** For daily
rebalancing, it is about 150 lines of pandas. It makes point-in-time joins against the
LLM-feature cache explicit, and it is easy to unit-test for look-ahead. Move to
NautilusTrader if the system goes intraday.

### 8.2 Evaluation protocol

1. **Timing.** Features use only rows with `available_at` ≤ close(t). Trades fill at
   open(t+1). A test plants future data and must fail.
2. **Costs.** Commission (Alpaca is commission-free for US equities, but model it
   anyway), half-spread plus a slippage assumption, and turnover penalties.
3. **Walk-forward** with rolling windows, as in FINSABER. No in-sample tuning on test
   windows.
4. **Contamination labels.** Every result is tagged *pre-cutoff* (upper bound) or
   *post-cutoff* (evidence) relative to the pinned model's training cutoff.
5. **Ablation ladder.** Price-only core → +FinBERT → +Haiku features → +anonymised Haiku
   features → +Opus review.
6. **Benchmarks.** SPY buy-and-hold, equal-weight universe buy-and-hold, and a plain
   trend-following baseline.
7. **Metrics.** CAGR, Sharpe, Sortino, max drawdown, turnover, hit rate on large moves
   (per [#50]), and LLM cost per trade.
8. **Multiple-testing discipline.** Log every configuration tried, and report the count
   alongside the best result.
9. **Forward test.** At least three months of paper trading with the pinned models
   before any go-live decision.

## 9. Risk management and security

- **Hard limits in code.** Enforced after the LLM review, so the LLM cannot override
  them.
  - Maximum weight per position.
  - Maximum gross exposure.
  - Long-only for v1.
  - A symbol allowlist.
  - A minimum price and liquidity filter.
  - A maximum order notional.
  - A maximum number of orders per run.
- **Kill switches.** Halt new orders on any of:
  - daily loss beyond X%;
  - drawdown from peak beyond Y%;
  - reconciliation mismatch;
  - stale data (last bar older than expected);
  - repeated API errors.
- **Fail-safe defaults.** On an LLM error, timeout, schema failure or
  `stop_reason == "refusal"`, the review step is treated as "no change". The system
  never opens new risk because a component failed.
- **Idempotency and reconciliation.** Derive a deterministic `client_order_id` from
  (date, symbol, side) so reruns cannot double-submit. Compare broker positions with
  expected positions at the start of every run.
- **Prompt-injection hygiene** [#27].
  - Pass news and filings as quoted data with explicit "untrusted content" framing.
  - Accept LLM output only through a strict schema.
  - The LLM gets no tools that touch the broker.
  - Log every LLM input and output for audit.
- **Secrets.** Keys live in `.env`, which is already git-ignored. Use separate paper and
  live keys; Alpaca issues distinct keys per account.

## 10. Regulation and tax notes

*Not legal or tax advice.*

- **The PDT rule is gone.** The pattern day trader rule and its $25,000 minimum have
  been replaced by intraday margin standards.
  - Sources: FINRA [Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)
    (20 Apr 2026) and SEC Release
    [34-105226](https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf), the
    accelerated approval order dated 14 Apr 2026.
  - Firms must instead monitor "intraday margin deficits" in margin accounts.
  - Effective 4 Jun 2026, with a phase-in for firms to 20 Oct 2027. Broker
    implementation timing varies; for example, Schwab stopped counting day trades on
    8 Jun 2026 *(secondary:
    [Schwab](https://international.schwab.com/story/sec-approves-scrapping-25000-day-trader-minimum))*.
  - A daily/swing system rarely day-trades anyway.
- **Cash vs. margin account.** A cash account avoids margin entirely, but unsettled-funds
  rules apply. Check the broker's policy.
- **Wash-sale rule.** Frequent automated selling at a loss and re-buying can trigger
  wash-sale disallowance. Not researched in this note, so check IRS guidance or a tax
  adviser before live trading.
- **Trading other people's money** is out of scope. Get professional advice first.
- **Regulatory context for AI in trading.** The EU AI Act compared with sectoral
  algorithmic-trading rules [#692]. Relevant mainly to firms operating in the EU.

## 11. Costs

Prices are from the
[models overview](https://platform.claude.com/docs/en/about-claude/models/overview)
(checked 2026-10-16).
- Opus 5.5 costs $4 / $20 per million input/output tokens.
- Haiku 5.5 costs from $0.10 / $0.50 per million tokens (prompts up to 100K tokens).
- The Batch API is 50% off.
- Cache reads are billed at a fraction of the input price: 10% on most models, 5% on
  Opus 5.5.

| Item | Estimate | Working |
|---|---|---|
| Historical news backfill (about 2015 → today) | **≈ $50–100** | About 130 articles/day × 365 × about 11 years ≈ 520k articles. At about 1.5k input tokens each, that is ≈ 780M tokens × $0.05/MTok (batch) ≈ $39. Output of about 150–300 tokens each is ≈ 80–160M tokens × $0.25/MTok ≈ $20–40. Thinking tokens count as output, so keep effort low or thinking off for extraction. |
| Daily extraction (live) | ≈ $0.03/day | About 200 docs × 1.5k tokens via batch, plus small outputs. |
| Daily Opus review | ≈ $0.25–0.45/day | About 30–60k input tokens and 5–10k output tokens including thinking. |
| **Monthly LLM total (live)** | **< about $15** | About 21 trading days. |
| Market data | $0 (Basic) or $99/month (Algo Trader Plus) | Basic is enough for daily bars. |
| FinBERT | $0 | Runs on CPU. A GPU is optional. |

Reprocessing history is cheap, so changing the extraction prompt (a new
`prompt_version`) is affordable. Cache by version so results stay reproducible.

## 12. Open questions and next decisions

1. **Universe.** Options are mega/large caps with heavy news coverage (survivorship
   caveat) or sector ETFs, where news is harder to map. Proposed: about 50 large caps
   plus SPY for the regime filter, frozen at the start of each evaluation window.
2. **Strategy core.** Which price-only baseline should the LLM features modify?
   Proposed: a trend and regime-filtered long-only portfolio with volatility scaling,
   following FINSABER's recommendation.
3. **Local open model arm.** Worth adding only for privacy or FinCAD-style de-biasing
   research. On cost alone, Haiku 5.5 via batch is already cheap.
4. **Model upgrades.** Pin model IDs. Treat a model change as a new evaluation with a
   new contamination window.
5. **Go-live gate.** Define it before forward testing starts, for example:
   - at least three months of paper trading;
   - post-cutoff Sharpe above the price-only baseline after costs;
   - drawdown within limits;
   - zero risk-limit breaches;
   - a stable reconciliation record.
6. **Corpus mining.** Extract each LLM paper's evaluation details (window, post-cutoff
   status, costs modelled, beats buy-and-hold) into a table. This is Phase 2's
   shakedown task in the build plan.

## Sources

### Local corpus sources

All are under `papers/` and git-ignored. The date and venue are as listed by Paper
Digest.

| Rank | File | Paper |
|---|---|---|
| #5 | `0005_SOTA_Stock_Options_Trading_Agents_Guided_By_Option_Implied_Return_Distributions.pdf` | SOTA: Stock Options Trading Agents (arXiv 2610.10407, 2026) |
| #8 | `0008_The_Impact_of_Sentiment_Integration_on_Algorithmic_Trading_Performance_A_Systema.pdf` | Sentiment integration: systematic review of 80 studies (*Analytics*, 2026) |
| #27 | `0027_Poisoning_Agentic_Alpha_Adversarial_Vulnerabilities_Across_Roles_and_Architectur.pdf` | Poisoning Agentic Alpha (arXiv 2608.24069, 2026) |
| #50 | `0050_Financial_Sentiment_Analysis_Meets_Algorithmic_Trading_A_Performance_based_Appro.pdf` | Financial Sentiment Analysis Meets Algorithmic Trading (*Cogent Econ. & Finance*, 2026) |
| #61 | `0061_Resisting_Manipulative_Bots_in_Meme_Coin_Copy_Trading_A_Multi_Agent_Approach_wit.pdf` | Resisting Manipulative Bots in Meme Coin Copy Trading |
| #74 | `0074_MadEvolve_Evolutionary_Optimization_of_Trading_Systems_with_Large_Language_Model.pdf` | MadEvolve (arXiv 2605.23007, 2026) |
| #96 | `0096_QuantCode_Bench_A_Benchmark_for_Evaluating_The_Ability_of_Large_Language_Models.pdf` | QuantCode-Bench (arXiv 2604.15151, 2026) |
| #135 | `0135_Janus_Q_End_to_End_Event_Driven_Trading_Via_Hierarchical_Gated_Reward_Modeling.pdf` | Janus-Q (arXiv 2602.19919, 2026) |
| #153 | `0153_PredictionMarketBench_A_SWE_bench_Style_Framework_for_Backtesting_Trading_Agents.pdf` | PredictionMarketBench (arXiv 2602.00133, 2026) |
| #201 | `0201_Orchestration_Framework_for_Financial_Agents_From_Algorithmic_Trading_to_Agentic.pdf` | Orchestration Framework for Financial Agents (arXiv 2512.02227, 2025) |
| #202 | `0202_AI_Trader_Benchmarking_Autonomous_Agents_in_Real_Time_Financial_Markets.pdf` | AI-Trader (arXiv 2512.10971, 2025) |
| #248 | `0248_Trading_R1_Financial_Trading_with_LLM_Reasoning_Via_Reinforcement_Learning.pdf` | Trading-R1 (arXiv 2509.11420, 2025) |
| #249 | `0249_QuantAgent_Price_Driven_Multi_Agent_LLMs_for_High_Frequency_Trading.pdf` | QuantAgent / QuantHarness (arXiv 2509.09995, 2025) |
| #256 | `0256_TradingGroup_A_Multi_Agent_Trading_System_with_Self_Reflection_and_Data_Synthesi.pdf` | TradingGroup (arXiv 2508.17565, 2025) |
| #372 | `0372_FLAG_Trader_Fusion_LLM_Agent_with_Gradient_based_Reinforcement_Learning_for_Fina.pdf` | FLAG-Trader (arXiv 2502.11433, 2025) |
| #373 | `0373_HedgeAgents_A_Balanced_aware_Multi_agent_Financial_Trading_System.pdf` | HedgeAgents (arXiv 2502.13165, 2025) |
| #413 | `0413_TradingAgents_Multi_Agents_LLM_Financial_Trading_Framework.pdf` | TradingAgents (arXiv 2412.20138, 2024) |
| #462 | `0462_Financial_News_Driven_LLM_Reinforcement_Learning_for_Portfolio_Management.pdf` | Financial News-Driven LLM RL for Portfolio Management (arXiv 2411.11059, 2024) |
| #465 | `0465_FinLlama_LLM_Based_Financial_Sentiment_Analysis_for_Algorithmic_Trading.pdf` | FinLlama (ICAIF 2024) |
| #472 | `0472_CryptoTrade_A_Reflective_LLM_based_Agent_to_Guide_Zero_shot_Cryptocurrency_Tradi.pdf` | CryptoTrade (EMNLP 2024) |
| #531 | `0531_Exposing_Stealthy_Wash_Trading_on_Automated_Market_Maker_Exchanges.pdf` | Exposing Stealthy Wash Trading on AMM Exchanges |
| #535 | `0535_A_Multimodal_Foundation_Agent_for_Financial_Trading_Tool_Augmented_Diversified_a.pdf` | FinAgent (KDD 2024) |
| #554 | `0554_Large_Language_Model_Agent_in_Financial_Trading_A_Survey.pdf` | LLM Agent in Financial Trading: A Survey (arXiv 2408.06361, 2024) |
| #692 | `0692_AI_Governance_in_Algorithmic_Trading_Some_Regulatory_Insights_from_The_EU_AI_Act.pdf` | AI Governance in Algorithmic Trading: EU AI Act (SSRN) |
| #748 | `0748_FinMem_A_Performance_Enhanced_LLM_Trading_Agent_with_Layered_Memory_and_Characte.pdf` | FinMem (arXiv 2311.13743, 2023) |

### Web sources

Retrieved 2026-10-16.

**Primary**
- Li, Kim, Cucuringu & Ma, *Can LLM-based Financial Investing Strategies Outperform the
  Market in Long Run?* (FINSABER): <https://arxiv.org/abs/2505.07078>
- Gao, Jiang & Yan, *Detecting Lookahead Bias in LLM Forecasts*: <https://arxiv.org/abs/2512.23847>
- Benhenda, *Look-Ahead-Bench*: <https://arxiv.org/abs/2601.13770>
- Glasserman & Lin, *Assessing Look-Ahead Bias in Stock Return Predictions Generated By
  GPT Sentiment Analysis*: <https://arxiv.org/abs/2309.17322>
- Li, Wang & Ma, *Summoning the Oracle to Slay It* (FinCAD): <https://arxiv.org/abs/2605.24564>
- Anthropic models overview (IDs, prices, training cutoffs): <https://platform.claude.com/docs/en/about-claude/models/overview>
- Alpaca, About Market Data API: <https://docs.alpaca.markets/docs/about-market-data-api>
- Alpaca, Paper Trading: <https://docs.alpaca.markets/docs/paper-trading>
- Alpaca, Historical News Data: <https://docs.alpaca.markets/us/docs/historical-news-data>
- SEC, Accessing EDGAR Data: <https://www.sec.gov/os/accessing-edgar-data>
- FINRA Regulatory Notice 26-10: <https://www.finra.org/rules-guidance/notices/26-10>
- SEC Release 34-105226 (SR-FINRA-2025-017): <https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf>
- GitHub repositories (activity via the GitHub REST API): TauricResearch/TradingAgents,
  virattt/ai-hedge-fund, AI4Finance-Foundation/FinRobot, nautechsystems/nautilus_trader,
  QuantConnect/Lean, polakowo/vectorbt, stefan-jansen/zipline-reloaded,
  kernc/backtesting.py, mementum/backtrader, alpacahq/alpaca-py,
  alpacahq/alpaca-mcp-server, ib-api-reloaded/ib_async

**Secondary** (re-check before relying on them)
- Schwab, PDT change and implementation date: <https://international.schwab.com/story/sec-approves-scrapping-25000-day-trader-minimum>
- Polygon.io → Massive rebrand and pricing: <https://apicostcalc.com/blog/polygon-massive-rebrand-api-pricing.html>
- Alpaca community forum, news endpoint vs. `alpaca-py` v2 clients: <https://forum.alpaca.markets/t/historical-news-on-python-3-11/11727>
- TradingAgents release history: <https://github.com/TauricResearch/TradingAgents/releases>
