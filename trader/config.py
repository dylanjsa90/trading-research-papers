"""Single source of truth for the trading system: universe, core strategy, risk limits, models, paths.

Section numbers (§N) refer to docs/research/llm-algo-trading.md. Secrets come from the environment
(run with `uv run --env-file .env`); this module never reads .env itself.
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --- Universe (§12.1) ---------------------------------------------------------------------------
# About 50 heavily covered US large caps, frozen on UNIVERSE_AS_OF. Survivorship caveat (§4.2): these
# are today's large caps, so any backtest before that date on this list is biased upward.
UNIVERSE_AS_OF = date(2026, 10, 19)
UNIVERSE = (
    # technology and communication
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "AMD", "CSCO", "INTC", "QCOM", "TXN", "IBM", "NFLX",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
    # health care
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    # consumer
    "WMT", "PG", "HD", "COST", "PEP", "KO", "MCD", "NKE", "DIS",
    # energy and industrials
    "XOM", "CVX", "CAT", "BA", "GE", "HON", "UNP",
)  # fmt: skip
REGIME_SYMBOL = "SPY"  # regime filter and benchmark; never traded by the core
# Earlier tickers of universe names, so news tagged with the old ticker is not lost.
SYMBOL_ALIASES = {"FB": "META"}  # renamed 2022-06-09


# --- Price-only core (§12.2): trend- and regime-filtered, long-only, volatility-scaled ----------
@dataclass(frozen=True)
class CoreStrategy:
    trend_window: int = 200  # a name is eligible only while it closes above its 200-day SMA
    momentum_window: int = 126  # rank eligible names by ~6-month return...
    momentum_skip: int = 21  # ...skipping the most recent month (short-term reversal)
    max_names: int = 10
    vol_window: int = 60  # inverse-volatility weights from 60-day realised vol
    target_vol: float = 0.15  # scale the book toward 15% annualised vol, never above RISK.max_gross
    regime_window: int = 200  # hold cash while SPY closes below its 200-day SMA


CORE = CoreStrategy()


# --- Hard risk limits (§9), enforced in risk.py after any LLM review ---------------------------
@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.10
    max_gross: float = 1.0
    long_only: bool = True
    min_price: float = 5.0
    min_dollar_volume: float = 50e6  # 20-day average daily dollar volume
    max_order_notional: float = 25_000.0
    max_orders_per_run: int = 25
    daily_loss_halt: float = 0.03
    drawdown_halt: float = 0.15
    max_bar_age_days: int = 4  # stale-data kill switch, in calendar days (covers long weekends)
    max_api_errors: int = 3


RISK = RiskLimits()

# --- Models (§4.2): pinned snapshots; changing one starts a new evaluation ----------------------
TAGGER_MODEL = "claude-haiku-5-5"
REVIEW_MODEL = "claude-opus-5-5"
MODEL_TRAINING_CUTOFF = date(2026, 6, 30)  # conservative end of "Jun 2026"

# --- Data -----------------------------------------------------------------------------------------
HISTORY_START = date(2016, 1, 1)  # Alpaca's historical bars start in 2016
FILING_FORMS = ("8-K", "10-Q", "10-K")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TRADER_DATA_DIR") or ROOT / "data")

# --- Environment ----------------------------------------------------------------------------------
PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def require_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; add it to .env and run with `uv run --env-file .env`")
    return value


def alpaca_credentials():
    """(key_id, secret) for the Alpaca paper account. Refuses anything that looks live."""
    key = require_env("APCA_API_KEY_ID")
    base_url = require_env("APCA_API_BASE_URL").rstrip("/")
    if base_url != PAPER_BASE_URL or not key.startswith("PK"):
        raise RuntimeError(f"refusing non-paper Alpaca credentials: need a PK… key and {PAPER_BASE_URL}")
    return key, require_env("APCA_API_SECRET_KEY")
