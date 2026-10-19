"""Anonymised mode, the contamination control arm (§4.2): company names, tickers and dates become placeholders
before the text reaches the model, so it cannot look up what happened next to that company on that date.

Masking is lexical. Product names (iPhone, Azure) and executives' names still identify some companies, so a
gap between the normal and anonymised arms is a lower bound on how much the model's memory helps.
"""

import re

from trader import config

COMPANY_NAMES = {
    "AAPL": ("Apple",), "MSFT": ("Microsoft",), "NVDA": ("Nvidia", "NVIDIA"), "AMZN": ("Amazon",),
    "GOOGL": ("Alphabet", "Google"), "META": ("Meta Platforms", "Meta", "Facebook"), "TSLA": ("Tesla",),
    "AVGO": ("Broadcom",), "ORCL": ("Oracle",), "CRM": ("Salesforce",), "ADBE": ("Adobe",),
    "AMD": ("Advanced Micro Devices",), "CSCO": ("Cisco",), "INTC": ("Intel",), "QCOM": ("Qualcomm",),
    "TXN": ("Texas Instruments",), "IBM": ("International Business Machines",), "NFLX": ("Netflix",),
    "JPM": ("JPMorgan Chase", "JPMorgan", "JP Morgan", "J.P. Morgan"), "BAC": ("Bank of America", "BofA"),
    "WFC": ("Wells Fargo",), "GS": ("Goldman Sachs", "Goldman"), "MS": ("Morgan Stanley",), "V": ("Visa",),
    "MA": ("Mastercard",), "AXP": ("American Express", "Amex"), "UNH": ("UnitedHealth",),
    "JNJ": ("Johnson & Johnson", "J&J"), "LLY": ("Eli Lilly", "Lilly"), "ABBV": ("AbbVie",), "MRK": ("Merck",),
    "PFE": ("Pfizer",), "TMO": ("Thermo Fisher",), "ABT": ("Abbott",), "WMT": ("Walmart", "Wal-Mart"),
    "PG": ("Procter & Gamble", "P&G"), "HD": ("Home Depot",), "COST": ("Costco",), "PEP": ("PepsiCo", "Pepsi"),
    "KO": ("Coca-Cola", "Coke"), "MCD": ("McDonald's", "McDonalds"), "NKE": ("Nike",),
    "DIS": ("Walt Disney", "Disney"), "XOM": ("Exxon Mobil", "ExxonMobil", "Exxon"), "CVX": ("Chevron",),
    "CAT": ("Caterpillar",), "BA": ("Boeing",), "GE": ("General Electric", "GE Aerospace"),
    "HON": ("Honeywell",), "UNP": ("Union Pacific",),
}  # fmt: skip
EXTRA_TICKERS = {"GOOGL": ("GOOG",)}

_EXCHANGE = r"(?:(?:NYSE|NASDAQ|Nasdaq|NYSEArca|NYSEARCA|AMEX)\s*:\s*|\$)"
_MONTH = (r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
          r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?")  # fmt: skip
_DATES = [  # most specific first, so a full date never leaves a stray year behind
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
    (re.compile(_MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s+\d{4}\b)?"), "[DATE]"),
    (re.compile(r"\b\d{1,2}\s+" + _MONTH + r"(?:,?\s+\d{4}\b)?"), "[DATE]"),
    (re.compile(_MONTH + r"\s+\d{4}\b"), "[DATE]"),
    (re.compile(r"\b(?:19[89]\d|20[0-4]\d)\b"), "[YEAR]"),
]


def _tickers(symbol):
    return (symbol, *EXTRA_TICKERS.get(symbol, ()),
            *(old for old, new in config.SYMBOL_ALIASES.items() if new == symbol))  # fmt: skip


def _pattern(symbol):
    names = sorted(COMPANY_NAMES.get(symbol, ()), key=len, reverse=True)
    parts = [rf"(?<!\w){re.escape(n)}(?![\w&])" for n in names]
    for t in _tickers(symbol):
        parts.append(_EXCHANGE + re.escape(t) + r"\b")
        # a bare one-letter ticker (Visa's V) is only a ticker inside parentheses
        parts.append(rf"\b{re.escape(t)}\b" if len(t) > 1 else rf"(?<=\(){re.escape(t)}(?=\))")
    return re.compile("|".join(parts))


_PATTERNS = {symbol: _pattern(symbol) for symbol in COMPANY_NAMES}


def anonymise(text, tagged):
    """Mask `text`. Returns (masked text, {placeholder: symbol}). The publisher's tagged symbols are numbered
    first, in their given order, then any other universe company the text names, in order of appearance."""
    tagged = list(dict.fromkeys(config.SYMBOL_ALIASES.get(s, s) for s in tagged))
    seen = {s: m.start() for s, p in _PATTERNS.items() if s not in tagged and (m := p.search(text))}
    mapping = {f"Company {i}": s for i, s in enumerate([*tagged, *sorted(seen, key=seen.get)], start=1)}
    for placeholder, symbol in mapping.items():
        text = (_PATTERNS.get(symbol) or _pattern(symbol)).sub(placeholder, text)
    for pattern, mask in _DATES:
        text = pattern.sub(mask, text)
    return text, mapping


def restore(symbols, mapping):
    return [mapping.get(s, s) for s in symbols]
