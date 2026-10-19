"""What the LLMs must return. Every response is validated against these before it is cached (§5.3).

Local servers compile the JSON schema into a sampling grammar, so the schemas stay flat (Literal, not Enum,
which would add $defs), closed (extra="forbid") and fully required, with string lengths capped.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["earnings", "guidance", "m&a", "legal_regulatory", "product", "management", "macro",
                    "analyst_action", "other"]  # fmt: skip


class Extraction(BaseModel):
    """What one article or filing says about the companies it covers. Never a price forecast (§4.2)."""

    model_config = ConfigDict(extra="forbid")

    symbols: list[str]  # tickers the text is materially about
    event_type: EventType
    sentiment: float = Field(ge=-1, le=1)  # for the companies in `symbols`, as the text presents it
    surprise: Literal["none", "low", "med", "high"]
    confidence: float = Field(ge=0, le=1)
    horizon_days: int = Field(ge=1, le=365)  # how long the information stays relevant to fundamentals
    relevant: bool


YesNo = Literal["yes", "no", "not_stated"]


class PaperEvaluation(BaseModel):
    """How one trading paper evaluates its method: the Phase 2 shakedown over papers/ (§12, corpus mining)."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(max_length=300)
    market: str = Field(max_length=100)
    evaluated_models: list[str]
    test_window: str = Field(max_length=100)
    test_after_model_cutoff: YesNo
    costs_modelled: YesNo
    beats_buy_and_hold: Literal["yes", "no", "not_compared"]
