import pytest
from pydantic import ValidationError

from trader.llm import schemas

GOOD = {"symbols": ["AAPL"], "event_type": "earnings", "sentiment": 0.6, "surprise": "med", "confidence": 0.8,
        "horizon_days": 30, "relevant": True}  # fmt: skip


def test_extraction_accepts_a_well_formed_record():
    assert schemas.Extraction.model_validate(GOOD).event_type == "earnings"


@pytest.mark.parametrize(
    "change",
    [{"sentiment": 1.5}, {"confidence": -0.1}, {"event_type": "rumour"}, {"surprise": "huge"}, {"horizon_days": 0},
     {"price_target": 250}],
)  # fmt: skip
def test_extraction_rejects_out_of_range_or_unknown_values(change):
    with pytest.raises(ValidationError):
        schemas.Extraction.model_validate(GOOD | change)


@pytest.mark.parametrize("model", [schemas.Extraction, schemas.PaperEvaluation])
def test_json_schemas_are_strict_and_self_contained(model):
    """Local servers compile the schema into a sampling grammar; keep it flat, closed and fully required."""
    schema = model.model_json_schema()
    assert "$defs" not in schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
