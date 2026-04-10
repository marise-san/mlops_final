import pytest
from pydantic import ValidationError
from app.schemas import BankMarketingInput

# A valid payload following all typing and Literal choices of the BankMarketingInput
VALID_PAYLOAD = {
    "age": 30,
    "job": "admin.",
    "marital": "single",
    "education": "university.degree",
    "default": "no",
    "balance": 1500.50,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 250.0,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "nonexistent"
}

def test_valid_payload_accepted():
    model = BankMarketingInput.model_validate(VALID_PAYLOAD)
    assert model.age == 30
    assert model.job == "admin."

def test_reject_out_of_range():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["age"] = 15  # Must be >= 18
    with pytest.raises(ValidationError) as exc:
        BankMarketingInput.model_validate(invalid_payload)
    assert "age" in str(exc.value).lower()

def test_reject_extra_fields():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["extra_field"] = "should fail"
    with pytest.raises(ValidationError) as exc:
        BankMarketingInput.model_validate(invalid_payload)
    assert "extra_field" in str(exc.value)

def test_reject_invalid_categorical():
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["job"] = "astronaut"
    with pytest.raises(ValidationError) as exc:
        BankMarketingInput.model_validate(invalid_payload)
    assert "job" in str(exc.value).lower()
