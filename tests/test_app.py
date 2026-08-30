import pytest

from app.app import parse_payload


def valid_payload():
    return {"pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 20, "insulin": 80, "bmi": 28.5, "diabetes_pedigree": 0.3, "age": 35}


def test_parse_payload_converts_numbers():
    parsed = parse_payload(valid_payload())
    assert parsed["glucose"] == 120.0
    assert set(parsed) == set(valid_payload())


@pytest.mark.parametrize("field,value", [("age", -1), ("bmi", -2), ("pregnancies", -1)])
def test_parse_payload_rejects_negative_values(field, value):
    payload = valid_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        parse_payload(payload)


def test_parse_payload_requires_all_fields():
    payload = valid_payload()
    payload.pop("age")
    with pytest.raises(ValueError, match="Campo obbligatorio"):
        parse_payload(payload)
