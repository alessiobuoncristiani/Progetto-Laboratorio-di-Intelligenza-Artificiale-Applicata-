import numpy as np
import pandas as pd

from src.preprocessing import replace_invalid_zeros


def test_replace_invalid_zeros_preserves_valid_zero_and_input_frame():
    original = pd.DataFrame(
        {
            "pregnancies": [0, 3],
            "glucose": [0, 120],
            "blood_pressure": [0, 70],
            "skin_thickness": [0, 20],
            "insulin": [0, 80],
            "bmi": [0, 28.5],
            "diabetes_pedigree": [0.2, 0.3],
            "age": [25, 35],
        }
    )

    cleaned = replace_invalid_zeros(original)

    assert cleaned is not original
    assert cleaned.loc[0, "pregnancies"] == 0
    assert np.isnan(cleaned.loc[0, "glucose"])
    assert np.isnan(cleaned.loc[0, "bmi"])
    assert cleaned.loc[1, "glucose"] == 120
    assert original.loc[0, "glucose"] == 0


def test_replace_invalid_zeros_keeps_non_zero_measurements():
    frame = pd.DataFrame(
        {
            "pregnancies": [1], "glucose": [110], "blood_pressure": [72],
            "skin_thickness": [22], "insulin": [90], "bmi": [29.1],
            "diabetes_pedigree": [0.4], "age": [34],
        }
    )

    cleaned = replace_invalid_zeros(frame)

    assert not cleaned.isna().any().any()
