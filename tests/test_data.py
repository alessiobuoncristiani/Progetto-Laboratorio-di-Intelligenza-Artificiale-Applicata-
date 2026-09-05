import pandas as pd
import pytest

from src.data import SOURCE_COLUMNS, get_dataset


def test_get_dataset_reads_and_validates_local_csv(tmp_path):
    dataset_path = tmp_path / "diabetes.csv"
    row = [2, 120, 70, 20, 80, 28.5, 0.3, 35, 1]
    pd.DataFrame([row]).to_csv(dataset_path, header=False, index=False)

    data = get_dataset(dataset_path, download=False)

    assert data.columns.tolist() == SOURCE_COLUMNS
    assert data.loc[0, "outcome"] == 1


def test_get_dataset_rejects_non_binary_outcome(tmp_path):
    dataset_path = tmp_path / "invalid.csv"
    row = [2, 120, 70, 20, 80, 28.5, 0.3, 35, 2]
    pd.DataFrame([row]).to_csv(dataset_path, header=False, index=False)

    with pytest.raises(ValueError, match="Outcome must be binary"):
        get_dataset(dataset_path, download=False)


def test_get_dataset_requires_existing_file_when_download_disabled(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        get_dataset(tmp_path / "missing.csv", download=False)


def test_get_dataset_rejects_wrong_number_of_columns(tmp_path):
    dataset_path = tmp_path / "wrong-schema.csv"
    pd.DataFrame([[2, 120, 70]]).to_csv(dataset_path, header=False, index=False)

    with pytest.raises(ValueError, match="unexpected schema"):
        get_dataset(dataset_path, download=False)


def test_get_dataset_rejects_non_numeric_values(tmp_path):
    dataset_path = tmp_path / "non-numeric.csv"
    row = [2, "unknown", 70, 20, 80, 28.5, 0.3, 35, 1]
    pd.DataFrame([row]).to_csv(dataset_path, header=False, index=False)

    with pytest.raises(ValueError, match="numeric values only"):
        get_dataset(dataset_path, download=False)
