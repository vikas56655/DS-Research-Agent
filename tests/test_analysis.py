import io
import pandas as pd
import pytest
from research.analysis import baseline, load_csv, profile, sample_data


def test_sample_is_deterministic():
    assert sample_data().equals(sample_data())


def test_profile_counts():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
    result = profile(df)
    assert result["rows"] == 3
    assert result["missing"] == 1


def test_csv_roundtrip():
    original = sample_data(50)
    loaded = load_csv(original.to_csv(index=False).encode())
    assert loaded.shape == original.shape


def test_csv_rejects_empty():
    with pytest.raises(ValueError):
        load_csv(b"")


def test_classification_baseline():
    result = baseline(sample_data(), "churn", "classification")
    assert 0 <= result["metrics"]["Accuracy"] <= 1
    assert result["test_rows"] > 0


def test_regression_baseline():
    result = baseline(sample_data(), "monthly_spend", "regression")
    assert result["metrics"]["MAE"] >= 0
