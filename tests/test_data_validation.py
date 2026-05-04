import pandas as pd
import pytest

from src.data_validation import validate_ihdp_schema, validate_treatment_split


def make_valid_ihdp_data():
    data = {
        "treatment": [0, 1, 0, 1],
        "y_factual": [1.0, 2.0, 3.0, 4.0],
        "y_cfactual": [1.5, 2.5, 3.5, 4.5],
        "mu0": [1.0, 1.0, 1.0, 1.0],
        "mu1": [2.0, 2.0, 2.0, 2.0],
    }
    for index in range(1, 26):
        data[f"x{index}"] = [float(index), float(index + 1), float(index + 2), float(index + 3)]

    return pd.DataFrame(data)


def test_validate_ihdp_schema_accepts_valid_data():
    validate_ihdp_schema(make_valid_ihdp_data())


def test_validate_ihdp_schema_accepts_boolean_treatment():
    data = make_valid_ihdp_data()
    data["treatment"] = [False, True, False, True]

    validate_ihdp_schema(data)


def test_validate_ihdp_schema_requires_expected_columns():
    data = make_valid_ihdp_data().drop(columns=["x25"])

    with pytest.raises(ValueError, match="x25"):
        validate_ihdp_schema(data)


def test_validate_ihdp_schema_requires_benchmark_columns_in_benchmark_mode():
    data = make_valid_ihdp_data().drop(columns=["mu0", "mu1"])

    with pytest.raises(ValueError, match="mu0, mu1"):
        validate_ihdp_schema(data, benchmark_mode=True)


def test_validate_ihdp_schema_allows_missing_benchmark_columns_outside_benchmark_mode():
    data = make_valid_ihdp_data().drop(columns=["mu0", "mu1"])

    validate_ihdp_schema(data, benchmark_mode=False)


def test_validate_ihdp_schema_rejects_missing_values():
    data = make_valid_ihdp_data()
    data.loc[0, "y_factual"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_ihdp_schema(data)


def test_validate_ihdp_schema_rejects_non_numeric_outcomes():
    data = make_valid_ihdp_data()
    data["y_factual"] = ["bad", "data", "is", "here"]

    with pytest.raises(ValueError, match="non-numeric"):
        validate_ihdp_schema(data)


def test_validate_ihdp_schema_rejects_non_binary_treatment_values():
    data = make_valid_ihdp_data()
    data.loc[0, "treatment"] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_ihdp_schema(data)


def test_validate_ihdp_schema_requires_both_treatment_classes():
    data = make_valid_ihdp_data()
    data["treatment"] = 1

    with pytest.raises(ValueError, match="both treated and control"):
        validate_ihdp_schema(data)


def test_validate_treatment_split_requires_both_classes():
    with pytest.raises(ValueError, match="Validation split"):
        validate_treatment_split(pd.Series([1, 1, 1]), "Validation")
