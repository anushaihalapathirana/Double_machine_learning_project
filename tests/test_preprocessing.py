import numpy as np
import pandas as pd
import pytest

from src.preprocessing import fit_preprocessor, preprocess_data


def test_preprocess_data_processes_train_test_and_validation_sets():
    X_train = pd.DataFrame(
        {
            "continuous": [1.0, 2.0, 3.0],
            "binary": [0, 1, 0],
        },
        index=[10, 11, 12],
    )
    X_test = pd.DataFrame(
        {
            "continuous": [4.0, 5.0],
            "binary": [1, 0],
        },
        index=[20, 21],
    )
    X_validation = pd.DataFrame(
        {
            "continuous": [6.0],
            "binary": [1],
        },
        index=[30],
    )

    X_train_processed, X_test_processed, X_validation_processed = preprocess_data(
        X_train,
        X_test,
        X_validation,
    )

    assert list(X_train_processed.columns) == ["continuous", "binary"]
    assert list(X_test_processed.columns) == list(X_train_processed.columns)
    assert list(X_validation_processed.columns) == list(X_train_processed.columns)
    assert list(X_train_processed.index) == [10, 11, 12]
    assert list(X_test_processed.index) == [20, 21]
    assert list(X_validation_processed.index) == [30]
    assert np.isclose(X_train_processed["continuous"].mean(), 0.0)
    assert np.isclose(X_train_processed["continuous"].std(ddof=0), 1.0)
    pd.testing.assert_series_equal(
        X_train_processed["binary"],
        X_train["binary"],
        check_names=False,
    )
    assert X_test_processed.loc[20, "binary"] == 1
    assert X_validation_processed.loc[30, "binary"] == 1


def test_preprocess_data_requires_validation_set():
    X_train = pd.DataFrame({"continuous": [1.0, 2.0, 3.0], "binary": [0, 1, 0]})
    X_test = pd.DataFrame({"continuous": [4.0], "binary": [1]})

    with pytest.raises(TypeError):
        preprocess_data(X_train, X_test)


def test_fit_preprocessor_returns_reusable_transformer():
    X_train = pd.DataFrame(
        {
            "binary": [0, 1, 0, 1],
            "integer_continuous": [1, 2, 3, 4],
            "float_continuous": [10.0, 20.0, 30.0, 40.0],
        }
    )
    X_new = pd.DataFrame(
        {
            "binary": [1, 0],
            "integer_continuous": [5, 6],
            "float_continuous": [50.0, 60.0],
        },
        index=[20, 21],
    )

    preprocessor = fit_preprocessor(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_new_processed = preprocessor.transform(X_new)

    assert preprocessor.binary_cols == ["binary"]
    assert preprocessor.continuous_cols == ["integer_continuous", "float_continuous"]
    assert list(X_train_processed.columns) == list(X_train.columns)
    assert list(X_new_processed.columns) == list(X_train.columns)
    assert list(X_new_processed.index) == [20, 21]
    assert np.isclose(X_train_processed["integer_continuous"].mean(), 0.0)
    assert np.isclose(X_train_processed["integer_continuous"].std(ddof=0), 1.0)
    pd.testing.assert_series_equal(
        X_train_processed["binary"],
        X_train["binary"],
        check_names=False,
    )


def test_fit_preprocessor_rejects_non_numeric_features():
    X_train = pd.DataFrame({"numeric": [1.0, 2.0], "category": ["a", "b"]})

    with pytest.raises(ValueError, match="Non-numeric columns: category"):
        fit_preprocessor(X_train)


def test_preprocessor_transform_requires_training_feature_columns():
    X_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0, 1, 0]})
    preprocessor = fit_preprocessor(X_train)

    with pytest.raises(ValueError, match="x2"):
        preprocessor.transform(pd.DataFrame({"x1": [4.0]}))
