import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess_data


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
