import numpy as np
import pandas as pd

from src.baselines import get_naive_ate, regression_adjustment_ate


def test_get_naive_ate_returns_difference_in_group_means():
    outcomes = pd.Series([1.0, 2.0, 10.0, 14.0])
    treatment = pd.Series([0, 0, 1, 1])

    result = get_naive_ate(outcomes, treatment)

    assert result == 10.5


def test_regression_adjustment_ate_recovers_linear_treatment_effect():
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    T_train = pd.Series([0, 1, 0, 1])
    Y_train = 1.0 + 2.0 * X_train["x"] + 5.0 * T_train
    X_test = pd.DataFrame({"x": [4.0, 5.0]})
    T_test = pd.Series([0, 1])

    result = regression_adjustment_ate(X_train, X_test, T_train, Y_train, T_test)

    assert np.isclose(result, 5.0)
