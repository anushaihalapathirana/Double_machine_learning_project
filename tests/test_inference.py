import numpy as np
import pandas as pd
import pytest

from src.inference import assign_policy, predict_ite, score_treatment_policy


class FakeModel:
    def effect(self, X):
        return X["x"].to_numpy() * 2.0


class FakePreprocessor:
    def transform(self, X):
        transformed = X.copy()
        transformed["x"] = transformed["x"] + 1.0
        return transformed


def test_predict_ite_uses_preprocessor_when_provided():
    new_data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    result = predict_ite(FakeModel(), FakePreprocessor(), new_data)

    np.testing.assert_array_equal(result, np.array([4.0, 6.0, 8.0]))


def test_predict_ite_accepts_already_processed_data():
    new_data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    result = predict_ite(FakeModel(), None, new_data)

    np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))


def test_assign_policy_supports_top_fraction_policy():
    ite = np.array([0.1, 3.0, -1.0, 2.0, 0.5])

    policy = assign_policy(ite, policy_type="top_fraction", fraction=0.4)

    np.testing.assert_array_equal(policy, np.array([0, 1, 0, 1, 0]))


def test_assign_policy_supports_positive_policy():
    ite = np.array([-1.0, 0.0, 0.5])

    policy = assign_policy(ite, policy_type="positive")

    np.testing.assert_array_equal(policy, np.array([0, 0, 1]))


def test_assign_policy_supports_threshold_policy_with_default_mean_threshold():
    ite = np.array([0.1, 0.5, 0.9])

    policy = assign_policy(ite, policy_type="threshold")

    np.testing.assert_array_equal(policy, np.array([0, 0, 1]))


def test_assign_policy_rejects_unknown_policy_type():
    with pytest.raises(ValueError, match="Unknown policy_type"):
        assign_policy(np.array([1.0]), policy_type="unknown")


def test_score_treatment_policy_returns_ite_and_policy():
    new_data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    ite, policy = score_treatment_policy(
        FakeModel(),
        FakePreprocessor(),
        new_data,
        policy_type="top_fraction",
        fraction=1 / 3,
    )

    np.testing.assert_array_equal(ite, np.array([4.0, 6.0, 8.0]))
    np.testing.assert_array_equal(policy, np.array([0, 0, 1]))
