import numpy as np
import pandas as pd

from src.evaluation import ate_error, evaluate_budget_curve, evaluate_policies, pehe, policy_value


def test_ate_error_returns_absolute_error():
    assert ate_error(3.5, 2.0) == 1.5
    assert ate_error(2.0, 3.5) == 1.5


def test_pehe_returns_rmse_between_effect_vectors():
    estimated_ite = np.array([1.0, 3.0, 5.0])
    true_ite = np.array([1.0, 1.0, 2.0])

    result = pehe(estimated_ite, true_ite)

    assert np.isclose(result, np.sqrt((0.0**2 + 2.0**2 + 3.0**2) / 3))


def test_policy_value_uses_treated_and_control_potential_outcomes():
    policy = np.array([1, 0, 1])
    mu0 = pd.Series([1.0, 2.0, 3.0])
    mu1 = pd.Series([10.0, 20.0, 30.0])

    result = policy_value(policy, mu0, mu1)

    assert np.isclose(result, (10.0 + 2.0 + 30.0) / 3)


def test_evaluate_policies_is_reproducible_with_random_state():
    policy = np.array([1, 0, 1, 0])
    treatment = pd.Series([0, 1, 0, 1])
    mu0 = pd.Series([1.0, 2.0, 3.0, 4.0])
    mu1 = pd.Series([5.0, 6.0, 7.0, 8.0])

    first = evaluate_policies(policy, treatment, mu0, mu1, random_state=42)
    second = evaluate_policies(policy, treatment, mu0, mu1, random_state=42)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == ["Observed", "Random", "DML Policy", "Treat All", "Treat None"]


def test_evaluate_budget_curve_uses_top_effects_for_each_budget():
    ite = np.array([0.1, 3.0, -1.0, 2.0])
    mu0 = pd.Series([1.0, 1.0, 1.0, 1.0])
    mu1 = pd.Series([2.0, 10.0, 0.0, 5.0])

    result = evaluate_budget_curve(ite, mu0, mu1, treatment_rates=[0.25, 0.50])

    expected = pd.DataFrame({
        "Treatment Rate": [0.25, 0.50],
        "Treated Count": [1, 2],
        "Policy Value": [
            (1.0 + 10.0 + 1.0 + 1.0) / 4,
            (1.0 + 10.0 + 1.0 + 5.0) / 4,
        ],
    })
    pd.testing.assert_frame_equal(result, expected)
