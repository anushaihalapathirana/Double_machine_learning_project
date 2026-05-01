import numpy as np

from src.policy import get_fraction_policy, get_positive_policy, get_threshold_policy


def test_positive_policy_treats_only_positive_effects():
    ite = np.array([-1.0, 0.0, 0.2, 3.0])

    policy = get_positive_policy(ite)

    np.testing.assert_array_equal(policy, np.array([0, 0, 1, 1]))


def test_fraction_policy_treats_top_fraction():
    ite = np.array([0.1, 3.0, -1.0, 2.0, 0.5])

    policy = get_fraction_policy(ite, fraction=0.4)

    np.testing.assert_array_equal(policy, np.array([0, 1, 0, 1, 0]))


def test_fraction_policy_with_zero_fraction_treats_nobody():
    ite = np.array([0.1, 3.0, -1.0])

    policy = get_fraction_policy(ite, fraction=0.0)

    np.testing.assert_array_equal(policy, np.array([0, 0, 0]))


def test_threshold_policy_treats_effects_above_threshold():
    ite = np.array([0.1, 0.5, 0.9])

    policy = get_threshold_policy(ite, threshold=0.5)

    np.testing.assert_array_equal(policy, np.array([0, 0, 1]))
