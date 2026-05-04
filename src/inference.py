import numpy as np

from src.policy import get_fraction_policy, get_positive_policy, get_threshold_policy


def predict_ite(model, preprocessor, new_data):
    """
    Predict individual treatment effects for new feature rows.

    Args:
        model: Fitted causal effect model with an effect(X) method.
        preprocessor: Optional fitted transformer with transform(X). Pass None if
            new_data is already processed in the model's training feature space.
        new_data: Raw or processed feature dataframe.

    Returns:
        Numpy array of estimated individual treatment effects.
    """
    model_input = preprocessor.transform(new_data) if preprocessor is not None else new_data
    return np.asarray(model.effect(model_input))


def assign_policy(ite, policy_type="top_fraction", fraction=0.3, threshold=None):
    """
    Assign treatment decisions from estimated individual treatment effects.

    Args:
        ite: Estimated individual treatment effects.
        policy_type: One of "top_fraction", "positive", or "threshold".
        fraction: Fraction to treat for the top-fraction policy.
        threshold: Treatment threshold for threshold policy. Defaults to mean ITE.

    Returns:
        Binary numpy array where 1 means treat and 0 means do not treat.
    """
    ite = np.asarray(ite)

    if policy_type == "top_fraction":
        return get_fraction_policy(ite, fraction=fraction)
    if policy_type == "positive":
        return get_positive_policy(ite)
    if policy_type == "threshold":
        threshold = ite.mean() if threshold is None else threshold
        return get_threshold_policy(ite, threshold=threshold)

    raise ValueError(
        "Unknown policy_type. Expected one of: top_fraction, positive, threshold."
    )


def score_treatment_policy(model, preprocessor, new_data, policy_type="top_fraction", fraction=0.3, threshold=None):
    """
    Predict treatment effects and assign treatment decisions for new feature rows.

    Returns:
        Tuple of (estimated_ite, treatment_policy).
    """
    ite = predict_ite(model, preprocessor, new_data)
    policy = assign_policy(
        ite,
        policy_type=policy_type,
        fraction=fraction,
        threshold=threshold,
    )
    return ite, policy
