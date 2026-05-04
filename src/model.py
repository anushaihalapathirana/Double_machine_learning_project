from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression

from src.config import RANDOM_STATE


MODEL_SPECS = {
    "LinearDML_RF": {
        "estimator": "LinearDML",
        "model_y": {
            "class": "RandomForestRegressor",
            "params": {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "model_t": {
            "class": "RandomForestClassifier",
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "estimator_params": {"discrete_treatment": True, "random_state": RANDOM_STATE},
    },
    "LinearDML_GB": {
        "estimator": "LinearDML",
        "model_y": {
            "class": "GradientBoostingRegressor",
            "params": {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "model_t": {
            "class": "GradientBoostingClassifier",
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "estimator_params": {"discrete_treatment": True, "random_state": RANDOM_STATE},
    },
    "CausalForestDML_RF": {
        "estimator": "CausalForestDML",
        "model_y": {
            "class": "RandomForestRegressor",
            "params": {"n_estimators": 10, "max_depth": 10, "random_state": RANDOM_STATE},
        },
        "model_t": {
            "class": "RandomForestClassifier",
            "params": {"n_estimators": 10, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "estimator_params": {"discrete_treatment": True, "random_state": RANDOM_STATE},
    },
    "CausalForestDML_GB": {
        "estimator": "CausalForestDML",
        "model_y": {
            "class": "GradientBoostingRegressor",
            "params": {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "model_t": {
            "class": "GradientBoostingClassifier",
            "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE},
        },
        "estimator_params": {"discrete_treatment": True, "random_state": RANDOM_STATE},
    },
    "DML_Lasso": {
        "estimator": "LinearDML",
        "model_y": {
            "class": "Lasso",
            "params": {"alpha": 0.9, "random_state": RANDOM_STATE},
        },
        "model_t": {
            "class": "LogisticRegression",
            "params": {"max_iter": 1000, "random_state": RANDOM_STATE},
        },
        "estimator_params": {"discrete_treatment": True, "random_state": RANDOM_STATE},
    },
}


MODEL_CLASSES = {
    "LinearDML": LinearDML,
    "CausalForestDML": CausalForestDML,
    "RandomForestRegressor": RandomForestRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "Lasso": Lasso,
    "LogisticRegression": LogisticRegression,
}


def _get_model_spec(model_type):
    try:
        return MODEL_SPECS[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type: {model_type}") from exc


def _build_component(component_spec):
    model_class = MODEL_CLASSES[component_spec["class"]]
    return model_class(**component_spec["params"])


def get_model(model_type="LinearDML_RF"):
    """
    Return a configured DML estimator.

    Args:
        model_type: One of the keys in MODEL_SPECS.

    Returns:
        A DML model instance.
    """
    spec = _get_model_spec(model_type)
    estimator_class = MODEL_CLASSES[spec["estimator"]]

    return estimator_class(
        model_y=_build_component(spec["model_y"]),
        model_t=_build_component(spec["model_t"]),
        **spec["estimator_params"],
    )


def get_all_models():
    """Return all available DML estimators for comparison."""
    return {model_name: get_model(model_name) for model_name in MODEL_SPECS}


def get_model_metadata(model_type):
    """Return flattened metadata for the configured model."""
    spec = _get_model_spec(model_type)
    metadata = {
        "model_name": model_type,
        "estimator": spec["estimator"],
        "model_y_class": spec["model_y"]["class"],
        "model_t_class": spec["model_t"]["class"],
    }

    for key, value in spec["estimator_params"].items():
        metadata[f"estimator_{key}"] = value

    for key, value in spec["model_y"]["params"].items():
        metadata[f"model_y_{key}"] = value

    for key, value in spec["model_t"]["params"].items():
        metadata[f"model_t_{key}"] = value

    return metadata


def train_model(model, X, T, Y):
    model.fit(Y, T, X=X)
    return model
