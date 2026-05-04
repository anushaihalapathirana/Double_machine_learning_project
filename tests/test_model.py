import pytest

from src.model import get_all_models, get_model, get_model_metadata


def test_get_all_models_includes_available_estimators():
    models = get_all_models()

    assert set(models) == {
        "LinearDML_RF",
        "LinearDML_GB",
        "CausalForestDML_RF",
        "CausalForestDML_GB",
        "DML_Lasso",
    }


def test_get_model_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        get_model("unknown")


def test_get_model_metadata_returns_selected_model_hyperparameters():
    metadata = get_model_metadata("CausalForestDML_RF")

    assert metadata["model_name"] == "CausalForestDML_RF"
    assert metadata["estimator"] == "CausalForestDML"
    assert metadata["model_y_class"] == "RandomForestRegressor"
    assert metadata["model_y_n_estimators"] == 10
    assert metadata["model_y_max_depth"] == 10
    assert metadata["model_t_class"] == "RandomForestClassifier"
    assert metadata["model_t_n_estimators"] == 10
    assert metadata["model_t_max_depth"] == 5
    assert metadata["estimator_discrete_treatment"] is True


def test_get_model_metadata_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        get_model_metadata("unknown")
