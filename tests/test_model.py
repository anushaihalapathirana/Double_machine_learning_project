import pytest

from src.model import get_all_models, get_model


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
