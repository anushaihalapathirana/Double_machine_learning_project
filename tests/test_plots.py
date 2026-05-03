import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.plots import _extract_single_shap_explanation, plot_shap_explanations


class FakeExplanation:
    def __init__(self):
        self.values = np.array([[0.1, 0.2], [0.3, 0.4]])


class FakeModel:
    def __init__(self, explanation):
        self.explanation = explanation

    def shap_values(self, X):
        return {"Y0": {"T0": self.explanation}}


def test_extract_single_shap_explanation_handles_econml_nested_output():
    explanation = FakeExplanation()
    shap_values = {"Y0": {"T0": explanation}}

    result = _extract_single_shap_explanation(shap_values)

    assert result is explanation


def test_plot_shap_explanations_saves_summary_and_importance(monkeypatch, tmp_path):
    calls = []

    def fake_summary_plot(explanation, X, show, max_display, plot_type=None):
        calls.append(plot_type)
        plt.plot([0, 1], [0, 1])

    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(summary_plot=fake_summary_plot)
    )
    X = pd.DataFrame({"x1": [1.0, 2.0], "x2": [0.0, 1.0]})
    summary_path = tmp_path / "shap_summary.png"
    importance_path = tmp_path / "shap_importance.png"

    plot_shap_explanations(
        FakeModel(FakeExplanation()),
        X,
        summary_path,
        importance_path
    )

    assert calls == [None, "bar"]
    assert summary_path.exists()
    assert importance_path.exists()
