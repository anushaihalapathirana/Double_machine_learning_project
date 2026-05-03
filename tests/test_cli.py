from pathlib import Path

from main import config_from_args, parse_args
from src.config import MLFLOW_EXPERIMENT_NAME, TEST_SIZE, VALIDATION_SIZE


def test_parse_args_uses_project_defaults():
    args = parse_args([])

    assert args.experiment_name == MLFLOW_EXPERIMENT_NAME
    assert args.test_size == TEST_SIZE
    assert args.validation_size == VALIDATION_SIZE
    assert args.skip_mlflow is False
    assert args.no_shap is False


def test_config_from_args_maps_cli_options_to_experiment_config():
    args = parse_args([
        "--output-dir",
        "custom_outputs",
        "--experiment-name",
        "CustomExperiment",
        "--test-size",
        "0.25",
        "--validation-size",
        "0.15",
        "--random-state",
        "7",
        "--treatment-fraction",
        "0.4",
        "--treatment-rates",
        "0.1,0.5,1.0",
        "--skip-mlflow",
        "--no-shap",
    ])

    config = config_from_args(args)

    assert config.figure_dir == Path("custom_outputs") / "figures"
    assert config.metrics_dir == Path("custom_outputs") / "metrics"
    assert config.experiment_name == "CustomExperiment"
    assert config.test_size == 0.25
    assert config.validation_size == 0.15
    assert config.random_state == 7
    assert config.treatment_fraction == 0.4
    assert config.treatment_rates == [0.1, 0.5, 1.0]
    assert config.log_mlflow is False
    assert config.make_shap_plots is False
