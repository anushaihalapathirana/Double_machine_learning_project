import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.baselines import get_naive_ate, regression_adjustment_ate
from src.config import (
    DATA_PATH,
    FIGURE_DIR,
    METRICS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data import get_features_and_target, load_data
from src.data_validation import validate_ihdp_schema, validate_treatment_split
from src.evaluation import ate_error, evaluate_budget_curve, evaluate_policies, pehe
from src.policy import get_fraction_policy, get_positive_policy, get_threshold_policy
from src.preprocessing import fit_preprocessor

TREATMENT_RATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


@dataclass
class ExperimentConfig:
    data_path: Path = DATA_PATH
    figure_dir: Path = FIGURE_DIR
    metrics_dir: Path = METRICS_DIR
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    validation_size: float = VALIDATION_SIZE
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
    treatment_fraction: float = 0.3
    treatment_rates: list = field(default_factory=lambda: TREATMENT_RATES.copy())
    log_mlflow: bool = True
    make_shap_plots: bool = True


@dataclass
class ExperimentPaths:
    figure_paths: dict
    metric_paths: dict


@dataclass
class DataSplits:
    data: pd.DataFrame
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    T_train: pd.Series
    T_validation: pd.Series
    T_test: pd.Series
    Y_train: pd.Series
    Y_validation: pd.Series
    Y_test: pd.Series
    true_ite_train: pd.Series
    true_ite_validation: pd.Series
    true_ite_test: pd.Series
    X_train_processed: pd.DataFrame
    X_validation_processed: pd.DataFrame
    X_test_processed: pd.DataFrame
    preprocessor: object


@dataclass
class ModelSelectionResult:
    comparison_df: pd.DataFrame
    best_model_name: str
    true_ate_validation: float


@dataclass
class FinalEvaluationResult:
    best_model: object
    preprocessor: object
    X_final_train_processed: pd.DataFrame
    X_test_processed: pd.DataFrame
    comparison_df: pd.DataFrame
    metrics: pd.DataFrame
    positive_policy_results: pd.DataFrame
    fraction_policy_results: pd.DataFrame
    threshold_policy_results: pd.DataFrame
    budget_curve: pd.DataFrame
    policy_values: dict
    baseline_results: dict
    true_ate: float
    dml_ite: object


def build_output_paths(config):
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        figure_paths={
            "ite_distribution": config.figure_dir / "ite_distribution.png",
            "ite_scatter": config.figure_dir / "ite_scatter.png",
            "policy_comparison": config.figure_dir / "policy_comparison.png",
            "model_comparison": config.figure_dir / "model_comparison.png",
            "budget_curve": config.figure_dir / "budget_curve.png",
            "shap_summary": config.figure_dir / "shap_summary.png",
            "shap_importance": config.figure_dir / "shap_importance.png",
        },
        metric_paths={
            "model_metrics": config.metrics_dir / "model_metrics.csv",
            "model_comparison": config.metrics_dir / "model_comparison.csv",
            "positive_policy_results": config.metrics_dir / "positive_policy_results.csv",
            "fraction_policy_results": config.metrics_dir / "fraction_policy_results.csv",
            "threshold_policy_results": config.metrics_dir / "threshold_policy_results.csv",
            "budget_curve": config.metrics_dir / "budget_curve.csv",
        },
    )


def load_and_split_data(config):
    data = load_data(config.data_path, header=0)
    validate_ihdp_schema(data, benchmark_mode=True)
    X, T, Y, true_ite = get_features_and_target(data)

    X_train_val, X_test, T_train_val, T_test, Y_train_val, Y_test, true_ite_train_val, true_ite_test = train_test_split(
        X,
        T,
        Y,
        true_ite,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=T,
    )

    X_train, X_validation, T_train, T_validation, Y_train, Y_validation, true_ite_train, true_ite_validation = train_test_split(
        X_train_val,
        T_train_val,
        Y_train_val,
        true_ite_train_val,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=T_train_val,
    )

    validate_treatment_split(T_train, "Train")
    validate_treatment_split(T_validation, "Validation")
    validate_treatment_split(T_test, "Test")

    preprocessor = fit_preprocessor(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_validation_processed = preprocessor.transform(X_validation)

    return DataSplits(
        data=data,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        T_train=T_train,
        T_validation=T_validation,
        T_test=T_test,
        Y_train=Y_train,
        Y_validation=Y_validation,
        Y_test=Y_test,
        true_ite_train=true_ite_train,
        true_ite_validation=true_ite_validation,
        true_ite_test=true_ite_test,
        X_train_processed=X_train_processed,
        X_validation_processed=X_validation_processed,
        X_test_processed=X_test_processed,
        preprocessor=preprocessor,
    )


def run_model_selection(splits):
    from src.model import get_all_models, train_model

    print("\n" + "=" * 50)
    print("Validation Model Selection")
    print("=" * 50)

    true_ate_validation = splits.true_ite_validation.mean()
    mu0_validation = splits.data.loc[splits.X_validation.index, "mu0"]
    mu1_validation = splits.data.loc[splits.X_validation.index, "mu1"]
    comparison_results = []

    for model_name, model in get_all_models().items():
        print(f"\nTraining {model_name}...")
        trained_model = train_model(model, splits.X_train_processed, splits.T_train, splits.Y_train)

        est_ate = trained_model.ate(splits.X_validation_processed)
        est_ite = trained_model.effect(splits.X_validation_processed)
        threshold_policy_validation = get_threshold_policy(est_ite, threshold=est_ite.mean())

        ate_err = ate_error(est_ate, true_ate_validation)
        pehe_val = pehe(est_ite, splits.true_ite_validation)
        policy_val = (
            threshold_policy_validation * mu1_validation
            + (1 - threshold_policy_validation) * mu0_validation
        ).mean()

        comparison_results.append({
            "Model": model_name,
            "Split": "validation",
            "Estimated ATE": est_ate,
            "ATE Error": ate_err,
            "PEHE": pehe_val,
            "Threshold Policy Value": policy_val,
        })

        print(
            f"  Validation ATE: {est_ate:.4f}, "
            f"ATE Error: {ate_err:.4f}, "
            f"PEHE: {pehe_val:.4f}, "
            f"Threshold Policy Value: {policy_val:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_results).sort_values(
        ["PEHE", "Threshold Policy Value"],
        ascending=[True, False],
    )
    best_model_name = comparison_df.iloc[0]["Model"]

    print("\n--- Validation Model Comparison (sorted by PEHE) ---")
    print(comparison_df.to_string(index=False))
    print(f"\nSelected Model: {best_model_name} (lowest validation PEHE)")

    return ModelSelectionResult(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        true_ate_validation=true_ate_validation,
    )


def build_metrics_table(true_ate, naive_ate, ra_ate, dml_ate, dml_ite, true_ite_test):
    return pd.DataFrame({
        "Metric": [
            "True ATE",
            "Naive ATE",
            "Regression ATE",
            "DML ATE",
            "Naive ATE Error",
            "Regression ATE Error",
            "DML ATE Error",
            "DML PEHE",
        ],
        "Value": [
            true_ate,
            naive_ate,
            ra_ate,
            dml_ate,
            ate_error(naive_ate, true_ate),
            ate_error(ra_ate, true_ate),
            ate_error(dml_ate, true_ate),
            pehe(dml_ite, true_ite_test),
        ],
    })


def run_final_evaluation(splits, selection, config):
    from src.model import get_model, train_model

    X_final_train = pd.concat([splits.X_train, splits.X_validation])
    T_final_train = pd.concat([splits.T_train, splits.T_validation])
    Y_final_train = pd.concat([splits.Y_train, splits.Y_validation])

    preprocessor = fit_preprocessor(X_final_train)
    X_final_train_processed = preprocessor.transform(X_final_train)
    X_test_processed = preprocessor.transform(splits.X_test)

    true_ate = splits.true_ite_test.mean()
    naive_ate = get_naive_ate(Y_final_train, T_final_train)
    ra_ate = regression_adjustment_ate(
        X_final_train_processed,
        X_test_processed,
        T_final_train,
        Y_final_train,
        splits.T_test,
    )

    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)
    print(f"Naive ATE: {naive_ate:.4f}")
    print(f"Regression Adjustment ATE: {ra_ate:.4f}")

    best_model = get_model(selection.best_model_name)
    best_model = train_model(best_model, X_final_train_processed, T_final_train, Y_final_train)
    dml_ate = best_model.ate(X_test_processed)
    dml_ite = best_model.effect(X_test_processed)
    dml_pehe = pehe(dml_ite, splits.true_ite_test)

    print(
        f"Selected DML Test ATE: {dml_ate:.4f}, "
        f"ATE Error: {ate_error(dml_ate, true_ate):.4f}, "
        f"PEHE: {dml_pehe:.4f}"
    )

    positive_policy = get_positive_policy(dml_ite)
    fraction_policy = get_fraction_policy(dml_ite, fraction=config.treatment_fraction)
    threshold_policy = get_threshold_policy(dml_ite, threshold=dml_ite.mean())

    mu0_test = splits.data.loc[splits.X_test.index, "mu0"]
    mu1_test = splits.data.loc[splits.X_test.index, "mu1"]

    positive_policy_results = evaluate_policies(
        positive_policy,
        splits.T_test,
        mu0_test,
        mu1_test,
        random_state=config.random_state,
    )
    fraction_policy_results = evaluate_policies(
        fraction_policy,
        splits.T_test,
        mu0_test,
        mu1_test,
        random_state=config.random_state,
    )
    threshold_policy_results = evaluate_policies(
        threshold_policy,
        splits.T_test,
        mu0_test,
        mu1_test,
        random_state=config.random_state,
    )
    budget_curve = evaluate_budget_curve(
        dml_ite,
        mu0_test,
        mu1_test,
        treatment_rates=config.treatment_rates,
    )

    policy_values = {
        "Observed": (splits.T_test * mu1_test + (1 - splits.T_test) * mu0_test).mean(),
        "Positive policy": (positive_policy * mu1_test + (1 - positive_policy) * mu0_test).mean(),
        f"Top {config.treatment_fraction:.0%} policy": (
            fraction_policy * mu1_test
            + (1 - fraction_policy) * mu0_test
        ).mean(),
        "Threshold policy": (threshold_policy * mu1_test + (1 - threshold_policy) * mu0_test).mean(),
        "Treat All": mu1_test.mean(),
        "Treat None": mu0_test.mean(),
    }
    metrics = build_metrics_table(true_ate, naive_ate, ra_ate, dml_ate, dml_ite, splits.true_ite_test)
    baseline_results = {
        "naive_ate": naive_ate,
        "ra_ate": ra_ate,
        "dml_ate": dml_ate,
        "dml_pehe": dml_pehe,
    }

    return FinalEvaluationResult(
        best_model=best_model,
        preprocessor=preprocessor,
        X_final_train_processed=X_final_train_processed,
        X_test_processed=X_test_processed,
        comparison_df=selection.comparison_df,
        metrics=metrics,
        positive_policy_results=positive_policy_results,
        fraction_policy_results=fraction_policy_results,
        threshold_policy_results=threshold_policy_results,
        budget_curve=budget_curve,
        policy_values=policy_values,
        baseline_results=baseline_results,
        true_ate=true_ate,
        dml_ite=dml_ite,
    )


def save_outputs(paths, splits, selection, results, config):
    from src.plots import (
        plot_budget_curve,
        plot_ite_distribution,
        plot_ite_scatter,
        plot_model_comparison,
        plot_policy_comparison,
        plot_shap_explanations,
    )

    figure_paths = paths.figure_paths
    metric_paths = paths.metric_paths

    plot_ite_distribution(results.dml_ite, figure_paths["ite_distribution"])
    plot_ite_scatter(splits.true_ite_test, results.dml_ite, figure_paths["ite_scatter"])
    plot_policy_comparison(results.policy_values, figure_paths["policy_comparison"])
    plot_budget_curve(results.budget_curve, figure_paths["budget_curve"])
    plot_model_comparison(
        selection.comparison_df,
        figure_paths["model_comparison"],
        true_ate=selection.true_ate_validation,
    )
    if config.make_shap_plots:
        plot_shap_explanations(
            results.best_model,
            results.X_test_processed,
            figure_paths["shap_summary"],
            figure_paths["shap_importance"],
        )

    results.metrics.to_csv(metric_paths["model_metrics"], index=False)
    selection.comparison_df.to_csv(metric_paths["model_comparison"], index=False)
    results.positive_policy_results.to_csv(metric_paths["positive_policy_results"])
    results.fraction_policy_results.to_csv(metric_paths["fraction_policy_results"])
    results.threshold_policy_results.to_csv(metric_paths["threshold_policy_results"])
    results.budget_curve.to_csv(metric_paths["budget_curve"], index=False)


def log_experiment(paths, results, config):
    from src.mlflow_tracking import log_full_experiment, setup_mlflow
    from src.model import get_model_metadata

    print("\n" + "=" * 50)
    print("Logging to MLflow")
    print("=" * 50)

    setup_mlflow(experiment_name=config.experiment_name)

    model_params = {
        "test_size": config.test_size,
        "validation_size": config.validation_size,
        "random_state": config.random_state,
        "treatment_fraction": config.treatment_fraction,
    }
    model_params.update(get_model_metadata(results.comparison_df.iloc[0]["Model"]))

    artifact_paths = list(paths.figure_paths.values()) + list(paths.metric_paths.values())
    if not config.make_shap_plots:
        artifact_paths = [
            path for path in artifact_paths
            if path.name not in {"shap_summary.png", "shap_importance.png"}
        ]

    run_id = log_full_experiment(
        experiment_name=config.experiment_name,
        comparison_df=results.comparison_df,
        policy_results_list=[
            results.positive_policy_results,
            results.fraction_policy_results,
            results.threshold_policy_results,
        ],
        baseline_results=results.baseline_results,
        true_ate=results.true_ate,
        model_params=model_params,
        fitted_best_model=results.best_model,
        model_input_example=results.X_test_processed.head(5),
        artifact_paths=artifact_paths,
    )

    print(f"MLflow Run ID: {run_id}")
    print("Experiment logged successfully!")
    return run_id


def print_best_mlflow_runs(config):
    from src.mlflow_tracking import compare_best_models

    best_runs = compare_best_models(
        experiment_name=config.experiment_name,
        metric="best_model_validation_pehe",
    )
    best_run_columns = [
        "run_id",
        "metrics.best_model_validation_pehe",
        "metrics.best_model_validation_ate_error",
        "metrics.dml_test_pehe",
        "params.best_model",
        "params.model_selection_split",
    ]

    print("\n--- Best MLflow Runs (sorted by validation PEHE) ---")
    print(best_runs[best_run_columns].to_string(index=False))


def run_experiment(config=None):
    config = config or ExperimentConfig()
    paths = build_output_paths(config)
    splits = load_and_split_data(config)
    selection = run_model_selection(splits)
    results = run_final_evaluation(splits, selection, config)
    save_outputs(paths, splits, selection, results, config)

    if config.log_mlflow:
        log_experiment(paths, results, config)
        print_best_mlflow_runs(config)

    return results


def _fraction(value):
    value = float(value)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return value


def _rate(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("Value must be greater than 0 and no more than 1.")
    return value


def _treatment_rates(value):
    rates = [_rate(item.strip()) for item in value.split(",")]
    return rates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the DML causal inference experiment."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--test-size", type=_fraction, default=TEST_SIZE)
    parser.add_argument("--validation-size", type=_fraction, default=VALIDATION_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--treatment-fraction", type=_fraction, default=0.3)
    parser.add_argument(
        "--treatment-rates",
        type=_treatment_rates,
        default=TREATMENT_RATES,
        help="Comma-separated treatment budget rates, for example: 0.05,0.1,0.2,0.3",
    )
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--no-shap", action="store_true")

    return parser.parse_args(argv)


def config_from_args(args):
    return ExperimentConfig(
        data_path=args.data_path,
        figure_dir=args.output_dir / "figures",
        metrics_dir=args.output_dir / "metrics",
        random_state=args.random_state,
        test_size=args.test_size,
        validation_size=args.validation_size,
        experiment_name=args.experiment_name,
        treatment_fraction=args.treatment_fraction,
        treatment_rates=args.treatment_rates,
        log_mlflow=not args.skip_mlflow,
        make_shap_plots=not args.no_shap,
    )
