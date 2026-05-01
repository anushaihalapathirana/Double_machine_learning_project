"""
MLflow Tracking Module for DML Causal Inference Project

This module provides utilities for tracking experiments with MLflow,
including model training, metrics logging, and artifact saving.
"""

import mlflow
import mlflow.pyfunc
from datetime import datetime
import os
import tempfile

from src.config import ROOT_DIR, MLFLOW_EXPERIMENT_NAME


class EconMLCausalEffectModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper that predicts treatment effects with an EconML model."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.effect(model_input)


def setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=None):
    """
    Set up MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: Optional MLflow tracking server URI
    
    Returns:
        The experiment object
    """
    # Use the local SQLite backend by default.
    if tracking_uri is None:
        tracking_uri = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.set_experiment(experiment_name)
    
    return experiment


def _log_dataframe_artifact(df, filename):
    """Log a dataframe as a CSV artifact without creating root-level files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, filename)
        df.to_csv(artifact_path, index=False)
        mlflow.log_artifact(artifact_path)


def log_model_comparison(run_name, comparison_df, true_ate):
    """
    Log model comparison results to MLflow.
    
    Args:
        run_name: Name for this run
        comparison_df: DataFrame with model comparison results
        true_ate: True ATE value
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log best model metrics
        best_model = comparison_df.iloc[0]
        mlflow.log_metric("best_model_ate_error", best_model["ATE Error"])
        mlflow.log_metric("best_model_pehe", best_model["PEHE"])
        mlflow.log_metric("best_model_estimated_ate", best_model["Estimated ATE"])
        
        # Log true ATE
        mlflow.log_param("true_ate", true_ate)
        
        # Log all model results as a table
        for _, row in comparison_df.iterrows():
            mlflow.log_metric(f"{row['Model']}_ate_error", row["ATE Error"])
            mlflow.log_metric(f"{row['Model']}_pehe", row["PEHE"])
        
        # Log best model name
        mlflow.log_param("best_model", best_model["Model"])
        
        # Log comparison DataFrame as artifact
        _log_dataframe_artifact(comparison_df, "model_comparison.csv")
        
        return run.info.run_id


def log_baseline_comparison(run_name, naive_ate, ra_ate, dml_ate, true_ate):
    """
    Log baseline comparison results to MLflow.
    
    Args:
        run_name: Name for this run
        naive_ate: Naive ATE estimate
        ra_ate: Regression Adjustment ATE
        dml_ate: DML ATE estimate
        true_ate: True ATE
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log ATE estimates
        mlflow.log_metric("naive_ate", naive_ate)
        mlflow.log_metric("ra_ate", ra_ate)
        mlflow.log_metric("dml_ate", dml_ate)
        mlflow.log_metric("true_ate", true_ate)
        
        # Log ATE errors
        mlflow.log_metric("naive_ate_error", abs(naive_ate - true_ate))
        mlflow.log_metric("ra_ate_error", abs(ra_ate - true_ate))
        mlflow.log_metric("dml_ate_error", abs(dml_ate - true_ate))
        
        return run.info.run_id


def log_full_experiment(
    experiment_name,
    comparison_df,
    policy_results_list,
    baseline_results,
    true_ate,
    model_params=None,
    fitted_best_model=None,
    model_input_example=None
):
    """
    Log a complete experiment with all results.
    
    Args:
        experiment_name: Name for this experiment
        comparison_df: DataFrame with model comparison results
        policy_results_list: List of policy evaluation DataFrames
        baseline_results: Dict with naive_ate, ra_ate, dml_ate
        true_ate: True ATE value
        model_params: Optional dict of model parameters
        fitted_best_model: Optional fitted EconML model to log to MLflow
        model_input_example: Optional example features for the logged model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if model_params:
            for key, value in model_params.items():
                mlflow.log_param(key, value)
        
        mlflow.log_param("true_ate", true_ate)
        
        # Log baseline results
        mlflow.log_metric("naive_ate", baseline_results["naive_ate"])
        mlflow.log_metric("ra_ate", baseline_results["ra_ate"])
        mlflow.log_metric("dml_ate", baseline_results["dml_ate"])
        if "dml_pehe" in baseline_results:
            mlflow.log_metric("dml_test_pehe", baseline_results["dml_pehe"])
        
        # Log baseline errors
        mlflow.log_metric("naive_ate_error", abs(baseline_results["naive_ate"] - true_ate))
        mlflow.log_metric("ra_ate_error", abs(baseline_results["ra_ate"] - true_ate))
        mlflow.log_metric("dml_ate_error", abs(baseline_results["dml_ate"] - true_ate))
        
        # Log best model info
        best_model = comparison_df.iloc[0]
        mlflow.log_param("best_model", best_model["Model"])
        if "Split" in best_model:
            mlflow.log_param("model_selection_split", best_model["Split"])
        mlflow.log_metric("best_model_ate_error", best_model["ATE Error"])
        mlflow.log_metric("best_model_pehe", best_model["PEHE"])
        mlflow.log_metric("best_model_validation_ate_error", best_model["ATE Error"])
        mlflow.log_metric("best_model_validation_pehe", best_model["PEHE"])
        if "Positive Policy Value" in best_model:
            mlflow.log_metric("best_model_validation_positive_policy_value", best_model["Positive Policy Value"])

        if fitted_best_model is not None:
            mlflow.pyfunc.log_model(
                name="best_model",
                python_model=EconMLCausalEffectModel(fitted_best_model),
                input_example=model_input_example
            )
        
        # Log all model comparison
        for _, row in comparison_df.iterrows():
            mlflow.log_metric(f"{row['Model']}_ate_error", row["ATE Error"])
            mlflow.log_metric(f"{row['Model']}_pehe", row["PEHE"])
        
        # Log policy results
        policy_names = ["positive", "fraction", "threshold"]
        for i, policy_df in enumerate(policy_results_list):
            for idx, row in policy_df.iterrows():
                metric_name = f"{policy_names[i]}_policy_{idx.lower().replace(' ', '_')}"
                mlflow.log_metric(metric_name, row["Policy Value"])
        
        # Log comparison CSV
        _log_dataframe_artifact(comparison_df, "model_comparison.csv")
        
        return run.info.run_id


def get_experiment_runs(experiment_name=None):
    """
    Get all runs from an experiment.
    
    Args:
        experiment_name: Name of the experiment (optional)
    
    Returns:
        DataFrame with run information
    """
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            runs = mlflow.search_runs([exp.experiment_id])
            return runs
    return mlflow.search_runs()


def compare_best_models(experiment_name=None, metric="pehe"):
    """
    Compare the best models across all runs.
    
    Args:
        experiment_name: Name of the experiment (optional)
        metric: Metric to sort by (default: pehe)
    
    Returns:
        DataFrame with best runs
    """
    runs = get_experiment_runs(experiment_name)
    if not runs.empty:
        # Filter for best model metrics
        best_runs = runs.sort_values(f"metrics.{metric}", ascending=True)
        return best_runs.head(10)
    return runs
