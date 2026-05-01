import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FIGURE_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    MLFLOW_EXPERIMENT_NAME
)
from src.data import load_data, get_features_and_target
from src.preprocessing import preprocess_data
from src.baselines import get_naive_ate, regression_adjustment_ate
from src.model import get_model, train_model, get_all_models
from src.policy import get_positive_policy, get_fraction_policy, get_threshold_policy
from src.evaluation import ate_error, pehe, evaluate_policies
from src.mlflow_tracking import setup_mlflow, log_full_experiment, compare_best_models

from src.plots import (
    plot_ite_distribution,
    plot_ite_scatter,
    plot_policy_comparison,
    plot_model_comparison
)

def main():
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data(DATA_PATH, header=0)
    X, T, Y, true_ite = get_features_and_target(data)

    # Hold out the test set first so it is only used for final evaluation.
    X_train_val, X_test, T_train_val, T_test, Y_train_val, Y_test, true_ite_train_val, true_ite_test = train_test_split(
        X,
        T,
        Y,
        true_ite,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=T
    )

    # Split the remaining data into train and validation for model selection.
    X_train, X_validation, T_train, T_validation, Y_train, Y_validation, true_ite_train, true_ite_validation = train_test_split(
        X_train_val,
        T_train_val,
        Y_train_val,
        true_ite_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=T_train_val
    )

    X_train_processed, X_test_processed, X_validation_processed = preprocess_data(
        X_train=X_train,
        X_test=X_test,
        X_validation=X_validation
    )

    true_ate_validation = true_ite_validation.mean()
    mu0_validation = data.loc[X_validation.index, "mu0"]
    mu1_validation = data.loc[X_validation.index, "mu1"]

    # ============================================
    # Validation Model Selection
    # ============================================
    print("\n" + "="*50)
    print("Validation Model Selection")
    print("="*50)
    
    models_dict = get_all_models()
    comparison_results = []
    
    for model_name, model in models_dict.items():
        print(f"\nTraining {model_name}...")
        trained_model = train_model(model, X_train_processed, T_train, Y_train)
        
        est_ate = trained_model.ate(X_validation_processed)
        est_ite = trained_model.effect(X_validation_processed)
        threshold_policy_validation = get_threshold_policy(est_ite, threshold=est_ite.mean() )
        
        ate_err = ate_error(est_ate, true_ate_validation)
        pehe_val = pehe(est_ite, true_ite_validation)
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
            "Threshold Policy Value": policy_val
        })
        
        print(
            f"  Validation ATE: {est_ate:.4f}, "
            f"ATE Error: {ate_err:.4f}, "
            f"PEHE: {pehe_val:.4f}, "
            f"Threshold Policy Value: {policy_val:.4f}"
        )
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values(
        ["PEHE", "Threshold Policy Value"],
        ascending=[True, False]
    )
    print("\n--- Validation Model Comparison (sorted by PEHE) ---")
    print(comparison_df.to_string(index=False))
    
    # Select the best model on validation, then refit it on train + validation.
    best_model_name = comparison_df.iloc[0]["Model"]
    print(f"\nSelected Model: {best_model_name} (lowest validation PEHE)")

    X_final_train = pd.concat([X_train, X_validation])
    T_final_train = pd.concat([T_train, T_validation])
    Y_final_train = pd.concat([Y_train, Y_validation])

    X_final_train_processed, X_test_processed, _ = preprocess_data(
        X_train=X_final_train,
        X_test=X_test,
        X_validation=X_validation
    )

    true_ate = true_ite_test.mean()

    # Baseline ATE estimates on the final held-out test set.
    naive_ate = get_naive_ate(Y_final_train, T_final_train)
    ra_ate = regression_adjustment_ate(
        X_final_train_processed,
        X_test_processed,
        T_final_train,
        Y_final_train,
        T_test
    )

    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    print(f"Naive ATE: {naive_ate:.4f}")
    print(f"Regression Adjustment ATE: {ra_ate:.4f}")

    best_model = get_model(best_model_name)
    best_model = train_model(best_model, X_final_train_processed, T_final_train, Y_final_train)
    dml_ate = best_model.ate(X_test_processed)
    dml_ite = best_model.effect(X_test_processed) 
    print(
        f"Selected DML Test ATE: {dml_ate:.4f}, "
        f"ATE Error: {ate_error(dml_ate, true_ate):.4f}, "
        f"PEHE: {pehe(dml_ite, true_ite_test):.4f}"
    )
    
    # Evaluate policies based on estimated ITE
    positive_policy = get_positive_policy(dml_ite)
    fraction_policy = get_fraction_policy(dml_ite, fraction=0.3)
    threshold_policy = get_threshold_policy(dml_ite, threshold=dml_ite.mean())

    mu0_test = data.loc[X_test.index, "mu0"]
    mu1_test = data.loc[X_test.index, "mu1"]

    evaluate_positive_policy = evaluate_policies(
        positive_policy,
        T_test,
        mu0_test,
        mu1_test,
        random_state=RANDOM_STATE
    )
    evaluate_fraction_policy = evaluate_policies(
        fraction_policy,
        T_test,
        mu0_test,
        mu1_test,
        random_state=RANDOM_STATE
    )
    evaluate_threshold_policy = evaluate_policies(
        threshold_policy,
        T_test,
        mu0_test,
        mu1_test,
        random_state=RANDOM_STATE
    )
    
    metrics = pd.DataFrame({
        "Metric": [
            "True ATE",
            "Naive ATE",
            "Regression ATE",
            "DML ATE",
            "Naive ATE Error",
            "Regression ATE Error",
            "DML ATE Error",
            "DML PEHE"
        ],
        "Value": [
            true_ate,
            naive_ate,
            ra_ate,
            dml_ate,
            ate_error(naive_ate, true_ate),
            ate_error(ra_ate, true_ate),
            ate_error(dml_ate, true_ate),
            pehe(dml_ite, true_ite_test)
        ]
    })


    # plots
    policy_values = {
        "Observed": (T_test * mu1_test + (1 - T_test) * mu0_test).mean(),
        "Positive policy": (positive_policy * mu1_test + (1 - positive_policy) * mu0_test).mean(),
        "Top 30% policy": (fraction_policy * mu1_test + (1 - fraction_policy) * mu0_test).mean(),
        "Threshold policy": (threshold_policy * mu1_test + (1 - threshold_policy) * mu0_test).mean(),
        "Treat All": mu1_test.mean(),
        "Treat None": mu0_test.mean()
    }
    plot_ite_distribution(dml_ite, FIGURE_DIR / "ite_distribution.png")
    plot_ite_scatter(true_ite_test, dml_ite, FIGURE_DIR / "ite_scatter.png")
    plot_policy_comparison(policy_values, FIGURE_DIR / "policy_comparison.png")
    plot_model_comparison(
        comparison_df,
        FIGURE_DIR / "model_comparison.png",
        true_ate=true_ate_validation
    )

    metrics.to_csv(METRICS_DIR / "model_metrics.csv", index=False)
    comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    evaluate_positive_policy.to_csv(METRICS_DIR / "positive_policy_results.csv")
    evaluate_fraction_policy.to_csv(METRICS_DIR / "fraction_policy_results.csv")
    evaluate_threshold_policy.to_csv(METRICS_DIR / "threshold_policy_results.csv")

    # ============================================
    # MLflow Experiment Tracking
    # ============================================
    print("\n" + "="*50)
    print("Logging to MLflow")
    print("="*50)
    
    # Set up MLflow experiment
    experiment = setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME)
    
    # Prepare baseline results
    baseline_results = {
        "naive_ate": naive_ate,
        "ra_ate": ra_ate,
        "dml_ate": dml_ate,
        "dml_pehe": pehe(dml_ite, true_ite_test)
    }
    
    # Model parameters for logging
    model_params = {
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": 100,
        "max_depth": 5
    }
    
    # Log full experiment
    run_id = log_full_experiment(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        comparison_df=comparison_df,
        policy_results_list=[evaluate_positive_policy, evaluate_fraction_policy, evaluate_threshold_policy],
        baseline_results=baseline_results,
        true_ate=true_ate,
        model_params=model_params,
        fitted_best_model=best_model,
        model_input_example=X_test_processed.head(5)
    )
    
    print(f"MLflow Run ID: {run_id}")
    print("Experiment logged successfully!")

    best_runs = compare_best_models(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        metric="best_model_validation_pehe"
    )
    best_run_columns = [
        "run_id",
        "metrics.best_model_validation_pehe",
        "metrics.best_model_validation_ate_error",
        "metrics.dml_test_pehe",
        "params.best_model",
        "params.model_selection_split"
    ]

    print("\n--- Best MLflow Runs (sorted by validation PEHE) ---")
    print(best_runs[best_run_columns].to_string(index=False))
    
if __name__ == "__main__":
    main()
