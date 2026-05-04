# Causal Treatment Policy Learning with Double Machine Learning

This python (v3.10) project builds a reproducible causal machine learning pipeline for estimating heterogeneous treatment effects and learning treatment assignment policies. It uses the semi-synthetic IHDP dataset, compares several Double Machine Learning estimators, selects a model on a validation split, evaluates final performance on a held-out test split, and tracks experiments with MLflow.

## Highlights

- Estimates Average Treatment Effect (ATE) and Individual Treatment Effects (ITE/CATE).
- Compares multiple EconML estimators:
  - `LinearDML` with random forests
  - `LinearDML` with gradient boosting
  - `CausalForestDML_RF` with random forest nuisance models
  - `CausalForestDML_GB` with gradient boosting nuisance models
  - `DML_Lasso` with Lasso/logistic nuisance models
- Uses a train/validation/test workflow:
  - train split for fitting candidate models
  - validation split for model selection
  - test split for final reporting
- Evaluates learned treatment policies against observed, random, treat-all, and treat-none baselines.
- Logs metrics, artifacts, and the selected fitted model to MLflow.
- Provides an inference helper for scoring new rows and assigning treatment policies from estimated ITE.
- Includes unit tests for policy logic, metrics, preprocessing, and baseline estimators.

## Dataset

The project uses the IHDP dataset, a common semi-synthetic benchmark for causal inference. It contains:

- pre-treatment covariates `x1` through `x25`
- binary treatment assignment
- factual and counterfactual outcomes
- true potential outcome functions `mu0` and `mu1`

Because `mu0` and `mu1` are available, the project can compute ground-truth ITE:

```text
ITE = mu1 - mu0
```

That makes metrics such as PEHE possible. In real production causal inference settings, true individual treatment effects are usually not directly observable.

## Benchmark vs Production Evaluation

This repository runs in **benchmark mode** because IHDP is semi-synthetic and includes `mu0` and `mu1`. In benchmark mode, the pipeline can compute true ITE, PEHE, true ATE error, and oracle policy value. These metrics are useful for comparing causal estimators under controlled conditions, but they rely on information that is usually unavailable in real deployments.

In a **production mode** causal ML workflow, model selection would not use true ITE or PEHE. Instead, selection and monitoring would rely on observable-data diagnostics and policy evaluation methods, such as:

- validation policy value estimated with inverse propensity weighting or doubly robust estimators
- nuisance model quality checks for outcome and treatment models
- propensity overlap and common-support diagnostics
- policy stability across folds, time periods, or population segments
- treatment budget, cost, and business constraints

The current implementation therefore treats PEHE-based model selection as a benchmark-only evaluation choice, not as a production selection strategy.


## Methodology

### 1. Data Splitting

The pipeline first holds out a test set using `TEST_SIZE` from `src/config.py`. The remaining data is split into train and validation using `VALIDATION_SIZE`.

Candidate models are trained on the training split and compared on the validation split. After model selection, the selected estimator and baseline models are refit on the combined train + validation data, then evaluated once on the held-out test split.

### 2. Preprocessing

`src/preprocessing.py` fits a reusable feature preprocessor on the training split. Binary `0/1` or `False/True` features are preserved, while all other numeric features are standardized using training-set statistics. The fitted preprocessor is then reused to transform validation, test, and future inference data.

```python
from src.preprocessing import fit_preprocessor

preprocessor = fit_preprocessor(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### 3. Baseline Estimators

The project includes two simple ATE baselines:

- **Naive ATE**: difference in observed mean outcomes between treated and control groups.
- **Regression Adjustment ATE**: linear regression adjustment using treatment and covariates.

These baselines provide context for the DML estimates.

### 4. DML Model Comparison

Candidate DML estimators are defined in `src/model.py`. Each model is trained on the training split and evaluated on validation using:

- validation ATE
- validation ATE error
- validation PEHE, available only in benchmark mode because IHDP has true ITE
- validation policy value, computed with known potential outcomes in benchmark mode

The selected model is the one with the lowest validation PEHE. This is appropriate for the semi-synthetic IHDP benchmark, but a production setting would replace this selection rule with observable policy-value estimates and diagnostics that do not require true treatment effects.

### 5. Treatment Policies

The project evaluates three treatment policies based on estimated ITE:

- **Positive policy**: treat if estimated ITE is greater than zero.
- **Top fraction policy**: treat the top 30% of individuals by estimated ITE.
- **Threshold policy**: treat if estimated ITE is above the mean estimated ITE.

It also evaluates a treatment-budget curve for top-fraction policies at 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100% treatment rates.

Policy value is computed using the known potential outcomes:

```text
V(policy) = mean(policy * mu1 + (1 - policy) * mu0)
```

### 6. Inference and Policy Scoring

`src/inference.py` defines the lightweight scoring interface used after a causal effect model and preprocessor have been trained:

```python
from src.inference import predict_ite, assign_policy, score_treatment_policy

ite = predict_ite(fitted_model, fitted_preprocessor, new_features)
policy = assign_policy(ite, policy_type="top_fraction", fraction=0.3)
ite, policy = score_treatment_policy(
    fitted_model,
    fitted_preprocessor,
    new_features,
    policy_type="top_fraction",
    fraction=0.3,
)
```

Pass `preprocessor=None` when `new_features` is already in the processed feature space expected by the model.

## Metrics

- **ATE Error**: absolute difference between estimated ATE and true ATE.
- **PEHE**: root mean squared error between estimated ITE and true ITE. This is a benchmark-only metric.
- **Policy Value**: expected outcome under a treatment assignment policy. In this benchmark, it is computed with known potential outcomes; in production it would need to be estimated from observed data.

## Current Result Snapshot

The latest generated outputs report validation model selection in `outputs/metrics/model_comparison.csv` and final test performance in `outputs/metrics/model_metrics.csv`.

Current validation selection:

| Selected Model | Validation PEHE | Validation ATE Error | Threshold Policy Value |
| --- | ---: | ---: | ---: |
| CausalForestDML_RF | 0.6288 | 0.0089 | 5.0053 |

Example final test metrics:

| Metric | Value |
| --- | ---: |
| True ATE | 3.9970 |
| Naive ATE | 4.1183 |
| Regression ATE | 3.9558 |
| DML ATE | 3.9237 |
| Naive ATE Error | 0.1213 |
| Regression ATE Error | 0.0412 |
| DML ATE Error | 0.0733 |
| DML PEHE | 0.6572 |


## Project Structure

```text
DML/
├── main.py                         # End-to-end experiment pipeline
├── pyproject.toml                  # Project metadata, dependencies, and test configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── data/
│   └── ihdp_data.csv               # IHDP dataset
├── src/
│   ├── config.py                   # Paths and experiment configuration
│   ├── data.py                     # Data loading and feature extraction
│   ├── preprocessing.py            # Train-based scaling for train/test/validation
│   ├── baselines.py                # Naive and regression-adjustment ATE baselines
│   ├── model.py                    # EconML DML model definitions
│   ├── inference.py                # ITE scoring and treatment policy assignment helpers
│   ├── policy.py                   # Treatment policy rules
│   ├── evaluation.py               # ATE error, PEHE, and policy value metrics
│   ├── plots.py                    # Plot generation
│   └── mlflow_tracking.py          # MLflow tracking and model logging
├── tests/
│   ├── test_baselines.py
│   ├── test_evaluation.py
│   ├── test_policy.py
│   └── test_preprocessing.py
└── outputs/
    ├── figures/                    # Generated plots
    └── metrics/                    # Generated CSV metrics
```

## Installation

Create and activate a virtual environment, then install the project with development dependencies:

```bash
pip install -e ".[dev]"
```

The pinned runtime dependencies are also kept in `requirements.txt` for simple environment setup:

```bash
pip install -r requirements.txt
```

## Usage

Run the full experiment with default settings:

```bash
python main.py
```

Run with custom experiment settings:

```bash
python main.py \
  --experiment-name DML_Causal_Inference_Custom \
  --test-size 0.25 \
  --validation-size 0.2 \
  --treatment-fraction 0.4 \
  --treatment-rates 0.1,0.2,0.4,0.6,0.8,1.0
```

Run locally without MLflow logging or SHAP plot generation:

```bash
python main.py --skip-mlflow --no-shap
```

Write outputs to a different directory:

```bash
python main.py --output-dir outputs/custom_run
```

Run tests:

```bash
pytest
```

View MLflow experiments locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://localhost:5000
```

## Outputs

The pipeline writes:

- `outputs/metrics/model_comparison.csv`: validation model-selection results.
- `outputs/metrics/model_metrics.csv`: final held-out test metrics.
- `outputs/metrics/positive_policy_results.csv`: test policy evaluation for the positive policy.
- `outputs/metrics/fraction_policy_results.csv`: test policy evaluation for the top-fraction policy.
- `outputs/metrics/threshold_policy_results.csv`: test policy evaluation for the threshold policy.
- `outputs/metrics/budget_curve.csv`: policy value at each treatment budget.
- `outputs/figures/ite_distribution.png`: estimated ITE distribution.
- `outputs/figures/ite_scatter.png`: true vs estimated ITE.
- `outputs/figures/model_comparison.png`: validation model comparison.
- `outputs/figures/policy_comparison.png`: final test policy value comparison.
- `outputs/figures/budget_curve.png`: policy value versus treatment budget.
- `outputs/figures/shap_summary.png`: SHAP beeswarm explanation of final model ITE/CATE predictions.
- `outputs/figures/shap_importance.png`: mean absolute SHAP feature importance for final model ITE/CATE predictions.

MLflow logs:

- validation model-selection metrics
- final test DML PEHE
- baseline ATE estimates and errors
- policy evaluation metrics
- generated figure artifacts under `figures/`
- generated metric CSV artifacts under `metrics/`
- SHAP explanation plot artifacts under `figures/`
- fitted selected model as a pyfunc model named `best_model`

## Reproducibility

Core random behavior is controlled by `RANDOM_STATE` in `src/config.py`. The random policy baseline is also seeded.

Local MLflow state is intentionally ignored by Git:

```text
mlflow.db
mlruns/
```

Regenerate local experiment tracking data by running:

```bash
python main.py
```

## Causal Assumptions

This project follows standard causal inference assumptions for interpreting treatment effects:

- **Consistency / SUTVA**: each unit's observed outcome corresponds to its assigned treatment, with no interference between units.
- **Ignorability / unconfoundedness**: treatment assignment is independent of potential outcomes conditional on observed covariates.
- **Overlap / positivity**: each relevant covariate profile has a nonzero probability of receiving treatment and control.

The IHDP benchmark is semi-synthetic, so these assumptions are more controlled than they would be in a real observational dataset.

## Limitations

- IHDP is useful for benchmarking, but it is not a production dataset.
- PEHE relies on ground-truth ITE, which is generally unavailable in real-world applications.
- Hyperparameter tuning is intentionally lightweight.
- Policy value is computed with known potential outcomes; real deployments would require off-policy evaluation or experiment-based validation.
- MLflow model logging uses Python object serialization for the wrapped EconML estimator, which is convenient for local experimentation but should be hardened for production model serving.
