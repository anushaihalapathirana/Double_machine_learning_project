# Causal Treatment Policy Learning with Double Machine Learning

This project builds a reproducible causal machine learning pipeline for estimating heterogeneous treatment effects and learning treatment assignment policies. It uses the semi-synthetic IHDP dataset, compares several Double Machine Learning estimators, selects a model on a validation split, evaluates final performance on a held-out test split, and tracks experiments with MLflow.

## Highlights

- Estimates Average Treatment Effect (ATE) and Individual Treatment Effects (ITE/CATE).
- Compares multiple EconML estimators:
  - `LinearDML` with random forests
  - `LinearDML` with gradient boosting
  - `CausalForestDML`
  - `LinearDML` with Lasso/logistic nuisance models
- Uses a train/validation/test workflow:
  - train split for fitting candidate models
  - validation split for model selection
  - test split for final reporting
- Evaluates learned treatment policies against observed, random, treat-all, and treat-none baselines.
- Logs metrics, artifacts, and the selected fitted model to MLflow.
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

## Methodology

### 1. Data Splitting

The pipeline first holds out a test set using `TEST_SIZE` from `src/config.py`. The remaining data is split into train and validation using `VALIDATION_SIZE`.

Candidate models are trained on the training split and compared on the validation split. The final selected model is then evaluated once on the held-out test split.

### 2. Preprocessing

`src/preprocessing.py` standardizes continuous features using statistics learned from the training split and preserves binary features. The same learned scaling is applied to validation and test data.

### 3. Baseline Estimators

The project includes two simple ATE baselines:

- **Naive ATE**: difference in observed mean outcomes between treated and control groups.
- **Regression Adjustment ATE**: linear regression adjustment using treatment and covariates.

These baselines provide context for the DML estimates.

### 4. DML Model Comparison

Candidate DML estimators are defined in `src/model.py`. Each model is trained on the training split and evaluated on validation using:

- validation ATE
- validation ATE error
- validation PEHE
- validation policy value

The selected model is the one with the lowest validation PEHE, with policy value retained as an additional decision-relevant metric.

### 5. Treatment Policies

The project evaluates three treatment policies based on estimated ITE:

- **Positive policy**: treat if estimated ITE is greater than zero.
- **Top fraction policy**: treat the top 30% of individuals by estimated ITE.
- **Threshold policy**: treat if estimated ITE is above the mean estimated ITE.

Policy value is computed using the known potential outcomes:

```text
V(policy) = mean(policy * mu1 + (1 - policy) * mu0)
```

## Metrics

- **ATE Error**: absolute difference between estimated ATE and true ATE.
- **PEHE**: root mean squared error between estimated ITE and true ITE.
- **Policy Value**: expected outcome under a treatment assignment policy.

## Current Result Snapshot

The latest generated outputs report validation model selection in `outputs/metrics/model_comparison.csv` and final test performance in `outputs/metrics/model_metrics.csv`.


## Project Structure

```text
DML/
├── main.py                         # End-to-end experiment pipeline
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

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the full experiment:

```bash
python main.py
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
- `outputs/figures/ite_distribution.png`: estimated ITE distribution.
- `outputs/figures/ite_scatter.png`: true vs estimated ITE.
- `outputs/figures/model_comparison.png`: validation model comparison.
- `outputs/figures/policy_comparison.png`: final test policy value comparison.

MLflow logs:

- validation model-selection metrics
- final test DML PEHE
- baseline ATE estimates and errors
- policy evaluation metrics
- model comparison artifact
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

