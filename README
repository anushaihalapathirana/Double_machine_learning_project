# Causal Treatment Policy Learning with Double Machine Learning

This project estimates heterogeneous treatment effects using Double Machine Learning (DML) on the IHDP dataset and evaluates whether learned treatment policies improve outcomes compared with observed, random, treat-all, and treat-none policies.

## Project Structure

```
DML/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README                  # Project overview and instructions
├── data/
│   └── ihdp_data.csv       # IHDP dataset
├── src/
│   ├── config.py           # Configuration and paths
│   ├── data.py             # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing
│   ├── baselines.py        # Baseline ATE estimators
│   ├── model.py            # DML model (LinearDML)
│   ├── policy.py           # Treatment policies
│   ├── evaluation.py       # Evaluation metrics
│   └── plots.py            # Visualization functions
├── notebooks/
│   └── 1_explore_data_and_baseline_model.ipynb
└── outputs/
    ├── figures/            # Generated plots
    └── metrics/            # Evaluation results
```

## Methodology

### 1. Data Preprocessing
- Loads IHDP dataset with features, treatment indicator, and potential outcomes
- Separates continuous and binary features
- Standardizes continuous features; preserves binary features

### 2. Baseline ATE Estimation
- **Naive ATE**: Simple difference in means between treated and control groups
- **Regression Adjustment ATE**: Uses linear regression to adjust for covariates

### 3. Double Machine Learning Model
- Uses EconML's `LinearDML` with:
  - Random Forest Regressor for outcome model (Y)
  - Random Forest Classifier for treatment model (T)
- Estimates both Average Treatment Effect (ATE) and Individual Treatment Effects (ITE)

### 4. Treatment Policies
Three policy types based on estimated ITE:
- **Positive Policy**: Treat if predicted ITE > 0
- **Fraction Policy**: Treat top k% with highest predicted ITE
- **Threshold Policy**: Treat if predicted ITE exceeds threshold

### 5. Evaluation Metrics
- **ATE Error**: |Estimated ATE - True ATE|
- **PEHE**: Precision in Estimation of Heterogeneous Effect (RMSE of ITE)
- **Policy Value**: Expected outcome under each policy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- econml
- sympy

## Output

The program generates:
- **Model Metrics**: ATE estimates and errors, PEHE scores
- **Policy Evaluation**: Expected outcomes for each policy
- **Plots**:
  - ITE distribution histogram
  - True vs Predicted ITE scatter plot
  - Policy comparison bar chart

## Dataset

The IHDP (Infant Health and Development Program) dataset is a semi-synthetic dataset commonly used for causal inference benchmarking. It contains:
- Covariates (pre-treatment variables)
- Binary treatment indicator
- Observed outcomes (Y_factual)
- Counterfactual outcomes (Y_cfactual)
- True potential outcomes (mu0, mu1) for computing ground-truth ITE 