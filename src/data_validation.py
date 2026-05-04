import pandas as pd
from pandas.api.types import is_numeric_dtype

FEATURE_COLUMNS = [f"x{i}" for i in range(1, 26)]
CORE_COLUMNS = ["treatment", "y_factual", "y_cfactual"]
BENCHMARK_COLUMNS = ["mu0", "mu1"]
NUMERIC_COLUMNS = ["y_factual", "y_cfactual"] + BENCHMARK_COLUMNS + FEATURE_COLUMNS


def validate_ihdp_schema(data, benchmark_mode=True):
    """
    Validate that an IHDP dataframe has the columns and values required by the experiment.

    Args:
        data: Input dataframe.
        benchmark_mode: Whether to require semi-synthetic benchmark columns used for
            true ITE, PEHE, and oracle policy value.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("IHDP data must be a pandas DataFrame.")

    required_columns = CORE_COLUMNS + FEATURE_COLUMNS
    if benchmark_mode:
        required_columns = required_columns + BENCHMARK_COLUMNS

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "IHDP data is missing required columns: "
            + ", ".join(missing_columns)
        )

    null_columns = data[required_columns].columns[data[required_columns].isna().any()].tolist()
    if null_columns:
        raise ValueError(
            "IHDP data contains missing values in columns: "
            + ", ".join(null_columns)
        )

    non_numeric_columns = [
        column for column in required_columns
        if column in NUMERIC_COLUMNS and not is_numeric_dtype(data[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            "IHDP data contains non-numeric values in columns: "
            + ", ".join(non_numeric_columns)
        )

    treatment_values = set(data["treatment"].dropna().unique())
    if not treatment_values.issubset({0, 1, False, True}):
        raise ValueError("Column 'treatment' must be binary with values 0/1 or False/True.")

    if {bool(value) for value in treatment_values} != {False, True}:
        raise ValueError("Column 'treatment' must include both treated and control examples.")


def validate_treatment_split(treatment, split_name):
    """Validate that a treatment vector contains both treated and control examples."""
    if not isinstance(treatment, pd.Series):
        treatment = pd.Series(treatment)

    treatment_values = set(treatment.dropna().unique())
    if {bool(value) for value in treatment_values} != {False, True}:
        raise ValueError(
            f"{split_name} split must include both treated and control examples."
        )
