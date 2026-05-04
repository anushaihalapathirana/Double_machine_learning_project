from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler


@dataclass
class FeaturePreprocessor:
    continuous_cols: list
    binary_cols: list
    feature_columns: list
    scaler: StandardScaler

    def transform(self, X):
        missing_columns = [column for column in self.feature_columns if column not in X.columns]
        if missing_columns:
            raise ValueError(
                "Input data is missing required feature columns: "
                + ", ".join(missing_columns)
            )

        X = X[self.feature_columns]
        parts = []

        if self.continuous_cols:
            continuous = pd.DataFrame(
                self.scaler.transform(X[self.continuous_cols]),
                columns=self.continuous_cols,
                index=X.index,
            )
            parts.append(continuous)

        if self.binary_cols:
            parts.append(X[self.binary_cols].copy())

        if not parts:
            return pd.DataFrame(index=X.index)

        return pd.concat(parts, axis=1)[self.feature_columns]


def _is_binary_column(series):
    values = set(series.dropna().unique())
    return len(values) == 2 and values.issubset({0, 1, False, True})


def fit_preprocessor(X_train):
    """
    Fit a reusable feature preprocessor on training features.

    Binary 0/1 or False/True columns are preserved. All other numeric columns are
    standardized using training-set statistics.
    """
    non_numeric_cols = [
        column for column in X_train.columns
        if not is_numeric_dtype(X_train[column])
    ]
    if non_numeric_cols:
        raise ValueError(
            "All feature columns must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric_cols)
        )

    feature_columns = X_train.columns.tolist()
    binary_cols = [column for column in feature_columns if _is_binary_column(X_train[column])]
    continuous_cols = [column for column in feature_columns if column not in binary_cols]

    scaler = StandardScaler()
    if continuous_cols:
        scaler.fit(X_train[continuous_cols])

    return FeaturePreprocessor(
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        feature_columns=feature_columns,
        scaler=scaler,
    )


def preprocess_data(X_train, X_test, X_validation):
    """
    Backward-compatible helper for the experiment pipeline.

    Prefer fit_preprocessor(X_train).transform(...) when the fitted preprocessing
    object needs to be reused for inference.
    """
    preprocessor = fit_preprocessor(X_train)

    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_test),
        preprocessor.transform(X_validation),
    )
