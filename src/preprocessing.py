import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test, X_validation=None):
    
    # Separate continuous and binary columns
    continuous_cols = X_train.select_dtypes(include=["float", "float64"]).columns.tolist()
    binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]

    scaler = StandardScaler()
    X_train_cont = pd.DataFrame(
        scaler.fit_transform(X_train[continuous_cols]),
        columns=continuous_cols,
        index=X_train.index
    )
    
    X_test_cont = pd.DataFrame(
        scaler.transform(X_test[continuous_cols]),
        columns=continuous_cols,
        index=X_test.index
    )

    X_validation_cont = pd.DataFrame(
        scaler.transform(X_validation[continuous_cols]),
        columns=continuous_cols,
        index=X_validation.index
    )
    
    # Combine the scaled continuous features with the binary features
    X_train_processed = pd.concat([X_train_cont, X_train[binary_cols]], axis=1)
    X_test_processed = pd.concat([X_test_cont, X_test[binary_cols]], axis=1)
    X_validation_processed = pd.concat([X_validation_cont, X_validation[binary_cols]], axis=1)

    # Ensure the same column order in test set as in training set
    X_test_processed = X_test_processed[X_train_processed.columns]
    X_validation_processed = X_validation_processed[X_train_processed.columns]

    return X_train_processed, X_test_processed, X_validation_processed
    


