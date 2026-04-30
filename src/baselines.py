import numpy as np
from sklearn.linear_model import LinearRegression

def get_naive_ate(Y, T):
    return Y[T==1].mean() - Y[T==0].mean()

def regression_adjustment_ate(X_train, X_test, T_train, Y_train, T_test):
    model = LinearRegression()

    X_train_reg = np.column_stack([T_train, X_train])
    model.fit(X_train_reg, Y_train)

    y1_pred = model.predict(np.column_stack([np.ones_like(T_test), X_test]))
    y0_pred = model.predict(np.column_stack([np.zeros_like(T_test), X_test]))

    return (y1_pred - y0_pred).mean()