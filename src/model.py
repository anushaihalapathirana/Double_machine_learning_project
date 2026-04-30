from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from src.config import RANDOM_STATE

def get_model(model_type="LinearDML_RF"):
    """
    Function to get different DML estimators.
    
    Args:
        model_type: One of:
            - "LinearDML_RF": LinearDML with Random Forest (default)
            - "LinearDML_GB": LinearDML with Gradient Boosting
            - "CausalForestDML": Causal Forest DML
            - "DML_Lasso": LinearDML with Lasso for high-dimensional settings
    
    Returns:
        A DML model instance
    """
    if model_type == "LinearDML_RF":
        model = LinearDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            discrete_treatment=True,
            random_state=RANDOM_STATE
        )
    elif model_type == "LinearDML_GB":
        model = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            discrete_treatment=True,
            random_state=RANDOM_STATE
        )
    elif model_type == "CausalForestDML":
        model = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            model_t=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE ),
            discrete_treatment=True,
            random_state=RANDOM_STATE
        )
    elif model_type == "DML_Lasso":
        model = LinearDML(
            model_y=Lasso(alpha=1.0, random_state=RANDOM_STATE),
            model_t=LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            discrete_treatment=True,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def get_all_models():
    """
    Returns a dictionary of all available DML estimators for comparison.
    """
    return {
        "LinearDML_RF": get_model("LinearDML_RF"),
        "LinearDML_GB": get_model("LinearDML_GB"),
        "CausalForestDML": get_model("CausalForestDML"),
        "DML_Lasso": get_model("DML_Lasso")
    }

def train_model(model, X, T, Y):
    model.fit(Y, T, X=X)
    return model

