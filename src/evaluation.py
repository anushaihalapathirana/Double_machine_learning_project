import numpy as np
import pandas as pd

def ate_error(estimated_ate, true_ate):
    # Calculate the absolute error between the estimated ATE and the true ATE
    return abs(estimated_ate - true_ate)

def pehe(estimated_ite, true_ite):
    # Calculate the Precision in Estimation of Heterogeneous Effect (PEHE)
    # PEHE is the root mean squared error between the estimated and true individual treatment effects
    # It measures how well the model estimates the treatment effect for each individual
    # A lower PEHE indicates better performance in estimating heterogeneous treatment effects
    # PEHE requires ground truth ITE, so it is only available in synthetic or semi-synthetic datasets like IHDP.
    return np.sqrt(np.mean((estimated_ite - true_ite) ** 2))

def policy_value(policy, mu0, mu1):
    # Calculate the expected outcome under a given policy
    # The policy is a binary vector indicating whether to treat (1) or not (0) for each individual
    # mu0 and mu1 are the potential outcomes under control and treatment, respectively
    outcome = policy * mu1 + (1 - policy) * mu0
    return np.mean(outcome)

def evaluate_policies(policy, T_test, mu0_test, mu1_test):  
    # Create a random policy for comparison
    # The random policy assigns treatment with a 50% probability, independent of the predicted effects
    random_policy = np.random.binomial(1, 0.5, size=len(policy))

    results = {
        "Observed": policy_value(T_test, mu0_test, mu1_test),
        "Random": policy_value(random_policy, mu0_test, mu1_test),
        "DML Policy": policy_value(policy, mu0_test, mu1_test),
        "Treat All": mu1_test.mean(),
        "Treat None": mu0_test.mean(),
    }

    return pd.DataFrame.from_dict(results, orient="index", columns=["Policy Value"])