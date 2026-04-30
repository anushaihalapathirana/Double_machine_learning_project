import numpy as np
from sympy import fraction

def get_positive_policy(ite):
    # Create a policy that assigns treatment if the predicted effect is positive
    policy = (ite > 0).astype(int)
    
    return policy

def get_fraction_policy(ite, fraction):
    # Create a policy that assigns treatment to the top k individuals with the highest predicted effect
    k = int(fraction * len(ite))

    policy = np.zeros(len(ite), dtype=int)
    top_indices = np.argsort(ite)[-k:]
    policy[top_indices] = 1

    return policy

def get_threshold_policy(ite, threshold):
    # Create a policy that assigns treatment if the predicted effect is above a certain threshold
    policy = (ite > threshold).astype(int)
    
    return policy