import pandas as pd
from .config import DATA_PATH
    
def load_data(DATA_PATH, header=0):
    data = pd.read_csv(DATA_PATH, header=header)
    return data     

def get_features_and_target(data):
    X = data.drop(['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'], axis = 1)
    T = data["treatment"]
    Y = data["y_factual"]

    true_ite = data["mu1"] - data["mu0"]
    
    return X, T, Y, true_ite