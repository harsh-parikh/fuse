import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets

def gen_XY(n=1000, seed=0):
    """
    Generate covariates (X) and potential outcomes (Y0, Y1) using the Friedman1 dataset.
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrames for X and Y.
    """
    np.random.seed(seed)
    
    # Generate Friedman1 dataset
    X, Y0 = datasets.make_friedman1(n_samples=n, noise=1, random_state=seed)
    Y1 = Y0 + np.log(Y0 + 1)  # Treatment effect with a nonlinear transformation
    
    p = X.shape[1]
    return pd.DataFrame(X, columns=[f"X{i}" for i in range(p)]), pd.DataFrame({"Y0": Y0, "Y1": Y1})

def gen_S(X, seed=0):
    """
    Generate selection indicator S based on covariates X.
    
    Args:
        X (DataFrame): Covariate matrix.
        seed (int): Random seed for reproducibility.

    Returns:
        DataFrame: Selection indicator S.
    """
    np.random.seed(seed + 1)
    
    # Define probability of selection based on conditions on X0 and X1
    a = 0.25 - 2 * ((X["X0"] > 0.5) & (X["X0"] < 1) & (X["X1"] > 0.5) & (X["X1"] < 1))
    
    # Generate selection indicator using logistic function
    S = np.random.binomial(1, sp.expit(a))
    
    return pd.DataFrame({"S": S})

def gen_T(X, S, seed=0):
    """
    Generate treatment assignment T based on selection indicator S.
    
    Args:
        X (DataFrame): Covariate matrix.
        S (DataFrame): Selection indicator.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrame of treatment assignment T and assignment probabilities.
    """
    np.random.seed(seed - 1)
    
    pi_exp = 0.5  # Constant treatment probability in experimental setting
    pi_obs = sp.expit(X["X0"])  # Treatment probability depends on X0 in observational setting
    
    # Assign probability based on selection mechanism
    pi = S["S"] * pi_exp + (1 - S["S"]) * pi_obs
    
    # Generate treatment assignment
    T = np.random.binomial(1, pi)
    
    return pd.DataFrame({"T": T}), pi

def get_data(n=1000, seed=10240):
    """
    Generate a complete dataset with covariates (X), treatment assignment (T),
    selection indicator (S), and observed outcomes (Yobs).
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Full dataset with observed outcomes and true potential outcomes.
    """
    seed += 10  # Offset seed to avoid correlations across functions
    
    # Generate X and potential outcomes
    X, Y = gen_XY(n=n, seed=seed)
    
    # Generate selection mechanism
    S = gen_S(X, seed=seed)
    
    # Generate treatment assignment
    T, pi = gen_T(X, S, seed=seed)
    
    # Compute observed outcome based on treatment assignment
    X["Yobs"] = T["T"] * Y["Y1"] + (1 - T["T"]) * Y["Y0"]
    
    return pd.concat([X, S, T], axis=1), Y
