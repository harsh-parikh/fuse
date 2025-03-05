import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets

def gen_XY(n=1000, seed=0):
    """
    Generate covariates (X) and potential outcomes (Y0, Y1) with a nonlinear relationship.
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrames for X and Y.
    """
    np.random.seed(seed)
    
    # Generate a binary categorical variable C
    C = np.random.binomial(1, 0.75, size=n)
    
    # Generate two covariates X0 and X1
    X0 = C * np.random.normal(0, 1, size=n) + (1 - C) * np.random.normal(4, 1, size=n)
    X1 = np.random.normal(X0, 3, size=n)
    
    # Define potential outcomes
    Y0 = np.zeros(n)  # No treatment effect
    Y1 = X0**2 + X1**2  # Treatment effect is a quadratic function of X0 and X1
    
    # Create a DataFrame for covariates
    X = pd.DataFrame({"X0": X0, "X1": X1})
    
    return X, pd.DataFrame({"Y0": Y0, "Y1": Y1})

def gen_S(X, seed=0):
    """
    Generate selection indicator S based on the distance from the origin in the X0-X1 plane.
    
    Args:
        X (DataFrame): Covariate matrix.
        seed (int): Random seed for reproducibility.

    Returns:
        DataFrame: Selection indicator S.
    """
    np.random.seed(seed + 1)
    
    # Compute Euclidean distance from origin
    r = np.sqrt(X["X0"]**2 + X["X1"]**2)
    
    # Define probability of selection based on distance
    a = 0.5 * (r < 3) + 0.25 * ((r >= 3) & (r < 5)) + 0.05
    
    # Generate selection indicator
    S = np.random.binomial(1, a)
    
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
    
    pi_exp = 0.5 * np.ones((X.shape[0],))  # Treatment probability in experimental setting
    pi_obs = 0.5 * np.ones((X.shape[0],))  # Treatment probability in observational setting
    
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
