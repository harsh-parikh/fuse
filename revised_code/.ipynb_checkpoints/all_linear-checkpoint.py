import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets

def gen_XY(n=1000, seed=0):
    """
    Generate covariates (X) and potential outcomes (Y0, Y1) with a linear relationship.
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrames for X and Y, and the coefficient vector used for generating Y1.
    """
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, 20))  # Generate 20 normally distributed features
    
    # Generate coefficient vectors for the main effect and treatment heterogeneity
    coef = [(2 * np.random.binomial(1, 0.5) - 1) * (2 ** (-i)) for i in range(20)]
    coef2 = [4, 2, 1, 0.5, 0.25] + [0] * 15  # Treatment effect is concentrated in first 5 features
    
    # Compute potential outcomes
    Y0 = np.dot(X, coef)  # Base outcome
    Y1 = Y0 + np.dot(X, coef2)  # Treatment effect shifts the outcome
    
    p = X.shape[1]
    return (
        pd.DataFrame(X, columns=[f"X{i}" for i in range(p)]),
        pd.DataFrame(np.hstack((Y0.reshape(-1, 1), Y1.reshape(-1, 1))), columns=["Y0", "Y1"]),
        coef2,
    )

def gen_S(X, seed=0):
    """
    Generate selection indicator S based on covariates X.
    
    Args:
        X (DataFrame): Covariate matrix.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: DataFrame of selection indicator S and the coefficient vector used.
    """
    np.random.seed(seed + 1)
    
    # Generate coefficient vector for selection mechanism
    coef = [(2 * np.random.binomial(1, 0.5) - 1) * np.random.binomial(1, 0.6) for _ in range(20)]
    
    # Compute selection probabilities using a logistic function
    a = np.dot(X, coef)
    S = np.random.binomial(1, sp.expit(a))  # Apply sigmoid transformation
    
    return pd.DataFrame(S, columns=["S"]), coef

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
    
    return pd.DataFrame(T, columns=["T"]), pi

def get_data(n=1000, seed=42):
    """
    Generate a complete dataset with covariates (X), treatment assignment (T),
    selection indicator (S), and observed outcomes (Yobs).
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Full dataset with observed outcomes, true potential outcomes, and coefficient metadata.
    """
    seed += 10  # Offset seed to avoid correlations across functions
    
    # Generate X and potential outcomes
    X, Y, coef2 = gen_XY(n=n, seed=seed)
    
    # Generate selection mechanism
    S, coef = gen_S(X, seed=seed)
    
    # Generate treatment assignment
    T, pi = gen_T(X, S, seed=seed)
    
    # Compute observed outcome based on treatment assignment
    X["Yobs"] = T["T"] * Y["Y1"] + (1 - T["T"]) * Y["Y0"]
    
    return pd.concat([X, S, T], axis=1), Y, {"selection": coef, "heterogeneity": coef2}
