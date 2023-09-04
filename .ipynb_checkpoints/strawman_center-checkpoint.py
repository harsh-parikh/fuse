import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets


def gen_XY(n=1000, seed=0):
    np.random.seed(seed)
    C = np.random.binomial(1,0.75,size=(n))
    X0 = C * np.random.normal(0,1,size=(n)) + (1-C) * np.random.normal(4,1,size=(n))
    X1 = np.random.normal(X0,3,size=(n)) 
    Y0 = np.zeros(n)
    Y1 = X0**2 + X1**2
    X = pd.DataFrame()
    X['X0'] = X0
    X['X1'] = X1
    
    return X, pd.DataFrame(
        np.hstack((Y0.reshape(-1, 1), Y1.reshape(-1, 1))), columns=["Y0", "Y1"]
    )


def gen_S(X, seed=0):
    seed = seed + 1
    np.random.seed(seed)
    r = ( (X["X0"])**2  + (X["X1"])**2 )**(1/2)
    a = 0.5 * (r < 3) + 0.25 * (r >= 3) * (r < 5) + 0.05
    S = np.random.binomial(1, a)
    return pd.DataFrame(S, columns=["S"])


def gen_T(X, S, seed=0):
    seed = seed - 1
    np.random.seed(seed)
    pi_exp = 0.5 * np.ones((X.shape[0],))
    pi_obs = 0.5 * np.ones((X.shape[0],))
    pi = S["S"] * pi_exp + (1 - S["S"]) * pi_obs
    T = np.random.binomial(1, pi)
    return pd.DataFrame(T, columns=["T"]), pi


def get_data(n=1000, seed=10240):
    seed = seed + 10
    X, Y = gen_XY(n=n, seed=seed)
    S = gen_S(X, seed=seed)
    T, pi = gen_T(X, S, seed=seed)
    X["Yobs"] = T["T"] * Y["Y1"] + (1 - T["T"]) * Y["Y0"]
    return pd.concat([X, S, T], axis=1), Y
