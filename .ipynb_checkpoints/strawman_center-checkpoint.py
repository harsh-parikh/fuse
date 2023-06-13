import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets

def gen_XY(n=1000, seed=0):
    np.random.seed(seed)
    X, Y0 = datasets.make_friedman1(n_samples=n, noise=1, random_state=seed)
    Y1 = Y0 + np.log(Y0 + 1)
    p = X.shape[1]

    # p_discrete = 5
    # p_continuous = p - p_discrete
    # Xc = np.random.normal(0,1,size=(n,p_continuous))
    # Xd = np.random.binomial(1,0.5,size=(n,p_discrete))
    # X = np.hstack((Xc,Xd))
    return pd.DataFrame(X, columns=["X%d" % (i) for i in range(p)]), pd.DataFrame(
        np.hstack((Y0.reshape(-1, 1), Y1.reshape(-1, 1))), columns=["Y0", "Y1"]
    )


def gen_S(X, seed=0):
    seed = seed + 1
    np.random.seed(seed)
    a = - 3 * ((X["X0"] > 0.4) * (X["X0"] < 0.6) * (X["X1"] > 0.4) * (X["X1"] < 0.6))
    S = np.random.binomial(1, sp.expit(a))
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