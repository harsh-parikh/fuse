import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets


def gen_XY(n=1000, seed=0):
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, 20))
    coef = [(2 * np.random.binomial(1, 0.5) - 1) * (2 ** (-i)) for i in range(20)]
    coef2 = [4, 2, 1, 0.5, 0.25] + [0 for i in range(15)]
    # [
    #     (2 * np.random.binomial(1, 0.5) - 1)
    #     * np.random.binomial(1, 0.25)
    #     * (np.log(i + 1))
    #     for i in range(20)
    # ]
    Y0 = np.dot(X, coef)
    Y1 = Y0 + np.dot(X, coef2)
    p = X.shape[1]

    # p_discrete = 5
    # p_continuous = p - p_discrete
    # Xc = np.random.normal(0,1,size=(n,p_continuous))
    # Xd = np.random.binomial(1,0.5,size=(n,p_discrete))
    # X = np.hstack((Xc,Xd))
    return (
        pd.DataFrame(X, columns=["X%d" % (i) for i in range(p)]),
        pd.DataFrame(
            np.hstack((Y0.reshape(-1, 1), Y1.reshape(-1, 1))), columns=["Y0", "Y1"]
        ),
        coef2,
    )


def gen_S(X, seed=0):
    seed = seed + 1
    np.random.seed(seed)
    coef = [
        (2 * np.random.binomial(1, 0.5) - 1) * np.random.binomial(1, 0.6)
        for i in range(20)
    ]
    a = np.dot(X, coef)
    print((a, sp.expit(a)))
    S = np.random.binomial(1, sp.expit(a))
    return pd.DataFrame(S, columns=["S"]), coef


def gen_T(X, S, seed=0):
    seed = seed - 1
    np.random.seed(seed)
    pi_exp = 0.5 * np.ones((X.shape[0],))
    pi_obs = 0.5 * np.ones((X.shape[0],))
    pi = S["S"] * pi_exp + (1 - S["S"]) * pi_obs
    T = np.random.binomial(1, pi)
    return pd.DataFrame(T, columns=["T"]), pi


def get_data(n=1000, seed=42):
    seed = seed + 10
    X, Y, coef2 = gen_XY(n=n, seed=seed)
    S, coef = gen_S(X, seed=seed)
    T, pi = gen_T(X, S, seed=seed)
    X["Yobs"] = T["T"] * Y["Y1"] + (1 - T["T"]) * Y["Y0"]
    return pd.concat([X, S, T], axis=1), Y, {"selection": coef, "heterogeneity": coef2}
