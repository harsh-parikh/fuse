import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.tree as tree


def estimate_sim(data, outcome, treatment, sample, w=None):
    n = data.shape[0]

    S = data[sample]
    Y = data[outcome]
    T = data[treatment]
    X = data.drop(columns=[outcome, treatment, sample])

    if w is None:
        w = np.ones((n,))
        
    pi = S.mean()
    pi_m = en.GradientBoostingClassifier().fit(X, S)
    e_m = lm.LogisticRegressionCV().fit(X.loc[S == 1], T.loc[S == 1])

    lX = pi_m.predict_proba(X)[:, 1] / pi_m.predict_proba(X)[:, 0]
    b = 1 / lX
    a = ((S * T * Y) / e_m.predict_proba(X)[:, 1]) - (
        (S * (1 - T) * Y) / e_m.predict_proba(X)[:, 0]
    )
    v = a * b

    atte = np.sum( w * v ) / np.sum( w * b * S )
    std_err = np.sqrt(np.sum((v - atte) ** 2) / (n * pi)**2 )
    return atte, std_err


def estimate_out_reg(data, outcome, treatment, sample, w=None):
    n = data.shape[0]

    S = data[sample]
    Y = data[outcome]
    T = data[treatment]
    X = data.drop(columns=[outcome, treatment, sample])

    if w is None:
        w = np.ones((n,))
    
    pi = S.mean()
    pi_m = en.GradientBoostingClassifier().fit(X, S)
    mu_1_m = en.GradientBoostingRegressor().fit(
        X.loc[(S == 1) * (T == 1)], Y.loc[(S == 1) * (T == 1)]
    )
    mu_0_m = en.GradientBoostingRegressor().fit(
        X.loc[(S == 1) * (T == 0)], Y.loc[(S == 1) * (T == 0)]
    )

    v = (1 - S) * (mu_1_m.predict(X) - mu_0_m.predict(X))

    atte = np.sum( w * v ) / np.sum( w * (1 - S) )
    std_err = np.sqrt(np.sum((v - atte) ** 2) / (n * (1 - pi))**2 )
    return atte, std_err


def estimate_dml(data, outcome, treatment, sample, w=None):
    n = data.shape[0]

    S = data[sample]
    Y = data[outcome]
    T = data[treatment]
    X = data.drop(columns=[outcome, treatment, sample])
    
    if w is None:
        w = np.ones((n,))

    pi = S.mean()
    pi_m = en.GradientBoostingClassifier().fit(X, S)
    mu_1_m = en.GradientBoostingRegressor().fit(
        X.loc[(S == 1) * (T == 1)], Y.loc[(S == 1) * (T == 1)]
    )
    mu_0_m = en.GradientBoostingRegressor().fit(
        X.loc[(S == 1) * (T == 0)], Y.loc[(S == 1) * (T == 0)]
    )
    e_m = lm.LogisticRegressionCV().fit(X.loc[S == 1], T.loc[S == 1])

    lX = pi_m.predict_proba(X)[:, 1] / pi_m.predict_proba(X)[:, 0]
    a = ((S * T * (Y - mu_1_m.predict(X))) / e_m.predict_proba(X)[:, 1]) - (
        (S * (1 - T) * (Y - mu_0_m.predict(X))) / e_m.predict_proba(X)[:, 0]
    )
    b = 1 / lX

    v1 = a * b
    v2 = (1 - S) * (mu_1_m.predict(X) - mu_0_m.predict(X))

    v = v1 + v2

    atte = np.sum( w * v ) / (n * (1 - pi))
    std_err = np.sqrt( np.sum((v - atte) ** 2) / (n * (1 - pi))**2 )
    return atte, std_err
