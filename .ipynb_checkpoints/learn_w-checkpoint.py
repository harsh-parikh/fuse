import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.tree as tree
import scipy.optimize as optimize
import estimate_atte as est
import scipy.special as special
from sklearn.model_selection import train_test_split


def nonparam_opt( estimator, data, outcome, treatment, sample ):
    def obj(w):
        est, se = estimator(data, outcome, treatment, sample, w)
        return se
    w_init = np.ones((data.shape[0],))
    result = optimize.minimize(obj, x0 = w_init)
    return result.x, result.success

def linear_opt( estimator, data, outcome, treatment, sample ):
    X = data.drop(columns=[outcome, treatment, sample])
    def obj(a):
        w = (special.expit( np.matmul(X.values,a.reshape(-1,1)).reshape(-1,) ) > 0.5)
        est, se = estimator(data, outcome, treatment, sample, w)
        return se
    a_init = np.zeros((X.shape[1],))
    result = optimize.minimize(obj, x0 = a_init, method='COBYLA')
    return result.x, result.success

def tree_opt( estimator, data, outcome, treatment, sample ):
    n = data.shape[0] # total number of units
    
    training_data, testing_data = train_test_split(data, test_size=0.25)
    
    # Training
    def train(training_data):
        S = training_data[sample] # indicator for the sample
        Y = training_data[outcome] # outcome variable
        T = training_data[treatment] # indicator for the treatment
        X = training_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates

        pi = S.mean() # proportion of units in experimental study

        # P(S=1 | X)
        pi_m = en.GradientBoostingClassifier().fit(X, S) x

        # E[ Y | S=1, T=1, X ]
        mu_1_m = en.GradientBoostingRegressor().fit(
            X.loc[(S == 1) * (T == 1)], Y.loc[(S == 1) * (T == 1)]
        )

        # E[ Y | S=1, T=0, X ]
        mu_0_m = en.GradientBoostingRegressor().fit(
            X.loc[(S == 1) * (T == 0)], Y.loc[(S == 1) * (T == 0)]
        )

        # P(T=1 | X)
        e_m = lm.LogisticRegressionCV().fit(X.loc[S == 1], T.loc[S == 1])
        
        return pi, pi_m, mu_1_m, mu_0_m, e_m
    
    # Estimation
    def estimate(testing_data):
        S = testing_data[sample] # indicator for the sample
        Y = testing_data[outcome] # outcome variable
        T = testing_data[treatment] # indicator for the treatment
        X = testing_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates

        # l(X) = P(S=1 | X) / P(S=0 | X)
        lX = pi_m.predict_proba(X)[:, 1] / pi_m.predict_proba(X)[:, 0]

        # IPW Estimator for ATTE
        a = ((S * T * (Y - mu_1_m.predict(X))) / e_m.predict_proba(X)[:, 1]) - (
            (S * (1 - T) * (Y - mu_0_m.predict(X))) / e_m.predict_proba(X)[:, 0]
        )
        b = 1 / lX
        v1 = a * b

        # outcome regression estimator for ATTE
        v2 = (1 - S) * (mu_1_m.predict(X) - mu_0_m.predict(X))
        
        v = v1 + v2
        vsq = (v - atte) ** 2
        
        return v, vsq
    
    pi, pi_m, mu_1_m, mu_0_m, e_m = train(training_data)
    v, vsq = estimate(testing_data)
    
    # ATTE est and standard error
    atte = np.sum( v ) / np.sum( (1 - S) )
    std_err = np.sqrt( np.sum( vsq ) / (np.sum( (1 - S) ))**2 )
