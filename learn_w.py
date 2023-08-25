import numpy as np
import pandas as pd
import scipy.special as sp
import sklearn.datasets as datasets
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.tree as tree
import scipy.optimize as optimize
import scipy.special as special
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def estimate_dml(data, outcome, treatment, sample, w=None):
    n = data.shape[0] # total number of units
    
    training_data, testing_data = train_test_split(data, test_size=0.5)
    
    # Training
    def train(training_data):
        S = training_data[sample] # indicator for the sample
        Y = training_data[outcome] # outcome variable
        T = training_data[treatment] # indicator for the treatment
        X = training_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates

        pi = S.mean() # proportion of units in experimental study

        # P(S=1 | X)
        pi_m = en.GradientBoostingClassifier().fit(X, S)

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
        atte = np.sum( v ) / np.sum( (1 - S) )
        vsq = (v - atte) ** 2
        
        return v, vsq
    
    pi, pi_m, mu_1_m, mu_0_m, e_m = train(training_data)
    v, vsq = estimate(testing_data)
    
    return v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data
    

def nonparam_opt( data, outcome, treatment, sample ):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(data, outcome, treatment, sample)
    S = testing_data[sample] # indicator for the sample
    Y = testing_data[outcome] # outcome variable
    T = testing_data[treatment] # indicator for the treatment
    X = testing_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates
    n = testing_data.shape[0] # total number of units
    def obj(w):
        se = np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
        return se
    w_init = np.ones((n,))
    result = optimize.minimize(obj, x0 = w_init, method='SLSQP', options={'disp':True})
    w = result.x
    atte_unpruned = np.sum( v ) / np.sum( (1 - S) )
    se_unpruned = np.sqrt( np.sum( vsq ) / ( np.sum( (1 - S) ) )**2 )
    atte = np.sum( w * v ) / np.sum( w * (1 - S) )
    se = np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
    
    testing_data['w'] = w
    return result, atte, se, atte_unpruned, se_unpruned, w

def linear_opt( data, outcome, treatment, sample ):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(data, outcome, treatment, sample)
    S = testing_data[sample] # indicator for the sample
    Y = testing_data[outcome] # outcome variable
    T = testing_data[treatment] # indicator for the treatment
    X = testing_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates
    n = testing_data.shape[0] # total number of units
    def obj(a):
        w = (special.expit( np.matmul(X.values,a.reshape(-1,1)).reshape(-1,) ) > 0.5)
        se = np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
        return se
    a_init = np.random.normal(0,5,size=(X.shape[1],))
    result = optimize.minimize(obj, x0 = a_init, method='SLSQP', options={'disp':True})
    a = result.x
    w = (special.expit( np.matmul(X.values,a.reshape(-1,1)).reshape(-1,) ) > 0.5)
    atte_unpruned = np.sum( v ) / np.sum( (1 - S) )
    se_unpruned = np.sqrt( np.sum( vsq ) / ( np.sum( (1 - S) ) )**2 )
    atte = np.sum( w * v ) / np.sum( w * (1 - S) )
    se = np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
    
    testing_data['w'] = w
    return result, atte, se, atte_unpruned, se_unpruned, w

def kmeans_opt( data, outcome, treatment, sample, k=100 ):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(data, outcome, treatment, sample)
    S = testing_data[sample] # indicator for the sample
    Y = testing_data[outcome] # outcome variable
    T = testing_data[treatment] # indicator for the treatment
    X = testing_data.drop(columns=[outcome, treatment, sample]) # pre-treatment covariates
    n = testing_data.shape[0] # total number of units
    
    bounds = optimize.Bounds(lb=0, ub=1, keep_feasible=True)
    kmeans = KMeans(n_clusters=k, 
                    random_state=0, 
                    n_init="auto").fit(X)
    
    labels = pd.DataFrame(kmeans.labels_,index=X.index,columns=['group'])
    labels['w'] = np.ones((n,))
    
    def obj(a):
        for l in range(k):
            labels.loc[labels['group'] == l,'w'] = a[l]
        w = labels['w'].values
        return np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
   
    a_init = np.ones((k,))
    
    result = optimize.minimize(obj, x0 = a_init, method='SLSQP', options={'disp':True},bounds=bounds)
    a = result.x
    for l in range(k):
        labels.loc[labels['group'] == l,'w'] = a[l]
    w = labels['w'].values
    
    atte_unpruned = np.sum( v ) / np.sum( (1 - S) )
    se_unpruned = np.sqrt( np.sum( vsq ) / ( np.sum( (1 - S) ) )**2 )
    atte = np.sum( w * v ) / np.sum( w * (1 - S) )
    se = np.sqrt( np.sum( w * vsq ) / ( np.sum( w * (1 - S) ) )**2 )
    
    testing_data['w'] = w
    return result, atte, se, atte_unpruned, se_unpruned, testing_data