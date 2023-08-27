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
    n = data.shape[0]  # total number of units

    training_data, testing_data = train_test_split(data, test_size=0.5)

    # Training
    def train(training_data):
        S = training_data[sample]  # indicator for the sample
        Y = training_data[outcome]  # outcome variable
        T = training_data[treatment]  # indicator for the treatment
        X = training_data.drop(
            columns=[outcome, treatment, sample]
        )  # pre-treatment covariates

        pi = S.mean()  # proportion of units in experimental study

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
        S = testing_data[sample]  # indicator for the sample
        Y = testing_data[outcome]  # outcome variable
        T = testing_data[treatment]  # indicator for the treatment
        X = testing_data.drop(
            columns=[outcome, treatment, sample]
        )  # pre-treatment covariates

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
        atte = np.sum(v) / np.sum((1 - S))
        vsq = (v - atte) ** 2

        return v, vsq

    pi, pi_m, mu_1_m, mu_0_m, e_m = train(training_data)
    v, vsq = estimate(testing_data)

    return v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data


def characterize_tree(X, w):
    f = tree.DecisionTreeRegressor(max_leaf_nodes=4).fit(X, w)
    return f


def nonparam_opt(data, outcome, treatment, sample):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    def obj(w):
        se = np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)
        return se

    w_init = np.ones((n,))
    result = optimize.minimize(obj, x0=w_init, method="SLSQP", options={"disp": True})
    w = result.x
    atte_unpruned = np.sum(v) / np.sum((1 - S))
    se_unpruned = np.sqrt(np.sum(vsq) / (np.sum((1 - S))) ** 2)
    atte = np.sum(w * v) / np.sum(w * (1 - S))
    se = np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)

    testing_data["w"] = w

    f = characterize_tree(X, w)
    return result, atte, se, atte_unpruned, se_unpruned, f, testing_data


def linear_opt(data, outcome, treatment, sample):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    def obj(a):
        w = (
            special.expit(
                np.matmul(X.values, a.reshape(-1, 1)).reshape(
                    -1,
                )
            )
            > 0.5
        )
        se = np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)
        return se

    a_init = np.random.normal(0, 5, size=(X.shape[1],))
    result = optimize.minimize(obj, x0=a_init, method="SLSQP", options={"disp": True})
    a = result.x
    w = (
        special.expit(
            np.matmul(X.values, a.reshape(-1, 1)).reshape(
                -1,
            )
        )
        > 0.5
    )
    atte_unpruned = np.sum(v) / np.sum((1 - S))
    se_unpruned = np.sqrt(np.sum(vsq) / (np.sum((1 - S))) ** 2)
    atte = np.sum(w * v) / np.sum(w * (1 - S))
    se = np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)

    testing_data["w"] = w
    f = characterize_tree(X, w)
    return result, atte, se, atte_unpruned, se_unpruned, f, testing_data


def kmeans_opt(data, outcome, treatment, sample, k=100):
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    bounds = optimize.Bounds(lb=0, ub=1, keep_feasible=True)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

    labels = pd.DataFrame(kmeans.labels_, index=X.index, columns=["group"])
    labels["w"] = np.ones((n,))

    def obj(a):
        for l in range(k):
            labels.loc[labels["group"] == l, "w"] = a[l]
        w = labels["w"].values
        return np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)

    a_init = np.ones((k,))

    result = optimize.minimize(
        obj,
        x0=a_init,
        method="COBYLA",
        options={"disp": False},
    )
    a = result.x
    for l in range(k):
        labels.loc[labels["group"] == l, "w"] = a[l] > 0.5
    w = labels["w"].values

    atte_unpruned = np.sum(v) / np.sum((1 - S))
    se_unpruned = np.sqrt(np.sum(vsq) / (np.sum((1 - S))) ** 2)
    atte = np.sum(w * v) / np.sum(w * (1 - S))
    se = np.sqrt(np.sum(w * vsq) / (np.sum(w * (1 - S))) ** 2)

    testing_data["w"] = w

    f = characterize_tree(X, w)
    return result, atte, se, atte_unpruned, se_unpruned, f, testing_data


def tree_opt(data, outcome, treatment, sample, leaf_proba=0.25, seed=42):
    np.random.seed(seed)
    v, vsq, pi, pi_m, mu_1_m, mu_0_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    vsq_m = lm.RidgeCV().fit(X, vsq)

    features = ["leaf"] + list(X.columns)
    proba = np.array(
        [leaf_proba]
        + list(
            np.abs(
                vsq_m.coef_.reshape(
                    -1,
                )
            )
            / np.sum(
                np.abs(
                    vsq_m.coef_.reshape(
                        -1,
                    )
                )
            )
        )
    )
    proba = proba / np.sum(proba)
    split_feature = pd.Series(proba, index=features)
    # print(split_feature)

    D = X.copy(deep=True)
    D["vsq"] = vsq
    D["w"] = np.ones_like(vsq)
    D["S"] = S
    np.random.seed(seed)
    w_tree = split(split_feature, D, D, np.inf, 0)
    return w_tree, D, testing_data


def split(split_feature, X, D, parent_loss, depth):
    fj = choose(split_feature, depth)
    # base case
    if fj == "leaf":
        losses = [loss(0, X.index, D), loss(1, X.index, D)]
        w_exploit = np.argmin(losses)
        w_explore = np.random.binomial(1, 0.5)
        explore = np.random.binomial(1, 0.05)
        w = (explore * w_explore) + ( (1-explore) * w_exploit )
        D.loc[X.index, "w"] = w  # update the global dataset
        X.loc[
            X.index, "w"
        ] = w  # update the local dataset that will be used for recursion
        return {"node": fj, "w": w, "local objective": np.min(losses), "depth": depth}
    # induction case
    else:
        cj = midpoint(X[fj])
        X_left = X.loc[X[fj] <= cj]
        X_right = X.loc[X[fj] > cj]
        loss_left = [loss(0, X_left.index, D), loss(1, X_left.index, D)]
        loss_right = [loss(0, X_right.index, D), loss(1, X_right.index, D)]
        min_loss_left = np.min(loss_left)
        min_loss_right = np.min(loss_right)

        new_loss = (
            X_left.shape[0] * min_loss_left + X_right.shape[0] * min_loss_right
        ) / (X.shape[0])
        if new_loss <= parent_loss:
            w_left = np.argmin(loss_left)
            w_right = np.argmin(loss_right)

            D.loc[X_left.index, "w"] = w_left  # update the global dataset
            X_left.loc[
                X_left.index, "w"
            ] = w_left  # update the local dataset that will be used for recursion

            D.loc[X_right.index, "w"] = w_right  # update the global dataset
            X_right.loc[
                X_right.index, "w"
            ] = w_right  # update the local dataset that will be used for recursion
            return {
                "node": fj,
                "split": cj,
                "left_tree": split(split_feature, X_left, D, new_loss, depth + 1),
                "right_tree": split(split_feature, X_right, D, new_loss, depth + 1),
                "local objective": np.nan_to_num(
                    np.sqrt(
                        np.sum(D["vsq"] * D["w"])
                        / ((np.sum((1 - D["S"]) * D["w"]) ** 2))
                    ),
                    nan=np.inf,
                ),
                "depth": depth,
            }

        else:
            split_feature_updated = reduce_weight(fj, split_feature.copy(deep=True))
            return split(split_feature_updated, X, D, parent_loss, depth)


def midpoint(X):
    return (X.max() + X.min()) / 2


def choose(split_feature, depth):
    # print(split_feature)
    split_prob = split_feature.values
    split_prob[0] = split_prob[0] * (2 ** (depth/4))
    split_prob = split_prob / np.sum(split_prob)
    fj = np.random.choice(a=list(split_feature.index), p=split_prob)
    # print(fj)
    return fj


def loss(val, indices, D):
    D_ = D.copy(deep=True)
    D_.loc[indices, "w"] = val
    se = np.nan_to_num(
        np.sqrt(np.sum(D_["vsq"] * D_["w"]) / ((np.sum((1 - D_["S"]) * D_["w"]) ** 2))),
        nan=np.inf,
    )
    return se


def reduce_weight(fj, split_feature):
    split_feature.loc[fj] = split_feature.loc[fj] / 2
    split_feature = split_feature / np.sum(split_feature)
    return split_feature
