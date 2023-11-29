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
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

# ***Helper and Primary Functions***

# Training
def train(
    training_data,
    outcome,
    treatment,
    sample,
):
    S = training_data[sample]  # indicator for the sample
    Y = training_data[outcome]  # outcome variable
    T = training_data[treatment]  # indicator for the treatment
    X = training_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates

    pi = S.mean()  # proportion of units in experimental study

    # P(S=1 | X)
    pi_m = en.AdaBoostRegressor().fit(X, (S/pi))

    # P(T=1 | X)
    e_m = lm.LogisticRegressionCV().fit(X.loc[S == 1], T.loc[S == 1])

    return pi, pi_m, e_m


# Estimation
def estimate(testing_data, outcome, treatment, sample, pi, pi_m, e_m):
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates

    # pi = P(S=1)
    pi = np.mean(S.values)

    # l(X) = (P(S=1 | X)/P(S=1)) / (P(S=0 | X)/P(S=0))
    lX = (pi_m.predict(X) / pi) / ( (1/pi - pi_m.predict(X)) / (1 - pi))

    # IPW Estimator for ATTE
    a = ((S * T * (Y)) / e_m.predict_proba(X)[:, 1]) - (
        (S * (1 - T) * (Y)) / e_m.predict_proba(X)[:, 0]
    )
    b = 1 / lX
    v = a * b
    
    return v, a, b


def estimate_dml(data, outcome, treatment, sample, crossfit=5):
    n = data.shape[0]  # total number of units
    df_v = []
    skf = StratifiedKFold(n_splits=crossfit)
    for i, (train_index, test_index) in enumerate(
        skf.split(data.drop(columns=[sample]), data[sample])
    ):
        print(f"Fold {i}")
        training_data, testing_data = data.iloc[train_index], data.iloc[test_index]
        pi, pi_m, e_m = train(training_data, outcome, treatment, sample)
        v, a, b = estimate(
            testing_data, outcome, treatment, sample, pi, pi_m, e_m
        )
        df_v_ = pd.DataFrame(v.values, columns=["te"], index=list(testing_data.index))
        df_v_["primary_index"] = list(testing_data.index)
        df_v_["a"] = a
        df_v_["b"] = b
        df_v.append(df_v_)
    df_v = pd.concat(df_v)
    # df_v = df_v.groupby(by='primary_index').mean()
    df_v["te_sq"] = (df_v["te"] - df_v["te"].loc[data[sample] == 1].mean()) ** 2
    df_v["a_sq"] = (df_v["a"] - df_v["a"].loc[data[sample] == 1].mean()) ** 2
    df_v = df_v.groupby(by="primary_index").mean().loc[data[sample] == 1]
    data2 = data.loc[df_v.index]
    return df_v, pi, pi_m, e_m, data2


def characterize_tree(X, w, max_depth=3):
    f = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X, w)
    return f


def split(split_feature, X, D, parent_loss, depth, explore_proba=0.05):
    fj = choose(split_feature, depth)
    # base case
    if fj == "leaf":
        losses = [loss(0, X.index, D), loss(1, X.index, D)]
        w_exploit = np.argmin(losses)
        w_explore = np.random.binomial(1, 0.5)
        explore = np.random.binomial(1, explore_proba)
        w = (explore * w_explore) + ((1 - explore) * w_exploit)
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
            if np.random.binomial(1, 0.5):
                return {
                    "node": fj,
                    "split": cj,
                    "left_tree": split(split_feature, X_left, D, new_loss, depth + 1),
                    "right_tree": split(split_feature, X_right, D, new_loss, depth + 1),
                    "local objective": np.nan_to_num(
                        np.sqrt(np.sum(D["vsq"] * D["w"]) / ((np.sum(D["w"]) ** 2))),
                        nan=np.inf,
                    ),
                    "depth": depth,
                }
            else:
                return {
                    "node": fj,
                    "split": cj,
                    "right_tree": split(split_feature, X_right, D, new_loss, depth + 1),
                    "left_tree": split(split_feature, X_left, D, new_loss, depth + 1),
                    "local objective": np.nan_to_num(
                        np.sqrt(np.sum(D["vsq"] * D["w"]) / ((np.sum(D["w"]) ** 2))),
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
    split_prob[0] = split_prob[0] * (2 ** (0 * depth / 4))
    split_prob = split_prob / np.sum(split_prob)
    fj = np.random.choice(a=list(split_feature.index), p=split_prob)
    # print(fj)
    return fj


def loss(val, indices, D):
    D_ = D.copy(deep=True)
    D_.loc[indices, "w"] = val
    se = np.nan_to_num(
        np.sqrt(np.sum(D_["vsq"] * D_["w"]) / ((np.sum(D_["w"]) ** 2))),
        nan=np.inf,
    )
    return se


def reduce_weight(fj, split_feature):
    split_feature.loc[fj] = split_feature.loc[fj] / 2
    split_feature = split_feature / np.sum(split_feature)
    return split_feature


# ***Optimization Functions***


def linear_opt(data, outcome, treatment, sample, seed=42):
    np.random.seed(seed)
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    v = df_v["te"]
    vsq = df_v["te_sq"]

    def obj(a):
        w = special.expit(
            np.matmul(X.values, a.reshape(-1, 1)).reshape(
                -1,
            )
        )
        se = np.nan_to_num(np.sqrt(np.sum(w * vsq) / (np.sum(w)) ** 2), np.inf)
        return se

    a_init = np.random.normal(0, 0.005, size=(X.shape[1],))
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
    se_unpruned = np.sqrt(np.sum(vsq) / (np.sum(np.ones_like(w))) ** 2)
    atte = np.sum(w * v) / np.sum(w)
    se = np.sqrt(np.sum(w * vsq) / (np.sum(w)) ** 2)

    D_labels = X.copy(deep=True)
    D_labels["v"] = v
    D_labels["vsq"] = vsq
    D_labels["S"] = S
    D_labels["w"] = w

    print((atte, se, atte_unpruned, se_unpruned))

    f = characterize_tree(X, w)
    return D_labels, result, f, testing_data


def kmeans_opt(data, outcome, treatment, sample, k=100, threshold=0.5):
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    v = df_v["te"]
    vsq = df_v["te_sq"]

    bounds = optimize.Bounds(lb=0, ub=1, keep_feasible=True)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

    D_labels = X.copy(deep=True)
    D_labels["v"] = v
    D_labels["vsq"] = vsq
    D_labels["S"] = S
    D_labels["group"] = kmeans.labels_
    D_labels["w"] = np.ones((n,))

    def obj(a):
        for l in range(k):
            D_labels.loc[D_labels["group"] == l, "w"] = a[l]
        w = D_labels["w"].values
        return np.nan_to_num(np.sqrt(np.sum(w * vsq) / (np.sum(w)) ** 2), np.inf)

    a_init = np.random.binomial(1, 0.95, size=k)

    result = optimize.minimize(
        obj,
        x0=a_init,
        method="CG",
        # options={"disp": True},
        bounds=bounds,
    )
    a = result.x
    for l in range(k):
        D_labels.loc[D_labels["group"] == l, "a"] = a[l]
        D_labels.loc[D_labels["group"] == l, "w"] = a[l] > threshold
    w = D_labels["w"].astype(int)

    atte_unpruned = np.sum(v) / np.sum(np.ones_like(v))
    se_unpruned = np.sqrt(np.sum(vsq) / (np.sum(np.ones_like(v))) ** 2)
    atte = np.sum(w * v) / np.sum(w * np.ones_like(v))
    se = np.sqrt(np.sum(w * vsq) / ((np.sum(w * np.ones_like(v))) ** 2))

    print((atte, se, atte_unpruned, se_unpruned))
    f = characterize_tree(X, w)
    return D_labels, f, testing_data


def tree_opt(data, outcome, treatment, sample, leaf_proba=0.25, seed=42):
    np.random.seed(seed)
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units

    v = df_v["te"]
    vsq = df_v["te_sq"]

    vsq_m = lm.Ridge().fit(X, vsq)

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
    D["v"] = v
    D["vsq"] = vsq
    D["w"] = np.ones_like(vsq)
    D["S"] = S
    np.random.seed(seed)
    w_tree = split(split_feature, D, D, np.inf, 0)
    return D, w_tree, testing_data


def forest_opt(
    data,
    outcome,
    treatment,
    sample,
    leaf_proba=0.25,
    seed=42,
    num_trees=10,
    vote_threshold=2 / 3,
    explore_proba=0.05,
    feature_est="Ridge",
    top_k_trees=False,
    k=10,
    cutoff="baseline",
):
    np.random.seed(seed)
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample, crossfit=5
    )
    S = testing_data[sample]  # indicator for the sample
    Y = testing_data[outcome]  # outcome variable
    T = testing_data[treatment]  # indicator for the treatment
    X = testing_data.drop(
        columns=[outcome, treatment, sample]
    )  # pre-treatment covariates
    n = testing_data.shape[0]  # total number of units
    v = df_v["te"]
    vsq = df_v["te_sq"]
    print("ATE Est: %.4f" % (v.mean()))

    features = ["leaf"] + list(X.columns)
    if feature_est == "Ridge":
        vsq_m = lm.Ridge().fit(X, vsq)
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
    else:
        vsq_m = en.GradientBoostingRegressor(n_estimators=100).fit(X, vsq)
        proba = np.array(
            [leaf_proba]
            + list(
                np.abs(
                    vsq_m.feature_importances_.reshape(
                        -1,
                    )
                )
                / np.sum(
                    np.abs(
                        vsq_m.feature_importances_.reshape(
                            -1,
                        )
                    )
                )
            )
        )

    proba = proba / np.sum(proba)
    split_feature = pd.Series(proba, index=features)
    print(split_feature)

    np.random.seed(seed)

    w_forest = []
    D_forest = X.copy(deep=True)
    D_forest["v"] = v
    D_forest["vsq"] = vsq
    D_forest["S"] = S

    # selection_model = lm.LogisticRegressionCV().fit(X,S)
    D_forest["l(X)"] = 1/df_v['b']

    for t_iter in range(num_trees):
        D = X.copy(deep=True)
        D["v"] = v
        D["vsq"] = vsq
        D["w"] = np.ones_like(vsq)
        D["S"] = S
        w_tree = split(split_feature, D, D, np.inf, 0, explore_proba=explore_proba)
        D_forest["w_tree_%d" % (t_iter)] = D["w"]
        w_forest += [w_tree]

    if top_k_trees:
        obj_trees = [w_forest[i]["local objective"] for i in range(len(w_forest))]
        idx = np.argpartition(obj_trees, k)
        rashomon_set = [i for i in idx[:k]]
        not_in_rashomon_set = [i for i in idx[k:]]

    else:
        if cutoff == "baseline":
            baseline_loss = np.sqrt(
                np.sum(D_forest["vsq"]) / ((D_forest.shape[0]) ** 2)
            )
            cutoff = baseline_loss
        not_in_rashomon_set = [
            i for i in range(len(w_forest)) if w_forest[i]["local objective"] >= cutoff
        ]
        rashomon_set = [
            i for i in range(len(w_forest)) if w_forest[i]["local objective"] < cutoff
        ]

    D_rash = D_forest.drop(columns=["w_tree_%d" % (i) for i in not_in_rashomon_set])

    D_w_rash = D_forest[["w_tree_%d" % (i) for i in rashomon_set]]
    D_rash["w_opt"] = (D_w_rash.mean(axis=1) > (vote_threshold)).astype(int)
    D_rash["vote_count"] = D_w_rash.sum(axis=1)
    f = characterize_tree(X, D_rash["w_opt"])
    return D_rash, D_forest, w_forest, rashomon_set, f, testing_data
