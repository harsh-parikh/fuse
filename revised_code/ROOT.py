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

# Training function
def train(training_data, outcome, treatment, sample):
    # Extracting relevant columns from the training data
    S = training_data[sample]  # Indicator for the sample
    Y = training_data[outcome]  # Outcome variable
    T = training_data[treatment]  # Indicator for the treatment
    X = training_data.drop(columns=[outcome, treatment, sample])  # Pre-treatment covariates

    # Calculating the proportion of units in the experimental study
    pi = S.mean()

    # Training an AdaBoostRegressor for P(S=1 | X)
    pi_m = en.AdaBoostRegressor().fit(X, (S/pi))

    # Training a LogisticRegressionCV for P(T=1 | X) on the subset where S=1
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
    # Total number of units
    n = data.shape[0]

    # List to store results for each fold
    df_v = []

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=crossfit)
    
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(skf.split(data.drop(columns=[sample]), data[sample])):
        print(f"Fold {i}")
        
        # Split data into training and testing sets
        training_data, testing_data = data.iloc[train_index], data.iloc[test_index]

        # Train the model using the training data
        pi, pi_m, e_m = train(training_data, outcome, treatment, sample)

        # Estimate treatment effects using the testing data
        v, a, b = estimate(testing_data, outcome, treatment, sample, pi, pi_m, e_m)

        # Create a DataFrame to store the results
        df_v_ = pd.DataFrame(v.values, columns=["te"], index=list(testing_data.index))
        df_v_["primary_index"] = list(testing_data.index)
        df_v_["a"] = a
        df_v_["b"] = b
        df_v.append(df_v_)

    # Concatenate results from all folds
    df_v = pd.concat(df_v)

    # Replace infinite values with NaN and drop NaN values
    df_v.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_v.dropna(inplace=True)

    # Calculate squared differences and group by primary index
    df_v["te_sq"] = (df_v["te"] - df_v["te"].loc[data[sample] == 1].mean()) ** 2
    df_v["a_sq"] = (df_v["a"] - df_v["a"].loc[data[sample] == 1].mean()) ** 2
    df_v = df_v.groupby(by="primary_index").mean().loc[data[sample] == 1]

    # Filter the original data based on the primary index
    data2 = data.loc[df_v.index]

    return df_v, pi, pi_m, e_m, data2


def characterize_tree(X, w, max_depth=3):
    f = tree.DecisionTreeClassifier(max_depth=max_depth).fit(X, w)
    return f

def split(split_feature, X, D, parent_loss, depth, explore_proba=0.05):
    # Choose a feature to split on
    fj = choose(split_feature, depth)

    # Base case: if fj is a leaf node
    if fj == "leaf":
        # Calculate losses for both treatment groups
        losses = [loss(0, X.index, D), loss(1, X.index, D)]
        
        # Select exploitation and exploration weights
        w_exploit = np.argmin(losses)
        w_explore = np.random.binomial(1, 0.5)
        
        # Randomly decide whether to explore
        explore = np.random.binomial(1, explore_proba)
        
        # Combine exploitation and exploration weights
        w = (explore * w_explore) + ((1 - explore) * w_exploit)
        
        # Update the global and local datasets with the selected weight
        D.loc[X.index, "w"] = w
        X.loc[X.index, "w"] = w
        
        # Return information about the leaf node
        return {"node": fj, "w": w, "local objective": np.min(losses), "depth": depth}
    
    # Induction case: fj is not a leaf node
    else:
        # Choose the midpoint for the split
        cj = midpoint(X[fj])
        
        # Split the dataset into left and right based on the chosen feature and midpoint
        X_left = X.loc[X[fj] <= cj]
        X_right = X.loc[X[fj] > cj]
        
        # Calculate losses for both treatment groups in the left and right splits
        loss_left = [loss(0, X_left.index, D), loss(1, X_left.index, D)]
        loss_right = [loss(0, X_right.index, D), loss(1, X_right.index, D)]
        
        # Find the minimum losses for left and right splits
        min_loss_left = np.min(loss_left)
        min_loss_right = np.min(loss_right)

        # Calculate the new loss for the current split
        new_loss = (X_left.shape[0] * min_loss_left + X_right.shape[0] * min_loss_right) / X.shape[0]

        # Check if the new loss is smaller than the parent loss
        if new_loss <= parent_loss:
            # Choose exploitation weights for left and right
            w_left = np.argmin(loss_left)
            w_right = np.argmin(loss_right)

            # Update the global and local datasets with the selected weights for left and right
            D.loc[X_left.index, "w"] = w_left
            X_left.loc[X_left.index, "w"] = w_left

            D.loc[X_right.index, "w"] = w_right
            X_right.loc[X_right.index, "w"] = w_right

            # Randomly choose the order of left and right subtrees
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
            # If the new loss is not smaller, update the split feature and try again
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

def tree_opt(data, outcome, treatment, sample, leaf_proba=0.25, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Estimate treatment effects using DML
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(data, outcome, treatment, sample)

    # Extract relevant columns from the testing data
    S = testing_data[sample]  # Indicator for the sample
    Y = testing_data[outcome]  # Outcome variable
    T = testing_data[treatment]  # Indicator for the treatment
    X = testing_data.drop(columns=[outcome, treatment, sample])  # Pre-treatment covariates

    # Total number of units
    n = testing_data.shape[0]

    # Extract treatment effect and its squared values
    v = df_v["te"]
    vsq = df_v["te_sq"]

    # Fit a Ridge regression model on squared treatment effect
    vsq_m = lm.Ridge().fit(X, vsq)

    # Define features for split and calculate probabilities
    features = ["leaf"] + list(X.columns)
    proba = np.array(
        [leaf_proba]
        + list(
            np.abs(vsq_m.coef_.reshape(-1,))
            / np.sum(np.abs(vsq_m.coef_.reshape(-1,)))
        )
    )
    proba = proba / np.sum(proba)
    split_feature = pd.Series(proba, index=features)

    # Initialize D matrix for tree optimization
    D = X.copy(deep=True)
    D["v"] = v
    D["vsq"] = vsq
    D["w"] = np.ones_like(vsq)
    D["S"] = S

    # Set random seed for tree splitting
    np.random.seed(seed)

    # Build the weighted tree
    w_tree = split(split_feature, D, D, np.inf, 0)

    # Characterize the tree to get the final weights
    f = characterize_tree(X, D["w"].astype(int))

    return D, f, testing_data


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

    # Estimate treatment effects using DML
    df_v, pi, pi_m, e_m, testing_data = estimate_dml(
        data, outcome, treatment, sample, crossfit=5
    )

    # Extract relevant columns from the testing data
    S = testing_data[sample]  # Indicator for the sample
    Y = testing_data[outcome]  # Outcome variable
    T = testing_data[treatment]  # Indicator for the treatment
    X = testing_data.drop(columns=[outcome, treatment, sample])  # Pre-treatment covariates
    n = testing_data.shape[0]  # Total number of units
    v = df_v["te"]
    vsq = df_v["te_sq"]
    print("ATE Est: %.4f" % (v.mean()))

    # Calculate features for tree splitting
    features = ["leaf"] + list(X.columns)

    # Choose the feature importance estimation method
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

    # Add inverse propensity scores to the dataset
    D_forest["l(X)"] = 1 / df_v['b']

    # Build the forest of weighted trees
    for t_iter in range(num_trees):
        D = X.copy(deep=True)
        D["v"] = v
        D["vsq"] = vsq
        D["w"] = np.ones_like(vsq)
        D["S"] = S
        w_tree = split(split_feature, D, D, np.inf, 0, explore_proba=explore_proba)
        D_forest["w_tree_%d" % (t_iter)] = D["w"]
        w_forest += [w_tree]

    # Select top k trees if specified
    if top_k_trees:
        obj_trees = [w_forest[i]["local objective"] for i in range(len(w_forest))]
        idx = np.argpartition(obj_trees, k)
        rashomon_set = [i for i in idx[:k]]
        not_in_rashomon_set = [i for i in idx[k:]]
    else:
        # Select trees based on the cutoff value
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

    # Create datasets with and without the selected trees
    D_rash = D_forest.drop(columns=["w_tree_%d" % (i) for i in not_in_rashomon_set])
    D_w_rash = D_forest[["w_tree_%d" % (i) for i in rashomon_set]]

    # Combine the weights of selected trees
    D_rash["w_opt"] = (D_w_rash.mean(axis=1) > (vote_threshold)).astype(int)
    D_rash["vote_count"] = D_w_rash.sum(axis=1)

    # Characterize the final tree ensemble
    f = characterize_tree(X, D_rash["w_opt"])

    return D_rash, D_forest, w_forest, rashomon_set, f, testing_data
