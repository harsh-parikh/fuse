import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import sklearn.linear_model as lm
import sklearn.ensemble as en
from sklearn.model_selection import train_test_split

# *** Helper and Primary Functions ***

def train(training_data, outcome, treatment, sample):
    """
    Trains logistic regression models for sample selection and treatment assignment.

    Parameters:
    ----------
    training_data : pd.DataFrame
        Data containing outcome, treatment, and sample indicators along with covariates.
    outcome : str
        Column name for the outcome variable (Y).
    treatment : str
        Column name for the treatment indicator (T).
    sample : str
        Column name for the sample selection indicator (S).

    Returns:
    -------
    pi : float
        Proportion of units in the experimental study (P(S=1)).
    pi_m : LogisticRegressionCV
        Logistic regression model for estimating P(S=1 | X).
    e_m : LogisticRegressionCV
        Logistic regression model for estimating P(T=1 | X, S=1).
    """
    
    # Extract relevant variables
    S = training_data[sample]  # Sample indicator (S)
    Y = training_data[outcome]  # Outcome variable (Y)
    T = training_data[treatment]  # Treatment indicator (T)
    
    # Pre-treatment covariates (excluding outcome, treatment, and sample indicator)
    X = training_data.drop(columns=[outcome, treatment, sample])  

    # Estimate the overall probability of being in the experimental study
    pi = S.mean()  

    # Fit a logistic regression model to estimate P(S=1 | X)
    pi_m = LogisticRegressionCV().fit(X, S)

    # Fit a logistic regression model to estimate P(T=1 | X) within the experimental sample (S=1)
    e_m = LogisticRegressionCV().fit(X.loc[S == 1], T.loc[S == 1])

    return pi, pi_m, e_m


def estimate(testing_data, outcome, treatment, sample, pi, pi_m, e_m):
    """
    Estimates the Inverse Probability Weighting (IPW) effect using pre-trained models.

    Parameters:
    ----------
    testing_data : pd.DataFrame
        New dataset for estimation (contains covariates, outcome, treatment, and sample indicator).
    outcome : str
        Column name for the outcome variable (Y).
    treatment : str
        Column name for the treatment indicator (T).
    sample : str
        Column name for the sample selection indicator (S).
    pi : float
        Proportion of units in the experimental study (P(S=1)).
    pi_m : LogisticRegressionCV
        Pre-trained model for estimating P(S=1 | X).
    e_m : LogisticRegressionCV
        Pre-trained model for estimating P(T=1 | X, S=1).

    Returns:
    -------
    v : np.array
        Vector of estimated treatment effects.
    a : np.array
        Numerator of the IPW estimator.
    b : np.array
        Weighting factor for the IPW estimator.
    """

    # Extract relevant variables
    S = testing_data[sample]  # Sample indicator (S)
    Y = testing_data[outcome]  # Outcome variable (Y)
    T = testing_data[treatment]  # Treatment indicator (T)

    # Pre-treatment covariates (excluding outcome, treatment, and sample indicator)
    X = testing_data.drop(columns=[outcome, treatment, sample])

    # Estimate the sample probability
    pi = np.mean(S.values)

    # Compute selection likelihood ratio l(X) = (P(S=1 | X) / P(S=1)) / (P(S=0 | X) / P(S=0))
    lX = (pi_m.predict_proba(X)[:, 1] / pi) / ((pi_m.predict_proba(X)[:, 0]) / (1 - pi))

    # Compute the numerator of the IPW estimator
    a = ((S * T * Y) / e_m.predict_proba(X)[:, 1]) - ((S * (1 - T) * Y) / e_m.predict_proba(X)[:, 0])

    # Compute the denominator of the IPW estimator (weighting factor)
    b = 1 / lX

    # Compute the IPW estimate
    v = a * b

    return v, a, b


def estimate_ipw(data, outcome, treatment, sample):
    """
    Computes the IPW estimate for treatment effects with sample selection adjustments.

    Parameters:
    ----------
    data : pd.DataFrame
        Dataset containing covariates, outcome, treatment, and sample indicator.
    outcome : str
        Column name for the outcome variable (Y).
    treatment : str
        Column name for the treatment indicator (T).
    sample : str
        Column name for the sample selection indicator (S).

    Returns:
    -------
    df_v : pd.DataFrame
        DataFrame containing estimated treatment effects and additional statistics.
    pi : float
        Proportion of units in the experimental study (P(S=1)).
    pi_m : LogisticRegressionCV
        Logistic regression model for P(S=1 | X).
    e_m : LogisticRegressionCV
        Logistic regression model for P(T=1 | X, S=1).
    data2 : pd.DataFrame
        Filtered dataset used for estimation.
    """

    # Train logistic models on the given data
    pi, pi_m, e_m = train(data, outcome, treatment, sample)

    # Compute IPW estimates
    v, a, b = estimate(data, outcome, treatment, sample, pi, pi_m, e_m)

    # Store estimated treatment effects in a DataFrame
    df_v = pd.DataFrame(v, columns=["te"], index=data.index)
    df_v["primary_index"] = data.index
    df_v["a"] = a
    df_v["b"] = b

    # Handle infinite values and drop NaNs
    df_v.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_v.dropna(inplace=True)

    # Compute squared differences from mean within experimental sample (S=1)
    df_v["te_sq"] = (df_v["te"] - df_v["te"].loc[data[sample] == 1].mean()) ** 2
    df_v["a_sq"] = (df_v["a"] - df_v["a"].loc[data[sample] == 1].mean()) ** 2

    # Group by primary index and compute means for final estimation
    df_v = df_v.groupby(by="primary_index").mean().loc[data[sample] == 1]

    # Filter original dataset to match estimated indices
    data2 = data.loc[df_v.index]

    return df_v, pi, pi_m, e_m, data2


def characterize_tree(X, w, max_depth=3):
    """
    Fits a decision tree classifier to characterize the relationship between covariates and weights.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix used for training the decision tree.
    w : array-like
        Target variable (typically a weight or class label).
    max_depth : int, optional (default=3)
        Maximum depth of the decision tree.

    Returns:
    -------
    f : DecisionTreeClassifier
        Trained decision tree classifier.
    """
    f = DecisionTreeClassifier(max_depth=max_depth).fit(X, w)
    return f


def split(split_feature, X, D, parent_loss, depth, explore_proba=0.05):
    """
    Recursively splits the dataset to create a decision tree based on minimizing loss.

    Parameters:
    ----------
    split_feature : pd.Series
        Series indicating the probability of splitting on each feature.
    X : pd.DataFrame
        Feature matrix for the current subset of data.
    D : pd.DataFrame
        Global dataset containing additional variables including 'w' (weight assignment).
    parent_loss : float
        Loss value at the parent node.
    depth : int
        Current depth of the tree.
    explore_proba : float, optional (default=0.05)
        Probability of exploration (random choice instead of exploiting minimum loss).

    Returns:
    -------
    dict
        Dictionary representing the node structure of the tree.
    """

    # Choose a feature for splitting or determine if this is a leaf node
    fj = choose(split_feature, depth)

    # Base case: If 'fj' is a leaf node, assign treatment/exploration weights
    if fj == "leaf":
        # Compute loss for both treatment conditions (0 and 1)
        losses = [loss(0, X.index, D), loss(1, X.index, D)]
        w_exploit = np.argmin(losses)  # Choose the condition with minimum loss
        w_explore = np.random.binomial(1, 0.5)  # Random exploration assignment
        explore = np.random.binomial(1, explore_proba)  # Exploration decision
        w = (explore * w_explore) + ((1 - explore) * w_exploit)

        # Update weights in both local (X) and global (D) datasets
        D.loc[X.index, "w"] = w
        X.loc[X.index, "w"] = w

        return {
            "node": fj,
            "w": w,
            "local objective": np.min(losses),
            "depth": depth,
        }

    # Induction case: Split the data
    else:
        cj = midpoint(X[fj])  # Compute midpoint of the selected feature
        X_left = X.loc[X[fj] <= cj]
        X_right = X.loc[X[fj] > cj]

        # Compute loss for left and right branches
        loss_left = [loss(0, X_left.index, D), loss(1, X_left.index, D)]
        loss_right = [loss(0, X_right.index, D), loss(1, X_right.index, D)]
        min_loss_left = np.min(loss_left)
        min_loss_right = np.min(loss_right)

        # Compute new loss after the split
        new_loss = (X_left.shape[0] * min_loss_left + X_right.shape[0] * min_loss_right) / X.shape[0]

        # If splitting reduces the loss, proceed with the split
        if new_loss <= parent_loss:
            w_left = np.argmin(loss_left)  # Assign treatment with minimum loss for left node
            w_right = np.argmin(loss_right)  # Assign treatment with minimum loss for right node

            # Update weights in global (D) and local (X) datasets
            D.loc[X_left.index, "w"] = w_left
            X_left.loc[X_left.index, "w"] = w_left
            D.loc[X_right.index, "w"] = w_right
            X_right.loc[X_right.index, "w"] = w_right

            # Randomly decide left-first or right-first recursive splitting
            if np.random.binomial(1, 0.5):
                return {
                    "node": fj,
                    "split": cj,
                    "left_tree": split(split_feature, X_left, D, new_loss, depth + 1),
                    "right_tree": split(split_feature, X_right, D, new_loss, depth + 1),
                    "local objective": np.nan_to_num(
                        np.sqrt(np.sum(D["vsq"] * D["w"]) / (np.sum(D["w"]) ** 2)),
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
                        np.sqrt(np.sum(D["vsq"] * D["w"]) / (np.sum(D["w"]) ** 2)),
                        nan=np.inf,
                    ),
                    "depth": depth,
                }
        else:
            # If no improvement, reduce weight of feature and try again
            split_feature_updated = reduce_weight(fj, split_feature.copy(deep=True))
            return split(split_feature_updated, X, D, parent_loss, depth)


def midpoint(X):
    """
    Computes the midpoint of a given feature.

    Parameters:
    ----------
    X : pd.Series
        Feature values.

    Returns:
    -------
    float
        Midpoint value.
    """
    return (X.max() + X.min()) / 2


def choose(split_feature, depth):
    """
    Selects a feature to split on based on a probability distribution.

    Parameters:
    ----------
    split_feature : pd.Series
        Series containing feature selection probabilities.
    depth : int
        Current depth of the tree.

    Returns:
    -------
    str
        Selected feature name.
    """
    split_prob = split_feature.values
    split_prob[0] = split_prob[0] * (2 ** (0 * depth / 4))  # Adjust probability based on depth
    split_prob = split_prob / np.sum(split_prob)  # Normalize probabilities
    fj = np.random.choice(a=list(split_feature.index), p=split_prob)  # Select feature

    return fj


def loss(val, indices, D):
    """
    Computes the loss function based on weighted sum of squared errors.

    Parameters:
    ----------
    val : int (0 or 1)
        Treatment assignment value.
    indices : list or Index
        Indices of data points to compute loss for.
    D : pd.DataFrame
        Global dataset containing 'vsq' and 'w' columns.

    Returns:
    -------
    float
        Computed loss value.
    """
    D_ = D.copy(deep=True)
    D_.loc[indices, "w"] = val  # Assign treatment
    se = np.nan_to_num(
        np.sqrt(np.sum(D_["vsq"] * D_["w"]) / (np.sum(D_["w"]) ** 2)),
        nan=np.inf,
    )
    return se


def reduce_weight(fj, split_feature):
    """
    Reduces the probability of selecting a given feature in future splits.

    Parameters:
    ----------
    fj : str
        Feature to reduce weight for.
    split_feature : pd.Series
        Series of feature selection probabilities.

    Returns:
    -------
    pd.Series
        Updated split feature probabilities.
    """
    split_feature.loc[fj] = split_feature.loc[fj] / 2  # Reduce the probability
    split_feature = split_feature / np.sum(split_feature)  # Re-normalize
    return split_feature



def tree_opt(data, outcome, treatment, sample, leaf_proba=0.25, seed=42):
    """
    Optimizes tree-based partitioning

    This function estimates inverse probability weighting (IPW), computes feature importance 
    based on variance, and recursively partitions the data using a decision tree structure.

    Parameters:
    ----------
    data : pd.DataFrame
        Input dataset containing covariates, outcome, treatment, and sample indicator.
    outcome : str
        Column name for the outcome variable (Y).
    treatment : str
        Column name for the treatment indicator (T).
    sample : str
        Column name for the sample selection indicator (S).
    leaf_proba : float, optional (default=0.25)
        Probability of selecting a leaf node during tree splits.
    seed : int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    -------
    D : pd.DataFrame
        Dataset with added computed values (treatment effect estimates, squared differences, weights).
    f : DecisionTreeClassifier
        Trained decision tree classifier that characterizes the partitioning structure.
    testing_data : pd.DataFrame
        Filtered dataset used for estimation.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Estimate inverse probability weighting (IPW) for treatment effects
    df_v, pi, pi_m, e_m, testing_data = estimate_ipw(data, outcome, treatment, sample)

    # Extract relevant columns from the dataset
    S = testing_data[sample]  # Sample indicator (S)
    Y = testing_data[outcome]  # Outcome variable (Y)
    T = testing_data[treatment]  # Treatment indicator (T)
    
    # Pre-treatment covariates (excluding outcome, treatment, and sample indicator)
    X = testing_data.drop(columns=[outcome, treatment, sample])
    n = testing_data.shape[0]  # Total number of units in the test dataset

    # Treatment effect estimates and their squared differences
    v = df_v["te"]
    vsq = df_v["te_sq"]

    # Fit a ridge regression model to predict variance (vsq) based on X
    vsq_m = Ridge().fit(X, vsq)

    # Define splitting features (include 'leaf' as a stopping condition)
    features = ["leaf"] + list(X.columns)

    # Compute probabilities for feature selection based on absolute coefficients of vsq_m
    proba = np.array(
        [leaf_proba]  # Assign a probability for selecting a leaf node
        + list(
            np.abs(vsq_m.coef_.reshape(-1)) / np.sum(np.abs(vsq_m.coef_.reshape(-1)))
        )
    )
    proba = proba / np.sum(proba)  # Normalize probabilities

    # Convert feature probabilities into a Pandas Series
    split_feature = pd.Series(proba, index=features)

    # Create a copy of X to store additional computed values
    D = X.copy(deep=True)
    D["v"] = v  # Estimated treatment effect
    D["vsq"] = vsq  # Squared deviation of treatment effect
    D["w"] = np.ones_like(vsq)  # Initialize weights (uniform at start)
    D["S"] = S  # Store sample indicator

    # Recursively split the data using the computed feature selection probabilities
    np.random.seed(seed)
    w_tree = split(split_feature, D, D, np.inf, 0)

    # Train a decision tree classifier to characterize the final partitions
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
    """
    Args:
        data (pd.DataFrame): The input dataset containing covariates, treatment, and outcome.
        outcome (str): Column name representing the outcome variable.
        treatment (str): Column name representing the treatment indicator.
        sample (str): Column name representing the sample indicator.
        leaf_proba (float, optional): Probability of selecting a leaf for splitting. Defaults to 0.25.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        num_trees (int, optional): Number of trees to generate in the forest. Defaults to 10.
        vote_threshold (float, optional): Threshold for voting mechanism in ensemble selection. Defaults to 2/3.
        explore_proba (float, optional): Exploration probability for feature selection. Defaults to 0.05.
        feature_est (str, optional): Method for estimating feature importance ('Ridge' or 'GradientBoosting'). Defaults to 'Ridge'.
        top_k_trees (bool, optional): Whether to use the top-k trees based on local objective. Defaults to False.
        k (int, optional): Number of top trees to select if top_k_trees is True. Defaults to 10.
        cutoff (str or float, optional): Threshold for selecting Rashomon set trees ('baseline' or custom). Defaults to 'baseline'.
    
    Returns:
        tuple: (D_rash, D_forest, w_forest, rashomon_set, f, testing_data)
            - D_rash (pd.DataFrame): Data frame containing Rashomon set trees with optimized weights.
            - D_forest (pd.DataFrame): Full data frame including all decision trees.
            - w_forest (list): List of tree structures built during training.
            - rashomon_set (list): Indices of trees included in the Rashomon set.
            - f (dict): Characterization of the final decision tree structure.
            - testing_data (pd.DataFrame): The testing dataset used for evaluation.
    """
    np.random.seed(seed)
    
    # Estimate inverse probability weighting (IPW) for treatment effect adjustment
    df_v, pi, pi_m, e_m, testing_data = estimate_ipw(data, outcome, treatment, sample, crossfit=5)
    
    # Extract relevant variables from testing data
    S = testing_data[sample]  # Sample membership indicator
    Y = testing_data[outcome]  # Outcome variable
    T = testing_data[treatment]  # Treatment indicator
    X = testing_data.drop(columns=[outcome, treatment, sample])  # Pre-treatment covariates
    n = testing_data.shape[0]  # Total number of observations
    v = df_v["te"]  # Treatment effect estimates
    vsq = df_v["te_sq"]  # Squared treatment effect estimates
    
    print("ATE Estimate: %.4f" % v.mean())
    
    # Determine feature importance using Ridge Regression or Gradient Boosting
    features = ["leaf"] + list(X.columns)
    
    if feature_est == "Ridge":
        vsq_m = lm.Ridge().fit(X, vsq)  # Fit Ridge regression model
        feature_importance = np.abs(vsq_m.coef_).reshape(-1)
    else:
        vsq_m = en.GradientBoostingRegressor(n_estimators=100).fit(X, v)  # Fit Gradient Boosting model
        feature_importance = np.abs(vsq_m.feature_importances_).reshape(-1)
    
    # Compute probabilities for feature selection
    proba = np.array([leaf_proba] + list(feature_importance / np.sum(feature_importance)))
    proba /= np.sum(proba)  # Normalize probabilities
    
    split_feature = pd.Series(proba, index=features)
    print(split_feature)
    
    # Initialize storage for tree-based learning
    w_forest = []  # List to store tree structures
    D_forest = X.copy(deep=True)
    D_forest["v"] = v
    D_forest["vsq"] = vsq
    D_forest["S"] = S
    D_forest["l(X)"] = 1 / df_v['b']  # Sample selection weight
    
    # Build decision trees in the forest
    for t_iter in range(num_trees):
        D = X.copy(deep=True)
        D["v"] = v
        D["vsq"] = vsq
        D["w"] = np.ones_like(vsq)  # Initialize weights for splitting
        D["S"] = S
        
        # Perform feature splitting based on learned probabilities
        w_tree = split(split_feature, D, D, np.inf, 0, explore_proba=explore_proba)
        D_forest[f"w_tree_{t_iter}"] = D["w"]
        w_forest.append(w_tree)
    
    # Select Rashomon set trees based on local objective values
    if top_k_trees:
        obj_trees = [tree["local objective"] for tree in w_forest]
        idx = np.argpartition(obj_trees, k)
        rashomon_set = list(idx[:k])
        not_in_rashomon_set = list(idx[k:])
    else:
        if cutoff == "baseline":
            baseline_loss = np.sqrt(np.sum(D_forest["vsq"]) / (D_forest.shape[0] ** 2))
            cutoff = baseline_loss
        
        rashomon_set = [i for i in range(len(w_forest)) if w_forest[i]["local objective"] < cutoff]
        not_in_rashomon_set = [i for i in range(len(w_forest)) if w_forest[i]["local objective"] >= cutoff]
    
    # Filter Rashomon set trees from the complete forest
    D_rash = D_forest.drop(columns=[f"w_tree_{i}" for i in not_in_rashomon_set])
    D_w_rash = D_forest[[f"w_tree_{i}" for i in rashomon_set]]
    
    # Compute optimal weights based on voting mechanism
    D_rash["w_opt"] = (D_w_rash.mean(axis=1) > vote_threshold).astype(int)
    D_rash["vote_count"] = D_w_rash.sum(axis=1)
    
    # Characterize the final decision tree structure
    f = characterize_tree(X, D_rash["w_opt"])
    
    return D_rash, D_forest, w_forest, rashomon_set, f, testing_data


### Auxiliary Functions

def linear_opt(data, outcome, treatment, sample, seed=42):
    df_v, pi, pi_m, e_m, testing_data = estimate_ipw(
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
    
    D_labels = X.copy(deep=True)
    D_labels["v"] = v
    D_labels["vsq"] = vsq
    D_labels["S"] = S
    D_labels["w"] = np.ones((n,))
    
    np.random.seed(seed)

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

    D_labels["w"] = w

    f = characterize_tree(X, D_labels["w"])
    return D_labels, f, testing_data


def kmeans_opt(data, outcome, treatment, sample, k=100, threshold=0.5):
    df_v, pi, pi_m, e_m, testing_data = estimate_ipw(
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
    
    print(X.shape)
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