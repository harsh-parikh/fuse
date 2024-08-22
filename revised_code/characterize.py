import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import scipy.special as sp
import sklearn.datasets as datasets
import sklearn.linear_model as lm
import sklearn.ensemble as en
import sklearn.tree as tree
from sklearn.neighbors import NearestNeighbors

"""
# Characterize Under-represented Population
1. Estimate CATE $\tau(x)$ and ATE $\tau$ for experimental sample
2. Estimate proclivity-score $P(S=1 \mid X=x)$ using both experimental and observational sample
3. Calculate heterogeneity-score $h(x) = (\tau(x) - \tau)^2$, and representation-score $r(x) = log P(S=1 \mid X=x) - log P(S=0 \mid X=x)$
4. Combine them using predefined weights $\alpha$ to calculate objective value $z(x) = \alpha_0 r(x) - \alpha_1 h(x)$
5. Fit regularized decision-tree to tasselate the space into (non-overlapping) hyper-boxes -- $z \sim x$
    1. For regulatization, one can use minimum number of nodes in a leaf and/or maximum tree depth
6. Find the hyper-box(es) with the minimum average objective value
"""


def characterize(
    df,
    sample="S",
    treatment="T",
    outcome="Yobs",
    alpha=[5, 1],
    method="tree",
    characterization_depth=2,
    n_neighbors=100,
    p=np.inf,
    min_samples_leaf=50,
    smallest_k=1,
):

    df_exp = df.loc[(df[sample] == 1)]

    ### Estimate TE

    y1_est = en.GradientBoostingRegressor(max_depth=3).fit(
        df_exp.loc[(df_exp["T"] == 1)].drop(columns=["Yobs", "T", "S"]),
        df_exp.loc[(df_exp["T"] == 1), "Yobs"],
    )

    y0_est = en.GradientBoostingRegressor(max_depth=3).fit(
        df_exp.loc[(df_exp["T"] == 0)].drop(columns=["Yobs", "T", "S"]),
        df_exp.loc[(df_exp["T"] == 0), "Yobs"],
    )

    tau = y1_est.predict(df_exp.drop(columns=["Yobs", "T", "S"])) - y0_est.predict(
        df_exp.drop(columns=["Yobs", "T", "S"])
    )

    h_score = en.GradientBoostingRegressor().fit(
        df_exp.drop(columns=["Yobs", "T", "S"]), tau
    )

    ### Estimate Proclivity Score

    r_score = en.GradientBoostingClassifier(max_depth=2).fit(
        df.drop(columns=["Yobs", "T", "S"]), df["S"]
    )

    ### Estimate representiveness-score and heterogeneity-score

    r = r_score.predict_log_proba(df.drop(columns=["Yobs", "T", "S"]))
    result = pd.DataFrame(r[:, 1] - r[:, 0], columns=["r"], index=df.index)

    h = h_score.predict(df.drop(columns=["Yobs", "T", "S"]))
    h = (h - np.mean(h)) ** 2
    result["h"] = h

    ### Objective Score
    result["obj"] = alpha[0] * result["r"] - alpha[1] * result["h"]
    M = None
    dist_metric_learn = tree.DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf, max_leaf_nodes=8
    ).fit(df.drop(columns=["Yobs", "T", "S"]), result["obj"])
    M = dist_metric_learn.feature_importances_.reshape(
        -1,
    )
    X = df.drop(columns=["Yobs", "T", "S"]).values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", p=p).fit(
        X * M
    )
    distances, indices = nbrs.kneighbors(X)
    m = pd.DataFrame(
        list(map(lambda x: result.iloc[x]["obj"].mean(), indices)),
        index=df.index,
        columns=["avg.obj"],
    )

    indices = pd.DataFrame(
        indices,
        index=df.index,
        columns=["neighbor_%d" % (i) for i in range(indices.shape[1])],
    )
    matched_groups = pd.concat([m, indices], axis=1)

    df_result = df.copy(deep=True)
    df_result["flagged"] = 0

    mg_sorted = matched_groups.sort_values(by="avg.obj").drop(columns=["avg.obj"])
    flagged_idx = []
    for k in range(smallest_k):
        flagged_idx = flagged_idx + list(mg_sorted.iloc[k])
    df_result.iloc[flagged_idx, -1] = 1

    m_tree = tree.DecisionTreeRegressor(max_depth=characterization_depth).fit(
        df_result.drop(columns=["Yobs", "T", "S", "flagged"]), df_result["flagged"]
    )
    return matched_groups, result, tau, M, df_result, m_tree
