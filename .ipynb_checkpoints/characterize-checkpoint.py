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

### Estimate TE

def characterize( df, sample="S", treatment="T", outcome="Yobs", alpha = [5, 1], max_leaf_nodes=8, min_samples_leaf=50):
    df_exp = df.loc[(df[sample] == 1)]

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

    r = r_score.predict_log_proba(df_exp.drop(columns=["Yobs", "T", "S"]))
    result = pd.DataFrame(r[:, 1] - r[:, 0], columns=["r"], index=df_exp.index)

    h = h_score.predict(df_exp.drop(columns=["Yobs", "T", "S"]))
    h = (h - np.mean(h)) ** 2
    result["h"] = h

    ### Objective Score
    result["obj"] = alpha[0] * result["r"] - alpha[1] * result["h"]

    ### Tessellate

    m = tree.DecisionTreeRegressor(min_samples_leaf = min_samples_leaf, max_leaf_nodes = max_leaf_nodes).fit(
        df_exp.drop(columns=["Yobs", "T", "S"]), result["obj"]
    )

    return m, result, tau 