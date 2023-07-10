# %%
# Import candidate models
from doubt import Boot
from sklearn.linear_model import LinearRegression

from folktables import ACSDataSource, ACSIncome
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split

# Import external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
import warnings
import seaborn as sns
import pdb

sns.set_theme(style="whitegrid")
import shap

# Import internal classes

import warnings

warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders.m_estimate import MEstimateEncoder


# %%
def get_metrics_test(true, y_hat, selected):
    if np.sum(selected) > 0:
        coverage = len(selected[selected == 1]) / len(selected)
        mae = mean_absolute_error(true[selected == 1], y_hat[selected == 1])
        mse = mean_squared_error(true[selected == 1], y_hat[selected == 1])
    else:
        coverage = -1
        mae = -1
        mse = -1
    tmp = pd.DataFrame([[coverage, mae, mse]], columns=["coverage", "MAE", "MSE"])
    return tmp


def explain(xgb: bool = True):
    """
    Provide a SHAP explanation by fitting MEstimate and GBDT
    """
    if xgb:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", GradientBoostingRegressor())]
        )
        pipe.fit(X, y)
        explainer = shap.Explainer(pipe[1])
        shap_values = explainer(pipe[:-1].transform(X))
        shap.plots.beeswarm(shap_values)
        return pd.DataFrame(np.abs(shap_values.values), columns=X.columns).sum()
    else:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", LogisticRegression())]
        )
        pipe.fit(X, y)
        coefficients = pd.concat(
            [pd.DataFrame(X_tr.columns), pd.DataFrame(np.transpose(pipe[1].coef_))],
            axis=1,
        )
        coefficients.columns = ["feat", "val"]

        return coefficients.sort_values(by="val", ascending=False)


# %%
# Load Data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
# features, label, group = ACSEmployment.df_to_numpy(acs_data)
ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
ca_features = ca_features.drop(columns="RAC1P")
ca_features["group"] = ca_group
# ca_features["label"] = ca_labels
# Rename SCHL as label
ca_features = ca_features.rename(columns={"SCHL": "label"})
ca_features
# Smaller dataset
ca_features = ca_features.sample(4_000)
# Split train, test, val and holdout set in 25-25-25-25
X = ca_features.drop(columns="label")

# Add random noise to X
X["random"] = np.random.normal(0, 1, X.shape[0])
y = ca_features.label
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X1, y1, test_size=0.5, random_state=0)
X_hold, X_te, y_hold, y_te = train_test_split(X2, y2, test_size=0.5, random_state=0)


# %%
# Fit model
reg = Boot(LinearRegression(), random_seed=42)
reg.fit(X_tr, y_tr)
# Make predictions
_, unc_te_new = reg.predict(X_te, return_all=True)
_, unc_val_new = reg.predict(X_val, return_all=True)
_, unc_hold_new = reg.predict(X_hold, return_all=True)
interval_te_new = np.var(unc_te_new.T, axis=0)
interval_val_new = np.var(unc_val_new.T, axis=0)
interval_hold_new = np.var(unc_hold_new.T, axis=0)
y_hat = reg.predict(X_te)
res = pd.DataFrame()
coverage = 0.50
tau = np.quantile(interval_val_new, coverage)
sel_hold = np.where(interval_hold_new <= tau, 1, 0)
sel_te = np.where(interval_te_new <= tau, 1, 0)
tmp = get_metrics_test(y_te, y_hat, sel_te)
tmp["target_coverage"] = coverage
res = pd.concat([res, tmp], axis=0)
# %%
# Basic Experiment
# audit = LogisticRegression()
audit = LogisticRegression(penalty="l1", solver="liblinear")
audit.fit(X_hold, sel_hold)
# Lets evaluate on val -- Funny but it does not seem so too bad
print("AUC", roc_auc_score(sel_te, audit.predict_proba(X_te)[:, 1]))
print("F1", f1_score(sel_te, audit.predict(X_te)))
print("Precision", precision_score(sel_te, audit.predict(X_te)))
print("Recall", recall_score(sel_te, audit.predict(X_te)))
# %%
# Explain Auditor
explainer = shap.Explainer(audit, X_tr)
shap_values = explainer(X_te)
shap.plots.bar(shap_values)
# %%
# Shifted feature experiment
X_te2 = X_te.copy()
X_te2["random"] = X_te2["random"] + np.random.normal(5, 1, X_te2.shape[0])
X_te2["COW"] = X_te2["COW"] + np.random.normal(5, 1, X_te2.shape[0])
X_hold2 = X_hold.copy()
X_hold2["random"] = X_hold2["random"] + np.random.normal(5, 1, X_hold2.shape[0])
X_hold2["COW"] = X_hold2["COW"] + np.random.normal(5, 1, X_hold2.shape[0])

audit.fit(X_hold2, sel_hold)
# Lets evaluate on val -- Funny but it does not seem so too bad
print(roc_auc_score(sel_te, audit.predict_proba(X_te2)[:, 1]))
print(f1_score(sel_te, audit.predict(X_te2)))
print(precision_score(sel_te, audit.predict(X_te2)))
print(recall_score(sel_te, audit.predict(X_te2)))

explainer = shap.Explainer(audit, X_tr)
shap_values = explainer(X_te2)
shap.plots.bar(shap_values)
# %%
shap.plots.waterfall(shap_values[0])
# %%
shap.plots.beeswarm(shap_values)
# %%
print("AUC", roc_auc_score(sel_te, audit.predict_proba(X_te)[:, 1]))
print("F1", f1_score(sel_te, audit.predict(X_te)))
print("Precision", precision_score(sel_te, audit.predict(X_te)))
print("Recall", recall_score(sel_te, audit.predict(X_te)))
print("---------Shift")
print("AUC", roc_auc_score(sel_te, audit.predict_proba(X_te2)[:, 1]))
print("f1", f1_score(sel_te, audit.predict(X_te2)))
print("prec", precision_score(sel_te, audit.predict(X_te2)))
print("rec", recall_score(sel_te, audit.predict(X_te2)))

# %%
