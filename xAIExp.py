# %%
# Import candidate models
from doubt import Boot
from sklearn.linear_model import LinearRegression

from folktables import ACSDataSource, ACSIncome
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, wasserstein_distance

# Import external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tools.methods import PlugIn

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
# Font size
rcParams["font.size"] = 18
rcParams["axes.titlesize"] = 22
rcParams["axes.labelsize"] = 18
rcParams["legend.fontsize"] = 16
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16

plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
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
from sklearn.preprocessing import StandardScaler
from category_encoders.m_estimate import MEstimateEncoder

# Ser seed
np.random.seed(4)


# %%
# Utils
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
# https://www.kaggle.com/code/lucabasa/house-price-cleaning-without-dropping-features/output?select=trainclean.csv
df = pd.read_csv("data/trainclean.csv")
df = df.drop(columns="Id")
# %%
# Quick feature importance
# X = df.drop(columns="SalePrice")._get_numeric_data()
# y = df.SalePrice
# m = XGBRegressor()
# m.fit(X, y)
# pd.DataFrame(m.feature_importances_, index=X.columns).sort_values(    by=0, ascending=False)

# %%
df["Random"] = np.random.normal(0, 10, df.shape[0])

df = df[
    [
        "OverallQual",
        "GarageCars",
        "BsmtQual",
        "KitchenQual",
        "KitchenAbvGr",
        "GrLivArea",
        "CentralAir",
        "SalePrice",
        "Random",
    ]
]

# %%
# Smaller dataset

X = df.drop(columns="SalePrice")
# Add random noise to X

# Standardize
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# Train test split
y = df.SalePrice
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X1, y1, test_size=0.5, random_state=0)
X_hold, X_te, y_hold, y_te = train_test_split(X2, y2, test_size=0.5, random_state=0)

# %%
# Fit model
reg = Boot(XGBRegressor(), random_seed=42)
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
coverage = 0.80
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
pd.DataFrame(audit.coef_, columns=X_tr.columns)
# %%
# Lets evaluate on val -- Funny but it does not seem so too bad
print("AUC", roc_auc_score(sel_te, audit.predict_proba(X_te)[:, 1]))
print("F1", f1_score(sel_te, audit.predict(X_te)))
print("Precision", precision_score(sel_te, audit.predict(X_te)))
print("Recall", recall_score(sel_te, audit.predict(X_te)))
# %%
# We dont predict if sel_te==1
inst = X_te[sel_te == 1]

# %%
# Explain Auditor
explainer = shap.explainers.Linear(
    audit, X_tr, feature_perturbation="correlation_dependent"
)
shap_values = explainer(inst)
shap1 = pd.DataFrame(shap_values.values, columns=X_tr.columns)
# Local explanation

# Local explanation
plt.figure(figsize=(10, 5))
plt.title("Distribution of explanations", fontsize=18)
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.xlim(-8, 4)
plt.savefig("images/local_shap.pdf", bbox_inches="tight")
plt.close()
# %%
# Shift
inst_shift = inst.copy()
inst_shift["Random"] = inst_shift["Random"] + np.random.normal(
    5, 1, inst_shift.shape[0]
)
inst_shift["GrLivArea"] = inst_shift["GrLivArea"] + np.random.normal(
    5, 1, inst_shift.shape[0]
)

explainer = shap.explainers.Linear(
    audit, X_tr, feature_perturbation="correlation_dependent"
)
shap_values = explainer(inst_shift)
shap2 = pd.DataFrame(shap_values.values, columns=X_tr.columns)
# Local explanation
plt.figure(figsize=(10, 5))
plt.title(
    "Distribution of explanations with a Shift in GrLivArea and Random", fontsize=18
)
shap.plots.beeswarm(shap_values, show=False)
plt.xlim(-8, 4)
plt.tight_layout()
plt.savefig("images/local_shap_shift.pdf", bbox_inches="tight")
plt.close()
# %%
# Shift of all features
wass = []
for col in X_tr.columns:
    statistic = []
    for i in range(10):
        inst_shift = inst.copy()
        inst_shift[col] = inst_shift[col] + np.random.normal(5, 1, inst_shift.shape[0])

        explainer = shap.explainers.Linear(
            audit, X_tr, feature_perturbation="interventional"
        )
        shap_values = explainer(inst_shift)
        shap2 = pd.DataFrame(shap_values.values, columns=X_tr.columns)
        statistic.append(wasserstein_distance(shap1[col], shap2[col]))

    wass.append([col, np.mean(statistic), np.std(statistic)])

# Results
wass = pd.DataFrame(wass, columns=["feat", "wass", "ci"]).sort_values(
    by="wass", ascending=False
)
# %%
wass
# %%
# Loop over estimator and auditor
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

estimators = [
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    LinearRegression(),
    RandomForestRegressor(),
]
auditors = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
]
resultados = []
for estimator in estimators:
    for auditor in auditors:
        # Fit estimator
        reg = Boot(estimator)
        reg.fit(X_tr, y_tr)

        # Engineer predictions
        _, unc_te_new = reg.predict(X_te, return_all=True)
        _, unc_val_new = reg.predict(X_val, return_all=True)
        _, unc_hold_new = reg.predict(X_hold, return_all=True)
        interval_te_new = np.var(unc_te_new.T, axis=0)
        interval_val_new = np.var(unc_val_new.T, axis=0)
        interval_hold_new = np.var(unc_hold_new.T, axis=0)
        y_hat = reg.predict(X_te)
        coverage = 0.80
        tau = np.quantile(interval_val_new, coverage)
        sel_hold = np.where(interval_hold_new <= tau, 1, 0)
        sel_te = np.where(interval_te_new <= tau, 1, 0)
        tmp = get_metrics_test(y_te, y_hat, sel_te)
        tmp["target_coverage"] = coverage

        # Fit Auditor
        auditor.fit(X_hold, sel_hold)
        auc = roc_auc_score(sel_te, audit.predict_proba(X_te)[:, 1])
        resultados.append(
            [estimator.__class__.__name__, auditor.__class__.__name__, auc]
        )
# %%
# Pivot table
pd.DataFrame(resultados, columns=["estimator", "auditor", "auc"]).pivot(
    index="estimator", columns="auditor", values="auc"
).T.round(3).to_csv("estimator_auditor.csv")

# %%
