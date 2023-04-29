# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# Sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from folktables import ACSDataSource, ACSIncome

np.random.seed(42)

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
# %%
# Experiment params
# Params
# Doubt uncertainty
uncertainty = 0.05
# Coverage quantile of the selective regression
coverage = 0.1
boots = 100
# Benchmark against others
unc = []
unc_new = []

#### store coverage
unc_cov = []
unc_cov_New = []
# %%
# Boostrap Experiment
for n in range(boots):
    ca_features_ = ca_features.sample(1_000)
    # Split train, test and holdout
    X_tr, X_te, y_tr, y_te = train_test_split(
        ca_features_.drop(columns="label"),
        ca_features_.label,
        test_size=0.5,
        random_state=0,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_te, y_te, test_size=0.5, random_state=0
    )

    # Train Model
    clf = Boot(XGBRegressor(n_jobs=4), random_seed=42)
    clf.fit(X_tr, y_tr)
    _, unc_te = clf.predict(X_te, uncertainty=uncertainty)
    _, unc_val = clf.predict(X_val, uncertainty=uncertainty)
    interval_te = unc_te[:, 1] - unc_te[:, 0]
    interval_val = unc_val[:, 1] - unc_val[:, 0]

    # New uncertainty on Test and Val
    _, unc_te_New = clf.predict(X_te, return_all=True)
    interval_te_New = np.var(unc_te_New, axis=1)
    _, unc_val_New = clf.predict(X_val, return_all=True)
    interval_val_New = np.var(unc_val_New, axis=1)

    # Save data
    X_te_ = X_te.copy()
    # Add interval
    X_te_["interval"] = interval_te
    X_te_["interval_New"] = interval_te_New
    X_te_["y"] = y_te
    X_te_["y_hat"] = clf.predict(X_te)

    X_val_ = X_val.copy()
    # Add interval
    X_val_["interval"] = interval_val
    X_val_["interval_New"] = interval_val_New
    X_val_["y"] = y_val
    X_val_["y_hat"] = clf.predict(X_val)

    # Uncertainty with Doubt
    ## Interval is the validation one
    X_te_["uncertainty"] = np.where(
        X_te_["interval"] < np.quantile(interval_val, coverage), 1, 0
    )
    # Error
    aux = X_te_[X_te_["uncertainty"] == 1].copy()
    unc.append(mean_absolute_error(aux.y, aux.y_hat))

    # Coverage
    aux = X_te_[X_te_["uncertainty"] == 0].copy()
    empirical_coverage = aux.shape[0] / X_te_.shape[0]
    unc_cov.append(X_te_[X_te_["uncertainty"] == 1].shape[0])

    # New Dobut
    ## Interval is the validation one
    X_te_["uncertainty_New"] = np.where(
        X_te_["interval_New"] < np.quantile(interval_val_New, coverage), 1, 0
    )
    aux = X_te_[X_te_["uncertainty_New"] == 1].copy()
    unc_new.append(mean_absolute_error(aux.y, aux.y_hat))
    unc_cov_New.append(X_te_[X_te_["uncertainty_New"] == 1].shape[0])

# %%
# Pvalue
## TODO WHY PVALUE IS DIVIDED BY TWO
# Now the test is two sided, do we need just one
pval = ttest_rel(unc, unc_new)[1]
pval_cov = ttest_rel(unc_cov, unc_cov_New)[1]


# Plot

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.kdeplot(unc, alpha=0.5, fill=True, label="Doubt")
sns.kdeplot(unc_new, alpha=0.5, fill=True, label="New Doubt")
plt.title("Error Pvalue {:.3f}".format(pval))
ax.set_xlabel("MAE")
ax.set_ylabel("Frequency")
ax.legend()
plt.show()
# %%
# Plot
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.kdeplot(unc_cov, alpha=0.5, fill=True, label="Doubt")
sns.kdeplot(unc_cov_New, alpha=0.5, fill=True, label="New Doubt")
plt.title("Coverage Pvalue {:.2f}".format(pval_cov))
ax.set_xlabel("Number of samples")
ax.set_ylabel("Frequency")
ax.legend()
plt.show()

# %%
