# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from folktables import ACSDataSource, ACSIncome
from tools.SelectiveNet import SelectiveNetRegressor
from tools.utils import set_seed

np.random.seed(42)
set_seed(42)
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
ca_features = ca_features.sample(1_000)
# Split train, test and holdout
X_tr, X_te, y_tr, y_te = train_test_split(
    ca_features.drop(columns="label"), ca_features.label, test_size=0.5, random_state=0
)
X_val, X_te, y_val, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=0)
# %%
# %%
# Params
# Doubt uncertainty
uncertainty = 0.05
# Coverage quantile of the selective regression
# coverage = 0.9
# %%
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

# %%
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
body_dict = {
    "d_token": 192,
    "n_blocks": 3,
    "ffn_d_hidden": int(2 * 192),
    "attention_dropout": 0.4,
    "residual_dropout": 0,
}
# %%
# Benchmark against others
unc = []
unc_new = []
var = []
err = []
sln = []

#### store coverage

unc_cov = []
unc_cov_New = []
var_cov = []
err_cov = []
sln_cov = []


cov_list = np.linspace(0.7, 0.9, 5)
dict_selnet = {}
for coverage in cov_list:
    print("---------------")

    ### SelectiveNet
    sel = SelectiveNetRegressor(coverage=coverage, body_dict=body_dict)
    sel.fit(X_tr, y_tr, epochs=100, verbose=True)
    dict_selnet[coverage] = sel
    X_te_["y_hat_selnet"], unc_te_sel = sel.predict(X_te)
    _, unc_val_sel = sel.predict(X_val)
    # HERE WE REVERSE THE SIGN, AS WE WANT TO SELECT INSTANCES THAT ARE ABOVE A CERTAIN CONFIDENCE
    X_te_["selnet"] = np.where(
        unc_te_sel > np.quantile(unc_val_sel, 1 - coverage), 1, 0
    )
    aux2 = X_te_[X_te_["selnet"] == 1].copy()
    sln.append(mean_absolute_error(aux2.y, aux2.y_hat_selnet))
    print(
        "Selnet: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["selnet"] == 1].shape[0],
            mean_absolute_error(aux2.y, aux2.y_hat_selnet),
        )
    )
    sln_cov.append(X_te_[X_te_["selnet"] == 1].shape[0])

    # Gold Case - Best possible
    error = (y_te - clf.predict(X_te)) ** 2
    X_te_["error"] = np.where(error > np.quantile(error, 1 - coverage), 1, 0)
    aux0 = X_te_[X_te_["error"] == 0].copy()
    err.append(mean_absolute_error(aux0.y, aux0.y_hat))
    empirical_coverage = aux0.shape[0] / X_te_.shape[0]
    print(
        "Gold: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            empirical_coverage,
            X_te_[X_te_["error"] == 0].shape[0],
            mean_absolute_error(aux0.y, aux0.y_hat),
        )
    )
    err_cov.append(X_te_[X_te_["error"] == 1].shape[0])

    # Uncertainty with Doubt
    ## Interval is the validation one
    X_te_["uncertainty"] = np.where(
        X_te_["interval"] < np.quantile(interval_val, coverage), 1, 0
    )

    aux = X_te_[X_te_["uncertainty"] == 1].copy()
    unc.append(mean_absolute_error(aux.y, aux.y_hat))
    print(
        "Doubt: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["uncertainty"] == 1].shape[0],
            mean_absolute_error(aux.y, aux.y_hat),
        )
    )

    aux = X_te_[X_te_["uncertainty"] == 0].copy()

    empirical_coverage = aux.shape[0] / X_te_.shape[0]
    unc_cov.append(X_te_[X_te_["uncertainty"] == 1].shape[0])
    print(
        "Doubt: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            empirical_coverage,
            X_te_[X_te_["uncertainty"] == 0].shape[0],
            mean_absolute_error(aux.y, aux.y_hat),
            X_te_.shape[1],
        )
    )

    # New Dobut
    ## Interval is the validation one
    X_te_["uncertainty_New"] = np.where(
        X_te_["interval_New"] < np.quantile(interval_val_New, coverage), 1, 0
    )
    aux = X_te_[X_te_["uncertainty_New"] == 1].copy()
    unc_new.append(mean_absolute_error(aux.y, aux.y_hat))
    unc_cov_New.append(X_te_[X_te_["uncertainty_New"] == 1].shape[0])
    print(
        "New Doubt: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["uncertainty_New"] == 1].shape[0],
            mean_absolute_error(aux.y, aux.y_hat),
        )
    )

    # Variance
    ## Variance on the validation
    variance = (np.mean(clf.predict(X_val)) - clf.predict(X_val)) ** 2
    variance_test = (np.mean(clf.predict(X_te)) - clf.predict(X_te)) ** 2
    X_te_["variance"] = np.where(variance_test < np.quantile(variance, coverage), 1, 0)
    aux1 = X_te_[X_te_["variance"] == 1].copy()
    var.append(mean_absolute_error(aux1.y, aux1.y_hat))
    empirical_coverage = aux1.shape[0] / X_te_.shape[0]
    print(
        "Variance: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["variance"] == 1].shape[0],
            empirical_coverage,
            X_te_[X_te_["variance"] == 0].shape[0],
            mean_absolute_error(aux1.y, aux1.y_hat),
        )
    )
    var_cov.append(X_te_[X_te_["variance"] == 1].shape[0])

# %%
# Plot unc and var
plt.figure(figsize=(10, 10))
plt.plot(cov_list, unc, label="Doubt", marker="o")
plt.plot(cov_list, unc_new, label="New Doubt", marker="o")
plt.plot(cov_list, err, label="Gold Case (Best possible)", marker="s")
plt.plot(cov_list, var, label="Variance", marker="d")
plt.plot(cov_list, sln, label="SelNet", marker="^")
plt.ylabel("Error (Lower is better)")
plt.xlabel("Coverage")
plt.legend()
plt.savefig("images/folksBenchmark_performance.pdf", bbox_inches="tight")
# plt.show()


# %%
plt.figure(figsize=(10, 10))
plt.plot(cov_list, unc_cov, label="Doubt", marker="o")
plt.plot(cov_list, unc_cov_New, label="New Doubt", marker="o")
plt.plot(cov_list, err_cov, label="Gold Case (Best possible)", marker="s")
plt.plot(cov_list, var_cov, label="Variance", marker="d")
plt.plot(cov_list, sln_cov, label="SelNet", marker="^")
plt.ylabel("Error (Lower is better)")
plt.xlabel("Coverage")
plt.legend()
plt.savefig("images/folksBenchmark_coverage.pdf", bbox_inches="tight")
# plt.show()


# %%
