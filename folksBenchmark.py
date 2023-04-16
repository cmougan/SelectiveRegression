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

# %%
# Load Data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
ca_features = ca_features.drop(columns="RAC1P")
ca_features["group"] = ca_group
# ca_features["label"] = ca_labels
# Rename SCHL as label
ca_features = ca_features.rename(columns={"SCHL": "label"})

ca_features
# Smaller dataset
ca_features = ca_features.sample(5000)
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
clf = Boot(XGBRegressor())
clf.fit(X_tr, y_tr)
_, unc_te = clf.predict(X_te, uncertainty=uncertainty)
_, unc_val = clf.predict(X_val, uncertainty=uncertainty)
interval_te = unc_te[:, 1] - unc_te[:, 0]
interval_val = unc_val[:, 1] - unc_val[:, 0]
# %%
# Save data
X_te_ = X_te.copy()
# Add interval
X_te_["interval"] = interval_te
X_te_["y"] = y_te
X_te_["y_hat"] = clf.predict(X_te)

X_val_ = X_val.copy()
# Add interval
X_val_["interval"] = interval_val
X_val_["y"] = y_val
X_val_["y_hat"] = clf.predict(X_val)

# %%
# Benchmark against others
unc = []
var = []
err = []
cov_list = np.linspace(0.1, 0.9, 10)
for coverage in cov_list:
    print("---------------")
    # Uncertainty with Doubt
    ## Interval is the validation one
    X_te_["uncertainty"] = np.where(
        X_te_["interval"] > np.quantile(interval_val, 1 - coverage), 1, 0
    )

    aux = X_te_[X_te_["uncertainty"] == 0].copy()
    unc.append(mean_absolute_error(aux.y, aux.y_hat))
    print(
        "Doubt: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["uncertainty"] == 0].shape[0],
            mean_absolute_error(aux.y, aux.y_hat),
        )
    )

    # Gold Case - Best possible
    error = (y_te - clf.predict(X_te)) ** 2
    X_te_["error"] = np.where(error > np.quantile(error, 1 - coverage), 1, 0)
    aux0 = X_te_[X_te_["error"] == 0].copy()
    err.append(mean_absolute_error(aux0.y, aux0.y_hat))

    print(
        "Gold: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["error"] == 0].shape[0],
            mean_absolute_error(aux0.y, aux0.y_hat),
        )
    )
    # Variance
    ## Variance on the validation
    variance = (np.mean(clf.predict(X_val)) - clf.predict(X_val)) ** 2
    variance_test = (np.mean(clf.predict(X_te)) - clf.predict(X_te)) ** 2
    X_te_["variance"] = np.where(
        variance_test > np.quantile(variance, 1 - coverage), 1, 0
    )
    aux1 = X_te_[X_te_["variance"] == 0].copy()
    var.append(mean_absolute_error(aux1.y, aux1.y_hat))
    print(
        "Variance: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_[X_te_["variance"] == 0].shape[0],
            mean_absolute_error(aux1.y, aux1.y_hat),
        )
    )
    # Plot X, Y, model line and the uncertainty interval
    if True == False:
        plt.figure(figsize=(10, 10))
        plt.scatter(X_te.Var1, y_te, alpha=0.1)
        plt.plot(X_te.Var1, clf.predict(X_te), color="red")
        plt.scatter(
            X_te_[X_te_["uncertainty"] == 1].Var1,
            X_te_[X_te_["uncertainty"] == 1].y,
            alpha=0.1,
            color="green",
            label="Doubt",
        )
        plt.scatter(
            X_te_[X_te_["error"] == 1].Var1,
            X_te_[X_te_["error"] == 1].y,
            marker="x",
            alpha=0.1,
            color="red",
            label="Gold Case (Best possible)",
        )
        ##plt.scatter(X_te_[X_te_["variance"] == 1].Var1,X_te_[X_te_["variance"] == 1].y,alpha=0.1,color="k",label="Variance",)
        plt.legend()
        plt.show()

# %%
# Plot unc and var
plt.figure(figsize=(10, 10))
plt.plot(cov_list, unc, label="Doubt")
plt.plot(cov_list, err, label="Gold Case (Best possible)")
plt.plot(cov_list, var, label="Variance")
plt.ylabel("Error (Lower is better)")
plt.xlabel("Coverage")
plt.legend()
plt.savefig("images/folksBenchmark.pdf", bbox_inches="tight")
plt.show()


# %%
