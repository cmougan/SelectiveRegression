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

# Specific Libraries
from xgboost import XGBRegressor
from folktables import ACSDataSource, ACSIncome
from tqdm import tqdm

# %%
# Load and split data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
ca_features = ca_features.drop(columns="RAC1P")
ca_features["group"] = ca_group
# ca_features["label"] = ca_labels
# Rename SCHL as label
ca_features = ca_features.rename(columns={"SCHL": "label"})
# Smaller dataset
ca_features = ca_features.sample(5000)

X_group = ca_features[ca_features["group"] == 6]
ca_features = ca_features[ca_features["group"] != 6]

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

# %%
# Train Model
clf = Boot(XGBRegressor())
clf.fit(X_tr, y_tr)

# %%
# Coverage quantile of the selective regression
coverage = 0.1
ratio_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
unc = []
var = []
err = []
# Test data has R ratio of group 6
total_size = min(X_te.shape[0], X_group.shape[0])
X_te["label"] = y_te
for R in tqdm(ratio_params):
    print(R)
    # Sample and concat
    X_group_ = X_group.sample(n=int(R * total_size))
    X_te_ = X_te.sample(n=int((1 - R) * total_size))

    X_te_group = pd.concat([X_group_, X_te_])
    y_te_group = X_te_group["label"]
    X_te_group = X_te_group.drop(columns="label")

    # Predictions
    _, unc_te = clf.predict(X_te_group, uncertainty=uncertainty)
    _, unc_val = clf.predict(X_val, uncertainty=uncertainty)
    interval_te = unc_te[:, 1] - unc_te[:, 0]
    interval_val = unc_val[:, 1] - unc_val[:, 0]
    # Save data
    X_te_group_ = X_te_group.copy()

    # Add interval
    X_te_group_["interval"] = interval_te
    X_te_group_["y"] = y_te_group
    X_te_group_["y_hat"] = clf.predict(X_te_group)

    X_val_ = X_val.copy()
    # Add interval
    X_val_["interval"] = interval_val
    X_val_["y"] = y_val
    X_val_["y_hat"] = clf.predict(X_val)

    print("---------------")
    # Uncertainty with Doubt
    ## Interval is the validation one
    X_te_group_["uncertainty"] = np.where(
        X_te_group_["interval"] > np.quantile(interval_val, 1 - coverage), 1, 0
    )

    aux = X_te_group_[X_te_group_["uncertainty"] == 0].copy()
    unc.append(mean_absolute_error(aux.y, aux.y_hat))
    print(
        "Doubt: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_group_[X_te_group_["uncertainty"] == 0].shape[0],
            mean_absolute_error(aux.y, aux.y_hat),
        )
    )

    # Gold Case - Best possible
    error = (y_te_group - clf.predict(X_te_group)) ** 2
    X_te_group_["error"] = np.where(error > np.quantile(error, 1 - coverage), 1, 0)
    aux0 = X_te_group_[X_te_group_["error"] == 0].copy()
    err.append(mean_absolute_error(aux0.y, aux0.y_hat))

    print(
        "Gold: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_group_[X_te_group_["error"] == 0].shape[0],
            mean_absolute_error(aux0.y, aux0.y_hat),
        )
    )
    # Variance
    ## Variance on the validation
    variance = (np.mean(clf.predict(X_val)) - clf.predict(X_val)) ** 2
    variance_test = (np.mean(clf.predict(X_te_group)) - clf.predict(X_te_group)) ** 2
    X_te_group_["variance"] = np.where(
        variance_test > np.quantile(variance, 1 - coverage), 1, 0
    )
    aux1 = X_te_group_[X_te_group_["variance"] == 0].copy()
    var.append(mean_absolute_error(aux1.y, aux1.y_hat))
    print(
        "Variance: Coverage {:.2f} samples selected {} with error {:.2f}".format(
            coverage,
            X_te_group_[X_te_group_["variance"] == 0].shape[0],
            mean_absolute_error(aux1.y, aux1.y_hat),
        )
    )


# %%
# Plot unc and var
plt.figure(figsize=(10, 10))
plt.plot(ratio_params, unc, label="Doubt")
plt.plot(ratio_params, err, label="Gold Case (Best possible)")
plt.plot(ratio_params, var, label="Variance")
plt.ylabel("Error (Lower is better)")
plt.xlabel("Group Ratio (R)")
plt.legend()
plt.savefig("images/group.pdf", bbox_inches="tight")
plt.show()

# %%
