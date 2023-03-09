# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.datasets import make_regression


# Custom libraries
from tools.PlugIn import PlugInRule

# Doubt
from doubt import Boot

# %%
# Create data
n = 1000
X = np.random.normal(0, 1, n)
y = X + 10 * np.sin(1.2 * X) + np.random.normal(0, 1, n)
# Convert to dataframe
# df = pd.DataFrame(X, columns=["Var%d" % (i + 1) for i in range(X.shape[1])])
df = pd.DataFrame([X]).T
df.columns = ["Var1"]
df["label"] = y

# Plot X, Y and the model line
plt.figure(figsize=(10, 10))
plt.scatter(X, y, alpha=0.1)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
# %%
X_tr, X_te, y_tr, y_te = train_test_split(
    df.drop("label", axis=1),
    df["label"],
    test_size=0.5,
)
# %%
# Params
# Doubt uncertainty
uncertainty = 0.05
# Coverage quantile of the selective regression
# coverage = 0.9
# %%
# Train Model
clf = Boot(Lasso(alpha=0.01))
clf.fit(X_tr, y_tr)
y, unc = clf.predict(X_te, uncertainty=uncertainty)
interval = unc[:, 1] - unc[:, 0]
# %%
# Save data
X_te_ = X_te.copy()
# Add interval
X_te_["interval"] = interval
X_te_["y"] = y_te
X_te_["y_hat"] = clf.predict(X_te)

# %%
mae = []
mse = []
cov_list = np.linspace(0.001, 0.4, 10)
for coverage in cov_list:
    X_te_["selected"] = np.where(
        X_te_["interval"] > np.quantile(interval, 1 - coverage), 1, 0
    )
    aux = X_te_[X_te_["selected"] == 0].copy()
    print(
        "Coverage {:.2f} samples selected {}, which is {:.2f}%".format(
            coverage,
            X_te_[X_te_["selected"] == 0].shape[0],
            X_te_[X_te_["selected"] == 0].shape[0] / X_te_.shape[0],
        )
    )
    mae.append(mean_absolute_error(aux.y, aux.y_hat))
    mse.append(mean_squared_error(aux.y, aux.y_hat))

    # Plot X, Y, model line and the uncertainty interval
    if True == True:
        plt.figure(figsize=(10, 10))
        plt.scatter(X_te.Var1, y_te, alpha=0.1)
        plt.plot(X_te.Var1, clf.predict(X_te), color="red")
        plt.scatter(
            X_te_[X_te_["selected"] == 1].Var1,
            X_te_[X_te_["selected"] == 1].y,
            alpha=0.1,
            color="green",
        )
        plt.show()


# %%
# Plot MAE and MSE
plt.figure(figsize=(10, 10))
plt.plot(cov_list, mae, label="MAE")
plt.plot(cov_list, mse, label="MSE")
plt.ylabel("Error (Lower is better)")
plt.xlabel("Coverage")
plt.legend()
plt.show()

# %%
# Benchmark against others

unc = []
var = []
err = []
cov_list = np.linspace(0.001, 0.99, 10)
for coverage in cov_list:
    print("---------------")
    # Uncertainty with Doubt
    X_te_["uncertainty"] = np.where(
        X_te_["interval"] > np.quantile(interval, 1 - coverage), 1, 0
    )

    aux = X_te_[X_te_["uncertainty"] == 0].copy()
    print(
        "Doubt: Coverage {:.2f} samples selected {}".format(
            coverage, X_te_[X_te_["uncertainty"] == 0].shape[0]
        )
    )
    unc.append(mean_absolute_error(aux.y, aux.y_hat))

    # Gold Case - Best possible
    error = (y_te - clf.predict(X_te)) ** 2
    X_te_["error"] = np.where(error > np.quantile(error, 1 - coverage), 1, 0)
    aux0 = X_te_[X_te_["error"] == 0].copy()
    err.append(mean_absolute_error(aux0.y, aux0.y_hat))

    print(
        "Gold: Coverage {:.2f} samples selected {}".format(
            coverage, X_te_[X_te_["error"] == 0].shape[0]
        )
    )
    # Variance
    variance = clf.predict(X_te)
    X_te_["variance"] = np.where(variance > np.quantile(variance, 1 - coverage), 1, 0)
    aux1 = X_te_[X_te_["variance"] == 0].copy()
    var.append(mean_absolute_error(aux1.y, aux1.y_hat))
    print(
        "Variance: Coverage {:.2f} samples selected {}".format(
            coverage, X_te_[X_te_["variance"] == 0].shape[0]
        )
    )
    # Plot X, Y, model line and the uncertainty interval
    if True == True:
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
plt.show()


# %%

np.var(y - clf.predict(X_te))
# %%
# Calculate the deviation from the mean
np.mean(clf.predict(X_te)) - clf.predict(X_te)

# %%
np.var(clf.predict(X_te))
# %%
