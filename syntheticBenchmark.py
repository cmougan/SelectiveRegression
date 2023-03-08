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
y = X + 10 * np.sin(X) + np.random.normal(0, 1, n)
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
coverage = 0.95
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
X_te_["intererval"] = interval
X_te_["y"] = y_te
X_te_["y_hat"] = clf.predict(X_te)
X_te_["selected"] = np.where(
    X_te_["intererval"] > np.quantile(interval, coverage), 1, 0
)
# %%
# Plot X, Y, model line and the uncertainty interval
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
mae = []
mse = []
cov_list = np.linspace(0.5, 0.99, 10)
for coverage in cov_list:
    X_te_["selected"] = np.where(
        X_te_["intererval"] > np.quantile(interval, coverage), 1, 0
    )
    aux = X_te_[X_te_["selected"] == 1].copy()
    print(X_te_[X_te_["selected"] == 1].shape)
    mae.append(mean_absolute_error(aux.y, aux.y_hat))
    mse.append(mean_squared_error(aux.y, aux.y_hat))


# %%
# Plot MAE and MSE
plt.figure(figsize=(10, 10))
plt.plot(cov_list, mae, label="MAE")
plt.plot(cov_list, mse, label="MSE")
plt.ylabel("Error")
plt.xlabel("Coverage")
plt.legend()
plt.show()

# %%
