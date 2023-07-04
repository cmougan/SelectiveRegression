# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from tools.utils import set_seed

# Benchmark
from pmlb import regression_dataset_names, fetch_data

np.random.seed(42)
set_seed(42)
# %%
# Params
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
uncertainty = 0.05
l1table = []
# %%
for dataset in tqdm(regression_dataset_names[3:5]):
    try:
        X, y = fetch_data(dataset, return_X_y=True)
    except:
        continue

    # Convert to dataframe
    X = pd.DataFrame(X)
    X.columns = ["Var" + str(i) for i in X.columns]
    y = pd.Series(y)

    # Split train, test and holdout
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    X_val, X_te, y_val, y_te = train_test_split(
        X_te, y_te, test_size=0.5, random_state=0
    )

    # Train Model
    for model in [
        LinearRegression(),
        Lasso(),
        DecisionTreeRegressor(),
        XGBRegressor(n_jobs=4),
    ]:
        print(dataset, "--", model.__class__.__name__)
        clf = Boot(model, random_seed=42)
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

        cov_list = np.linspace(0.6, 0.95, 10)
        dict_selnet = {}
        for coverage in cov_list:
            # Gold Case - Best possible
            error = (y_te - clf.predict(X_te)) ** 2
            X_te_["error"] = np.where(error > np.quantile(error, 1 - coverage), 0, 1)
            aux0 = X_te_[X_te_["error"] == 0].copy()
            err.append(
                mean_absolute_error(aux0.y, aux0.y_hat)
            )  # TODO Do we want to use MAE or MSE?
            empirical_coverage = aux0.shape[0] / X_te_.shape[0]
            err_cov.append(X_te_[X_te_["error"] == 1].shape[0])

            # Uncertainty with Doubt
            ## Interval is the validation one
            X_te_["uncertainty"] = np.where(
                X_te_["interval"] < np.quantile(interval_val, coverage), 1, 0
            )
            aux = X_te_[X_te_["uncertainty"] == 1].copy()
            unc.append(mean_absolute_error(aux.y, aux.y_hat))

            aux = X_te_[X_te_["uncertainty"] == 0].copy()
            empirical_coverage = aux.shape[0] / X_te_.shape[0]
            unc_cov.append(X_te_[X_te_["uncertainty"] == 1].shape[0])

            # New Doubt
            ## Interval is the validation one
            X_te_["uncertainty_New"] = np.where(
                X_te_["interval_New"] < np.quantile(interval_val_New, coverage), 1, 0
            )
            aux = X_te_[X_te_["uncertainty_New"] == 1].copy()
            unc_new.append(mean_absolute_error(aux.y, aux.y_hat))
            unc_cov_New.append(X_te_[X_te_["uncertainty_New"] == 1].shape[0])

            # Variance
            ## Variance on the validation
            variance = (np.mean(clf.predict(X_val)) - clf.predict(X_val)) ** 2
            variance_test = (np.mean(clf.predict(X_te)) - clf.predict(X_te)) ** 2
            X_te_["variance"] = np.where(
                variance_test < np.quantile(variance, coverage), 1, 0
            )
            aux1 = X_te_[X_te_["variance"] == 1].copy()
            var.append(mean_absolute_error(aux1.y, aux1.y_hat))
            empirical_coverage = aux1.shape[0] / X_te_.shape[0]

            var_cov.append(X_te_[X_te_["variance"] == 1].shape[0])

            # TODO add mapie

        l1table.append(
            [
                dataset,
                X.shape,
                model.__class__.__name__,
                mean_absolute_error(unc, err),
                mean_absolute_error(unc_new, err),
                mean_absolute_error(var, err),
            ]
        )
        # %%
# Save results as dataframe
res = pd.DataFrame(
    l1table, columns=["Dataset", "Shape", "Model", "Doubt", "New Doubt", "Variance"]
)

# %%
# Group by  model and calculate mean
res.groupby(["Model"]).mean()
# %%
