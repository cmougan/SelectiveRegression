# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from tools.methods import SCross, PlugIn

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from tools.utils import set_seed
from lightgbm import LGBMRegressor

# Benchmark
from pmlb import regression_dataset_names, fetch_data
from mapie.regression import MapieRegressor


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


def experiment(
    X,
    y,
    dataset,
    metas,
    reg_base,
    uncertainty=0.05,
    seed=42,
    coverages=[1, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5],
    n_boot=None,
):
    if os.path.exists("results/") == False:
        os.mkdir("results")
    if os.path.exists("results/{}".format(dataset)) == False:
        os.mkdir("results/{}".format(dataset))
    # Convert to dataframe
    X = pd.DataFrame(X.copy()).reset_index(drop=True)
    X.columns = ["Var" + str(i) for i in X.columns]
    y = pd.Series(y.copy()).values
    # Split train, test and holdout
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_te, y_te, test_size=0.5, random_state=seed
    )
    scaler = MinMaxScaler()
    scaler.fit(X=X_tr)
    X_tr = scaler.transform(X=X_tr)
    X_te = scaler.transform(X=X_te)
    X_val = scaler.transform(X=X_val)
    scaler = MinMaxScaler()
    scaler.fit(y_tr.reshape(-1, 1))
    y_tr = scaler.transform(y_tr.reshape(-1, 1)).flatten()
    y_val = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_te = scaler.transform(y_te.reshape(-1, 1)).flatten()
    n_features = len(X.columns)
    train_size = X_tr.shape[0]
    reg = Boot(reg_base, random_seed=42)
    reg.fit(X_tr, y_tr, n_boots=n_boot)
    print(len(reg._models))
    results = pd.DataFrame()
    for meta in metas:
        if meta == "doubt":
            _, unc_te = reg.predict(X_te, uncertainty=uncertainty, n_boots=n_boot)
            _, unc_val = reg.predict(X_val, uncertainty=uncertainty, n_boots=n_boot)
            interval_te = unc_te[:, 1] - unc_te[:, 0]
            interval_val = unc_val[:, 1] - unc_val[:, 0]
            y_hat = reg.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val, coverage)
                sel_te = np.where(interval_te <= tau, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "doubtNew":
            _, unc_te_new = reg.predict(X_te, return_all=True, n_boots=n_boot)
            _, unc_val_new = reg.predict(X_val, return_all=True, n_boots=n_boot)
            interval_te_new = np.var(unc_te_new.T, axis=0)
            interval_val_new = np.var(unc_val_new.T, axis=0)
            y_hat = reg.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val_new, coverage)
                sel_te = np.where(interval_te_new <= tau, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "gold":
            error = (
                y_te - reg.predict(X_te, n_boots=n_boot)
            ) ** 2  # error on the test set
            y_hat = reg.predict(X_te, n_boots=n_boot)
            res = pd.DataFrame()
            for coverage in coverages:
                tau_gold = np.quantile(
                    error, coverage
                )  # tau on the test set - we accept instances below the error tau_gold
                sel_te = np.where(error <= tau_gold, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "plugin":
            rplug = PlugIn(reg_base)
            rplug.fit(X_tr, y_tr)
            y_hat = rplug.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rplug.select(X_te, X_val, coverage)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "pluginVar":
            rplug = PlugIn(reg_base)
            rplug.fit(X_tr, y_tr)
            y_hat = rplug.predict(X_te)
            y_hat_val = rplug.predict(X_val)
            var_val = (y_hat_val - np.mean(y_hat_val)) ** 2
            var_te = (y_hat - np.mean(y_hat)) ** 2
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(var_val, coverage)
                sel_te = np.where(var_te <= tau, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "mapieBase":
            mapie = MapieRegressor(reg_base, cv=5, random_state=seed)
            mapie.fit(X_tr, y_tr)
            y_hat, unc_te_map = mapie.predict(X_te, alpha=[uncertainty])
            y_hat_val, unc_val_map = mapie.predict(X_te, alpha=[uncertainty])
            interval_te = unc_te_map[:, 1, 0] - unc_te_map[:, 0, 0]
            interval_val = unc_val_map[:, 1, 0] - unc_val_map[:, 0, 0]
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val, coverage)
                sel_te = np.where(interval_te <= tau, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "scross":
            rcross = SCross(reg_base)
            rcross.fit(X_tr, y_tr)
            y_hat = rcross.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rcross.select(X_te, X_val, coverage)
                if coverage != 1:
                    tmp = get_metrics_test(y_te, y_hat, sel_te)
                else:
                    tmp = get_metrics_test(y_te, y_hat, np.ones(y_te.shape))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
        res["meta"] = meta
        res["model"] = reg_base.__class__.__name__
        res["dataset"] = dataset
        res["features"] = n_features
        res["trainingsize"] = train_size
        res["nboots"] = len(reg._models)
        results = pd.concat([results, res], axis=0)
    return results


def main(dataset, regressors, metas, nj=1, seed=42, nboots=None):
    np.random.seed(seed)
    set_seed(seed)
    try:
        X, y = fetch_data(dataset, return_X_y=True)
    except:
        raise FileNotFoundError("The dataset could not be retrieved. Please check.")
    for reg_string in regressors:
        if reg_string == "lasso":
            reg_base = Lasso(random_state=seed)
        elif reg_string == "lr":
            reg_base = LinearRegression(n_jobs=nj)
        elif reg_string == "xgb":
            reg_base = XGBRegressor(random_state=seed, n_jobs=nj)
        elif reg_string == "lgbm":
            reg_base = LGBMRegressor(random_state=seed, n_jobs=nj)
        elif reg_string == "dt":
            reg_base = DecisionTreeRegressor(random_state=seed)
        tmp = experiment(X, y, dataset, metas, reg_base, seed=seed, n_boot=nboots)
        tmp["seed"] = seed
        if nboots is None:
            boot_str = ""
        else:
            boot_str = "_BOOTS_{}".format(nboots)
        tmp.to_csv(
            "results/{}/ALL_RESULTS_{}_{}_{}_SEED{}{}.csv".format(
                dataset, dataset, reg_string, "-".join(metas), seed, boot_str
            ),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--reg", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(regression_dataset_names))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boots", type=int, default=None)

    args = parser.parse_args()
    regressors = args.reg
    metas = args.meta
    start = max(0, args.start)
    end = min(len(regression_dataset_names), args.end)
    # metas = ["doubt"]
    list_datasets = pd.read_csv("penn_ML_datasets.csv")["Dataset"].tolist()
    for dataset in tqdm(list_datasets[start:end]):
        print(dataset)
        main(
            dataset, regressors, metas, nj=args.jobs, seed=args.seed, nboots=args.boots
        )
