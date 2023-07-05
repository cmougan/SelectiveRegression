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
# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from tools.utils import set_seed
from lightgbm import LGBMRegressor
# Benchmark
from pmlb import regression_dataset_names, fetch_data





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

def experiment(X, y, dataset, metas, class_string, n_jobs_class=1, uncertainty=0.05,
               seed=42, coverages=[.99, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5],
               n_boot=None):
    if class_string == "lasso":
        reg_base = Lasso(random_state=seed)
    elif class_string == "lr":
        reg_base = LinearRegression(n_jobs=n_jobs_class)
    elif class_string == "xgb":
        reg_base = XGBRegressor(random_state=seed, n_jobs=n_jobs_class)
    elif class_string == "lgbm":
        reg_base = LGBMRegressor(random_state=seed, n_jobs=n_jobs_class)
    elif class_string == "dt":
        reg_base = DecisionTreeRegressor(random_state=seed)
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
    n_features = len(X.columns)
    train_size = X_tr.shape[0]
    reg = Boot(reg_base, random_seed=42)
    reg.fit(X_tr, y_tr, n_boots=n_boot)
    print(len(reg._models))
    results = pd.DataFrame()
    for meta in metas:
        if meta == "doubt":
            _, unc_te = reg.predict(X_te, uncertainty=uncertainty)
            _, unc_val = reg.predict(X_val, uncertainty=uncertainty)
            interval_te = unc_te[:, 1] - unc_te[:, 0]
            interval_val = unc_val[:, 1] - unc_val[:, 0]
            y_hat = reg.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val, coverage)
                sel_te = np.where(
                        interval_te <= tau, 1, 0
                    )
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "doubtNew":
            _, unc_te_new = reg.predict(X_te, return_all=True)
            _, unc_val_new = reg.predict(X_val, return_all=True)
            interval_te_new = np.var(unc_te_new.T, axis=0)
            interval_val_new = np.var(unc_val_new.T, axis=0)
            y_hat = reg.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val_new, coverage)
                sel_te = np.where(
                        interval_te_new <= tau, 1, 0
                    )
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "gold":
            error = (y_te - reg.predict(X_te)) ** 2  # error on the test set
            y_hat = reg.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                tau_gold = np.quantile(error,
                                       coverage)  # tau on the test set - we accept instances below the error tau_gold
                sel_te = np.where(error <= tau_gold, 1, 0)
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "plugin":
            rplug = PlugIn(reg_base)
            rplug.fit(X_tr, y_tr)
            y_hat = rplug.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rplug.select(X_te, X_val, coverage)
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "pluginVar":
            y_hat_val = reg.predict(X_val)
            y_hat = reg.predict(X_te)
            var_val = (y_hat_val - np.mean(y_hat_val))**2
            var_te = (y_hat - np.mean(y_hat)) ** 2
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(var_val, coverage)
                sel_te = np.where(
                    var_te <= tau, 1, 0
                )
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)
        elif meta == "scross":
            rcross = SCross(reg_base)
            rcross.fit(X_tr, y_tr)
            y_hat = rcross.predict(X_te)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rcross.select(X_te, X_val, coverage)
                tmp = get_metrics_test(y_te, y_hat, sel_te)
                tmp['target_coverage'] = coverage
                res = pd.concat([res, tmp], axis=0)

        res["meta"] = meta
        res["model"] = class_string
        res["dataset"] = dataset
        res["features"] = n_features
        res["trainingsize"] = train_size
        results = pd.concat([results, res], axis=0)
    return results

def main(dataset, regressors, metas, nj=1):
    np.random.seed(42)
    set_seed(42)
    try:
        X, y = fetch_data(dataset, return_X_y=True)
    except:
        raise FileNotFoundError("The dataset could not be retrieved. Please check.")
    for reg_string in regressors:
        tmp = experiment(X, y, dataset, metas, reg_string, n_jobs_class=nj)
        tmp.to_csv("results/{}/ALL_RESULTS_{}_{}_{}_BASE.csv".format(dataset,dataset,
                                                                     reg_string, "-".join(metas)), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Random seed
    set_seed(42)
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--reg", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int,
                        default=len(regression_dataset_names))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    regressors = args.reg
    metas = args.meta
    start = max(0, args.start)
    end = min(len(regression_dataset_names), args.end)
    # metas = ["doubt"]
    for dataset in tqdm(regression_dataset_names[start:end]):
        # print(dataset)
        main(dataset, regressors, metas, nj=args.jobs)
