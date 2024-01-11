# %%
# Imports
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from tools.methods import SCross, PlugIn, CrossFit
import category_encoders as ce
plt.style.use("seaborn-whitegrid")
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Doubt
from doubt import Boot
from xgboost import XGBRegressor
from tools.utils import set_seed
from lightgbm import LGBMRegressor

#eural Networks
import torch.optim as optim
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint
from skorch.dataset import ValidSplit
from tools.networks import *

# Benchmark
from mapie.regression import MapieRegressor
from pmlb import fetch_data



def get_metrics_test(true, y_hat, selected, mean_train=1):
    if np.sum(selected) > 0:
        coverage = len(selected[selected == 1]) / len(selected)
        mae = mean_absolute_error(true[selected == 1], y_hat[selected == 1])
        mse = mean_squared_error(true[selected == 1], y_hat[selected == 1])
        mse_stupid = mean_squared_error(true[selected == 1], (mean_train * np.ones(true.shape))[selected == 1])
        mae_stupid = mean_absolute_error(true[selected == 1], (mean_train * np.ones(true.shape))[selected == 1])
    else:
        coverage = -1
        mae = -1
        mse = -1
        mse_stupid = -1
        mae_stupid = -1
    tmp = pd.DataFrame([[coverage, mae, mse, mae_stupid, mse_stupid]], columns=["coverage", "MAE", "MSE", "MAE_ref", "MSE_ref"])
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
    nj = 1,
    bsize = 128,
    device = "cpu"
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
    perc_test = 0.2
    perc_hold = 0.125
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=perc_test, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, random_state=42, test_size=perc_hold)
    # // use target encoder for categorical variables
    print(X_train.shape, X_val.shape, X_test.shape)
    shape_original = X_train.shape
    cat_atts = list(
        X.select_dtypes(include=["object", "category"]).columns
    )
    cont_atts = list(
        X.select_dtypes(exclude=["object", "category"]).columns
    )
    te = ce.TargetEncoder()
    if cat_atts != []:
        X_train_cat = te.fit_transform(X_train[cat_atts], y_train).astype(np.float32)
        X_val_cat = te.transform(X_val[cat_atts]).astype(np.float32)
        X_test_cat = te.transform(X_test[cat_atts]).astype(np.float32)
        print(X_train_cat.shape, X_val_cat.shape, X_test_cat.shape)
        X_train = pd.concat([X_train[cont_atts], X_train_cat], axis=1)
        X_val = pd.concat([X_val[cont_atts], X_val_cat], axis=1)
        X_test = pd.concat([X_test[cont_atts], X_test_cat], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X=X_train)
    X_train = scaler.transform(X=X_train).astype(np.float32)
    X_test = scaler.transform(X=X_test).astype(np.float32)
    X_val = scaler.transform(X=X_val).astype(np.float32)
    print(X_train.shape, X_val.shape, X_test.shape)
    assert X_train.shape == shape_original
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    n_features = len(X.columns)
    train_size = X_train.shape[0]
    results = pd.DataFrame()
    if n_boot is None:
        if reg_base.__class__.__name__ == "NeuralNetRegressor":
            nb = min(50, np.sqrt(X_train.shape[0]).astype(int))
        else:
            nb = np.sqrt(X_train.shape[0]).astype(int)
    else:
        nb = n_boot
    print(nb)
    model_name = reg_base.__class__.__name__
    if reg_base.__class__.__name__ == "NeuralNetRegressor":
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    for meta in metas:
        if meta == "doubt":
            # initialize the model
            dbt = Boot(reg_base, random_seed=seed)
            if reg_base.__class__.__name__ == "NeuralNetRegressor":
                dbt.fit(X_train, y_train, n_jobs=1, n_boots=nb)
                model_name = dbt._model.module_.__class__.__name__
                y_hat, unc_te = dbt.predict(X_test.astype(np.float32), n_boots=nb, n_jobs=1, return_all=True)
                _, unc_val = dbt.predict(X_val.astype(np.float32), return_all=True, n_boots=nb, n_jobs=1)
                quantiles = [uncertainty / 2, 1 - uncertainty / 2]
                unc_te_ = np.transpose(np.quantile(unc_te, q=quantiles or [], axis=2))
                unc_val_ = np.transpose(np.quantile(unc_val, q=quantiles or [], axis=2))
                interval_te = (unc_te_[:, :, 1] - unc_te_[:, :, 0]).flatten()
                interval_val = (unc_val_[:, :, 1] - unc_val_[:, :, 0]).flatten()
                interval_te_new = np.var(unc_te.astype(np.float32), axis=2).flatten()
                interval_val_new = np.var(unc_val.astype(np.float32), axis=2).flatten()
            else:
                dbt.fit(X_train, y_train, n_jobs=nj, n_boots=nb)
                _, unc_te = dbt.predict(X_test, n_jobs=nj, n_boots=nb, uncertainty=uncertainty)
                _, unc_val = dbt.predict(X_val, n_jobs=nj, n_boots=nb, uncertainty=uncertainty)
                interval_te = unc_te[:, 1] - unc_te[:, 0]
                interval_val = unc_val[:, 1] - unc_val[:, 0]
                # New uncertainty on Test and Val
                _, unc_te_new = dbt.predict(X_test, n_jobs=nj, n_boots=nb, return_all=True)
                interval_te_new = np.var(unc_te_new, axis=1)
                _, unc_val_new = dbt.predict(X_val, n_jobs=nj, n_boots=nb, return_all=True)
                interval_val_new = np.var(unc_val_new, axis=1)
                y_hat = dbt._model.predict(X_test)
            res = pd.DataFrame()
            for coverage in coverages:
                tau = np.quantile(interval_val_new, coverage)
                sel_te = np.where(interval_te_new <= tau, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te, mean_train=np.mean(y_train))
                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                tmp["meta"] = "doubtVar"
                res = pd.concat([res, tmp], axis=0)
                tau_2 = np.quantile(interval_val, coverage)
                sel_te_2 = np.where(interval_te <= tau_2, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te_2, mean_train=np.mean(y_train))
                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                tmp["meta"] = "doubtInt"
                res = pd.concat([res, tmp], axis=0)
        elif meta == "plugin":
            rplug = PlugIn(reg_base)
            rplug.fit(X_train, y_train)
            y_hat = rplug.predict(X_test)
            res = pd.DataFrame()
            error = (y_test - rplug.predict(X_test)) ** 2
            for coverage in coverages:
                sel_te = rplug.select(X_test, X_val, coverage)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te,  mean_train=np.mean(y_train))

                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                tmp["meta"] = "plugin"
                res = pd.concat([res, tmp], axis=0)
                tau_gold = np.quantile(
                    error, coverage
                )  # tau on the test set - we accept instances below the error tau_gold
                sel_te = np.where(error <= tau_gold, 1, 0)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te, mean_train=np.mean(y_train))
                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                tmp["meta"] = "gold"
                res = pd.concat([res, tmp], axis=0)
        elif meta == "scross":
            rcross = SCross(reg_base)
            rcross.fit(X_train, y_train)
            y_hat = rcross.predict(X_test)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rcross.select(X_test, X_val, coverage)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te, mean_train=np.mean(y_train))
                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                res = pd.concat([res, tmp], axis=0)
            res["meta"] = "scross"
        elif meta == "crossfit":
            rcross = CrossFit(reg_base)
            rcross.fit(X_train, y_train)
            y_hat = rcross.predict(X_test)
            res = pd.DataFrame()
            for coverage in coverages:
                sel_te = rcross.select(X_test, X_val, coverage)
                if coverage != 1:
                    tmp = get_metrics_test(y_test, y_hat, sel_te, mean_train=np.mean(y_train))
                else:
                    tmp = get_metrics_test(y_test, y_hat, np.ones(y_test.shape), mean_train=np.mean(y_train))
                tmp["target_coverage"] = coverage
                tmp["meta"] = "crossfit"
                res = pd.concat([res, tmp], axis=0)
        elif meta == "mapie":
            if reg_base.__class__.__name__ == "NeuralNetRegressor":
                reg_base = NeuralNetRegressor2(
                        TabFTTransformer,
                        module__d_in=X.shape[1],
                        module__cat_cardinalities=[],
                        module__d_token=192,
                        module__d_out=1,
                        module__n_blocks=3,
                        batch_size=bsize,
                        max_epochs=100,
                        lr=0.01,
                        optimizer=optim.Adam,
                        criterion=CustomMSE,
                        train_split=ValidSplit(.1, random_state=seed),
                        callbacks=[
                                   LRScheduler(policy=optim.lr_scheduler.StepLR, step_size=10, gamma=0.5),
                                   Checkpoint(monitor='valid_loss_best', fn_prefix='best_model_{}_{}'.format(meta,dataset), load_best=True)
                                   ],
                        verbose=False,
                        device=device if torch.cuda.is_available() else 'cpu'
                    )
            mapie = MapieRegressor(reg_base, cv=5, random_state=seed)
            try:
                mapie.fit(X_train, y_train)
                y_hat, unc_te_map = mapie.predict(X_test, alpha=[uncertainty])
                y_hat_val, unc_val_map = mapie.predict(X_val, alpha=[uncertainty])
                interval_te = unc_te_map[:, 1, 0] - unc_te_map[:, 0, 0]
                interval_val = unc_val_map[:, 1, 0] - unc_val_map[:, 0, 0]
                res = pd.DataFrame()
                for coverage in coverages:
                    tau = np.quantile(interval_val, coverage)
                    sel_te = np.where(interval_te <= tau, 1, 0)
                    if coverage != 1:
                        tmp = get_metrics_test(y_test, y_hat, sel_te, mean_train=np.mean(y_train))
                    else:
                        tmp = get_metrics_test(y_test, y_hat, np.ones(y_hat.shape), mean_train=np.mean(y_train))
                    tmp["target_coverage"] = coverage
                    tmp["meta"] = "mapie"
                    res = pd.concat([res, tmp], axis=0)
            except ValueError:
                res = pd.DataFrame()
                mapie = copy.deepcopy(reg_base)
                mapie.fit(X_train, y_train)
                y_hat = mapie.predict(X_test)
                for coverage in coverages:
                    if coverage != 1:
                        tmp = get_metrics_test(y_test, y_hat, np.zeros(y_hat.shape), mean_train=np.mean(y_train))
                    else:
                        tmp = get_metrics_test(y_test, y_hat, np.ones(y_hat.shape), mean_train=np.mean(y_train))
                    tmp["target_coverage"] = coverage
                    tmp["meta"] = "mapie"
                    res = pd.concat([res, tmp], axis=0)
        res["model"] = model_name
        if hasattr(reg_base, "hidden_layers"):
            if reg_base.hidden_layers[0] > 100:
                res["model"] = "MLPRegressorLarge"
        res["dataset"] = dataset
        res["features"] = n_features
        res["trainingsize"] = train_size
        res["nboots"] = nb
        results = pd.concat([results, res], axis=0)
        results.to_csv(
                "resultsV3/{}/FINAL_RESULTS_{}_{}_{}_SEED{}{}.csv".format(
                    dataset, dataset, model_name, "-".join(metas), seed, ""
                ),
                index=False,
            )
    return results

def get_data_PMLB(dataset):
    X, y = fetch_data(dataset, return_X_y=True)
    X = pd.DataFrame(X.copy()).reset_index(drop=True)
    X.columns = ["Var" + str(i) for i in X.columns]
    y = pd.Series(y.copy()).values
    return X, y

def read_data(dataset, data_folder="data/grin_benchmark/", seed=42):
    df = pd.read_csv("{}/{}/{}.csv".format(data_folder, dataset, dataset))
    if df.shape[0] > 50000.0/0.7:
        df = df.sample(int(50000.0/0.7)+1, random_state=seed, axis=0)
    X = df.drop("TARGET", axis=1).copy()
    y = df["TARGET"].copy()
    return X, y


def main(dataset, regressors, metas, nj=1, seed=42, nboots=None, device="cuda:0"):
    np.random.seed(seed)
    set_seed(seed)
    try:
        X, y = get_data_PMLB(dataset)
    except:
        raise FileNotFoundError("The dataset could not be retrieved. Please check.")
    if X.shape[0] < 1000:
        bsize = 128
    elif X.shape[0] < 10000:
        bsize = 256
    elif X.shape[0] < 100000:
        bsize = 512
    else:
        bsize = 1024
    for reg_string in regressors:
        if reg_string == "lasso":
            reg_base = Lasso(random_state=seed, tol=0.001,
          max_iter=10000)
        elif reg_string == "lr":
            reg_base = LinearRegression(n_jobs=nj)
        elif reg_string == "xgb":
            reg_base = XGBRegressor(random_state=seed, n_jobs=nj)
        elif reg_string == "lgbm":
            reg_base = LGBMRegressor(random_state=seed, n_jobs=nj)
        elif reg_string == "dt":
            reg_base = DecisionTreeRegressor(random_state=seed, max_depth=10)
        elif reg_string == "rf":
            reg_base = RandomForestRegressor(random_state=seed)
        elif reg_string == "mlp":
            reg_base = MLPRegressor(random_state=seed, batch_size=bsize, max_iter=100)
        elif reg_string == "mlpLarge":
            import mkl
            mkl.set_num_threads(1)
            reg_base = MLPRegressor(
                random_state=seed, max_iter=100,
                hidden_layer_sizes=(256, 256, 256, 256), batch_size=bsize
            )
        elif reg_string == "ftt":
            reg_base = NeuralNetRegressor(
                        TabFTTransformer,
                        module__d_in= X.shape[1],
                        module__cat_cardinalities= [],
                        module__d_token= 192,
                        module__d_out= 1,
                        module__n_blocks= 3,
                        batch_size= bsize,
                        max_epochs=100,
                        lr=0.01,
                        optimizer=optim.Adam,
                        criterion=nn.MSELoss,
                        train_split=ValidSplit(.1, random_state=seed),
                        callbacks=[
                           LRScheduler(policy=optim.lr_scheduler.StepLR, step_size=10, gamma=0.5),
                           Checkpoint(monitor='valid_loss_best', fn_prefix='best_model_{}'.format(dataset), load_best=True)
                           ],
                        verbose=False,
                device=device if torch.cuda.is_available() else 'cpu'
                    )
        if os.path.exists("resultsV3/") == False:
            os.mkdir("resultsV3")
        if os.path.exists("resultsV3/{}/".format(dataset)) == False:
            os.mkdir("resultsV3/{}/".format(dataset))
        if (X.shape[0] >= 1000):
            tmp = experiment(X, y, dataset, metas, reg_base, seed=seed, n_boot=nboots, bsize=bsize, device=device)
            tmp["seed"] = seed
            if nboots is None:
                boot_str = ""
            else:
                boot_str = "_BOOTS_{}".format(nboots)
            tmp.to_csv(
                "resultsV3/{}/FINAL_RESULTS_{}_{}_{}_SEED{}{}.csv".format(
                    dataset, dataset, reg_string, "-".join(metas), seed, boot_str
                ),
                index=False,
            )
        else:
            print("Dataset not included")


if __name__ == "__main__":
    list_datasets = ['1028_SWD', '1029_LEV', '1030_ERA', '1193_BNG_lowbwt',
        '1199_BNG_echoMonths', '197_cpu_act', '215_2dplanes',
     '225_puma8NH', '227_cpu_small',
        '294_satellite_image', '344_mv',
        '4544_GeographicalOriginalofMusic', '503_wind', '529_pollen',
         '564_fried']
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--reg", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(list_datasets))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", nargs="+",  default=42)
    parser.add_argument("--boots", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    regressors = args.reg
    metas = args.meta
    start = max(0, args.start)
    end = min(len(list_datasets), args.end)
    for dataset in tqdm(list_datasets[start:end]):
        print(dataset)
        for seed in args.seed:
            # with EmissionsTracker() as tracker:
                main(dataset, regressors, metas, nj=args.jobs, seed=int(seed), nboots=args.boots, device=args.device)
