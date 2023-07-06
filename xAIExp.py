# %%
# Data from https://www.kaggle.com/datasets/chandramoulinaidu/house-price-prediction-cleaned-dataset?resource=download&select=Cleaned+train.csv
# Import candidate models
from doubt import Boot, QuantileRegressor, QuantileRegressionForest
from sklearn.linear_model import (
    LinearRegression,
    PoissonRegressor,
    GammaRegressor,
    HuberRegressor,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


# Import external libraries
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
from scipy.stats import ks_2samp, entropy, kruskal
import matplotlib.pyplot as plt
import itertools
from matplotlib import rcParams


plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8

plt.style.use("ggplot")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from collections import defaultdict
import seaborn as sns
import pdb

sns.set_theme(style="whitegrid")
import shap

# Import internal classes

import warnings

warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from category_encoders import MEstimateEncoder

# %%
def explain(xgb: bool = True):
    """
    Provide a SHAP explanation by fitting MEstimate and GBDT
    """
    if xgb:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", GradientBoostingRegressor())]
        )
        pipe.fit(X, y)
        explainer = shap.Explainer(pipe[1])
        shap_values = explainer(pipe[:-1].transform(X))
        shap.plots.beeswarm(shap_values)
        return pd.DataFrame(np.abs(shap_values.values), columns=X.columns).sum()
    else:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", LogisticRegression())]
        )
        pipe.fit(X, y)
        coefficients = pd.concat(
            [pd.DataFrame(X_tr.columns), pd.DataFrame(np.transpose(pipe[1].coef_))],
            axis=1,
        )
        coefficients.columns = ["feat", "val"]

        return coefficients.sort_values(by="val", ascending=False)


# %%
df = pd.read_csv("data/trainclean.csv")
df = df.drop(columns="Id")

df["random"] = np.random.random(df.shape[0])
df["random"].mean()
# %%
