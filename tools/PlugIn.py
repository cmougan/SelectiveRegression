from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
import numpy as np


class PlugInRule(BaseEstimator, ClassifierMixin):
    """
    TODO Doc

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression
    >>> from tools.xaiUtils import PlugInRule
    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> clf = PlugInRule(model=XGBRegressor())
    >>> clf.fit(X_tr, y_tr)
    >>> clf.predict(X_te)
    """

    def __init__(
        self,
        model,
        seed: int = 42,
    ):
        self.seed = seed
        self.model = model

    def fit(self, X, y):
        # Dimensionality check
        check_X_y(X, y)

        # Split the data and save hold out sets
        X_train, self.X_hold, y_train, self.y_hold = train_test_split(
            X, y, stratify=y, random_state=self.seed, test_size=0.1
        )
        # Fit the model
        self.model.fit(X_train, y_train)

    def compute_theta(self, q: int = 0.9):
        # Compute the scores
        scores = np.max(self.model.predict_proba(self.X_hold), axis=1)

        # TODO : deal with quantile lists

        # Compute the theta
        self.theta = np.quantile(scores, q)

    def predict(self, X, cov: int = 0.9):
        # TODO check if the below function has been called
        self.compute_theta(1 - cov)

        probas = self.model.predict_proba(X)
        confs = np.max(probas, axis=1)
        return confs > self.theta
