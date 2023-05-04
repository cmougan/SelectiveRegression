import copy
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class SCross(BaseEstimator, ClassifierMixin):
    """
    Class for SCrpss
    """

    def __init__(self, model, cv=5, random_seed=42):
        self.cv = cv
        self.seed = random_seed
        self.theta = None
        self.models = [copy.deepcopy(model) for _ in range(cv)]
        self.model = copy.deepcopy(model)

    def fit(self, X, y, conf="variance"):
        z = []
        skf = KFold(n_splits=len(self.models), shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
            else:
                X_train = X[train_index]
                X_test = X[test_index]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_index]
                # y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                # y_test = y[test_index]
            self.models[i].fit(X_train, y_train)
            # quantiles
            preds = self.models[i].predict(X_test)
            if conf == "variance":
            ####HERE we use variance as a confidence
                self.confidence_f = "variance"
                variance = (preds - np.mean(preds))**2
                z.append(variance)
            else:
                raise NotImplementedError("Confidence not yet implemented.")
        self.confs = np.concatenate(z).ravel()
        self.model.fit(X, y)

    def predict(self, X):
        scores = self.model.predict(X)
        return scores
    def calibrate(self, cov):
        sub_confs_1, sub_confs_2 = train_test_split(self.confs, test_size=.5, random_state=self.seed)
        tau = (1 / np.sqrt(2))
        self.theta = (tau * np.quantile(self.confs, cov) + (1 - tau) * (
                .5 * np.quantile(sub_confs_1, cov) + .5 * np.quantile(sub_confs_2, cov)))
    def select(self, X, cov):
            self.calibrate(cov)
            preds = self.predict(X)
            if self.confidence_f == "variance":
                return np.where((preds-np.mean(preds))**2 < self.theta, 1, 0)
            else:
                raise NotImplementedError("Confidence not yet implemented.")



