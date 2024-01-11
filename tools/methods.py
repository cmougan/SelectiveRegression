import copy
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class SCross(BaseEstimator, ClassifierMixin):
    """
    Class for SCrpss
    """

    def __init__(self, model, selector=None, cv=5, random_seed=42):
        self.cv = cv
        self.seed = random_seed
        self.theta = None
        self.models = [copy.deepcopy(model) for _ in range(cv)]
        self.model = copy.deepcopy(model)
        self.selector = selector
        if self.selector is None:
            self.err_model = copy.deepcopy(self.model)
        else:
            self.err_model = copy.deepcopy(self.selector)

    def fit(self, X, y, conf="error"):
        z = []
        idx = []
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
                y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                y_test = y[test_index]
            self.models[i].fit(X_train, y_train)
            # quantiles
            preds = self.models[i].predict(X_test)
            if conf == "variance":
                ####HERE we use variance as a confidence
                self.confidence_f = "variance"
                variance = (preds - np.mean(preds)) ** 2
                z.append(variance)
            elif conf == "error":
                self.confidence_f = "error"
                error = (y_test - self.models[i].predict(X_test)) ** 2
                z.append(error.reshape(-1, 1))
                idx.append(test_index)
            else:
                raise NotImplementedError("Confidence not yet implemented.")
        self.confs = np.concatenate(z).ravel()
        if self.confidence_f == "error":
            self.idxs = list(np.concatenate(idx).ravel())
            if isinstance(X, pd.DataFrame):
                self.err_model.fit(X.iloc[self.idxs, :], self.confs)
            else:
                self.err_model.fit(X[self.idxs, :], self.confs)
        self.model.fit(X, y)

    def predict(self, X):
        scores = self.model.predict(X)
        return scores

    def calibrate(self, X, cov):
        if self.confidence_f == "variance":
            sub_confs_1, sub_confs_2 = train_test_split(
                self.confs, test_size=0.5, random_state=self.seed
            )
            tau = 1 / np.sqrt(2)
            self.theta = tau * np.quantile(self.confs, cov) + (1 - tau) * (
                0.5 * np.quantile(sub_confs_1, cov)
                + 0.5 * np.quantile(sub_confs_2, cov)
            )
        elif self.confidence_f == "error":
            confs_val = self.err_model.predict(X)
            sub_confs_1, sub_confs_2 = train_test_split(
                confs_val, test_size=0.5, random_state=self.seed
            )
            tau = 1 / np.sqrt(2)
            self.theta = tau * np.quantile(confs_val, cov) + (1 - tau) * (
                0.5 * np.quantile(sub_confs_1, cov)
                + 0.5 * np.quantile(sub_confs_2, cov)
            )

    def select(self, X, X_cal, cov):
        self.calibrate(X_cal, cov)
        preds = self.predict(X)
        if self.confidence_f == "variance":
            return np.where((preds - np.mean(preds)) ** 2 < self.theta, 1, 0)
        if self.confidence_f == "error":
            return np.where((self.err_model.predict(X)) < self.theta, 1, 0)
        else:
            raise NotImplementedError("Confidence not yet implemented.")

class CrossFit(BaseEstimator, ClassifierMixin):
    """
    Class for SCrpss
    """

    def __init__(self, model, selector=None, cv=2, random_seed=42):
        self.cv = cv
        self.seed = random_seed
        self.theta = None
        self._models = [copy.deepcopy(model) for _ in range(cv)]
        if selector is None:
            self._selectors = [copy.deepcopy(model) for _ in range(cv)]
        else:
            self._selectors = [copy.deepcopy(selector) for _ in range(cv)]

    def fit(self, X, y, pso=None):
        z = []
        idx = []
        skf = KFold(n_splits=len(self._models), shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
            else:
                X_train = X[train_index]
                X_test = X[test_index]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]
            else:
                y_train = y[train_index]
                y_test = y[test_index]
            self._models[i].fit(X_train, y_train)
            # quantiles
            preds = self._models[i].predict(X_test)
            if pso is None:
                pseudo_outcome = (y_test - preds) ** 2
                self._selectors[i].fit(X_test, pseudo_outcome)
            elif pso == "pseudo":
                pseudo_outcome = y_test**2-(y_test*preds)
                self._selectors[i].fit(X_test, pseudo_outcome)
            else:
                raise NotImplementedError("Pseudo outcome not yet implemented.")

    def predict(self, X):
        scores = np.mean([self._models[i].predict(X) for i in range(self.cv)], axis=0)
        return scores

    def error_predict(self, X):
        scores = np.mean([self._selectors[i].predict(X) for i in range(self.cv)], axis=0)
        return scores

    def calibrate(self, X, cov):
        confs_val = self.error_predict(X)
        sub_confs_1, sub_confs_2 = train_test_split(
            confs_val, test_size=0.5, random_state=self.seed
        )
        tau = 1 / np.sqrt(2)
        self.theta = tau * np.quantile(confs_val, cov) + (1 - tau) * (
            0.5 * np.quantile(sub_confs_1, cov)
            + 0.5 * np.quantile(sub_confs_2, cov)
        )

    def select(self, X, X_cal, cov):
        self.calibrate(X_cal, cov)
        preds = self.predict(X)
        return np.where((self.error_predict(X)) < self.theta, 1, 0)






class PlugIn(BaseEstimator):
    """
    Class for SCrpss
    """

    def __init__(self, model, random_seed=42):
        self.seed = random_seed
        self.theta = None
        self.model = copy.deepcopy(model)
        self.err_model = copy.deepcopy(model)

    def fit(self, X, y):
        self.model.fit(X, y)
        errors = (y - self.model.predict(X)) ** 2
        self.err_model.fit(X, errors)

    def predict(self, X):
        scores = self.model.predict(X)
        return scores

    def calibrate(self, X, cov):
        confs_val = self.err_model.predict(X)
        self.theta = np.quantile(confs_val, cov)

    def select(self, X, X_cal, cov):
        self.calibrate(X_cal, cov)
        # preds = self.predict(X)
        return np.where((self.err_model.predict(X)) < self.theta, 1, 0)



class ThreeWayCrossFit(BaseEstimator):
    """
    Class for SCrpss
    """

    def __init__(self, model, selector, var_model, random_seed=42):
        self.seed = random_seed
        self.theta = None
        self._models = {(i, j): copy.deepcopy(model) for i in range(3) for j in range(2)}
        self._var_models = {(i, j): copy.deepcopy(var_model) for i in range(3) for j in range(2)}
        self._selectors = {(i, j): copy.deepcopy(selector) for i in range(3) for j in range(2)}
    def fit(self, X, y, pso=None):
        z = []
        idx = []
        skf = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            train_index_1, train_index_2 = train_test_split(
                train_index, test_size=0.5, random_state=self.seed
            )
            for k in range(2):
                if k == 0:
                    train_mod_index = train_index_1
                    train_modsq_index = train_index_2
                else:
                    train_mod_index = train_index_2
                    train_modsq_index = train_index_1
                if isinstance(X, pd.DataFrame):
                    X_train = X.iloc[train_mod_index]
                    X_train_modsq = X.iloc[train_modsq_index]
                    X_test = X.iloc[test_index]
                else:
                    X_train = X[train_mod_index]
                    X_train_modsq = X[train_modsq_index]
                    X_test = X[test_index]
                if isinstance(y, pd.Series):
                    y_train = y.iloc[train_mod_index]
                    y_train_modsq = y.iloc[train_modsq_index]**2
                    y_test = y.iloc[test_index]
                else:
                    y_train = y[train_mod_index]
                    y_train_modsq = y[train_modsq_index]**2
                    y_test = y[test_index]
                self._models[(i,k)].fit(X_train, y_train)
                self._var_models[(i,k)].fit(X_train_modsq, y_train_modsq)
                # quantiles
                preds = self._models[(i,k)].predict(X_test)
                preds_sq = self._var_models[(i,k)].predict(X_test)
                if pso == "influence":
                    pseudo_outcome = (
                                     (y_test - preds)**2
                   +( y_test**2 - preds_sq + 2*(preds**2) - 2*y_test*preds)
                   #  preds_sq - preds**2
                )
                elif pso == "influence2":
                    pseudo_outcome = (
                      preds_sq - preds**2
                )
                elif pso is None:
                    pseudo_outcome = (y_test - preds) ** 2
                else:
                    raise NotImplementedError("Pseudo outcome not yet implemented.")
                self._selectors[(i,k)].fit(X_test, pseudo_outcome)
    def predict(self, X):
        scores = np.mean([self._models[(i,j)].predict(X) for i in range(3) for j in range(2)], axis=0)
        return scores

    def error_predict(self, X):
        scores = np.mean([self._selectors[(i,j)].predict(X) for i in range(3) for j in range(2)], axis=0)
        return scores

    def calibrate(self, X, cov):
        confs_val = self.error_predict(X)
        sub_confs_1, sub_confs_2 = train_test_split(
            confs_val, test_size=0.5, random_state=self.seed
        )
        tau = 1 / np.sqrt(2)
        self.theta = tau * np.quantile(confs_val, cov) + (1 - tau) * (
                0.5 * np.quantile(sub_confs_1, cov)
                + 0.5 * np.quantile(sub_confs_2, cov)
        )

    def select(self, X, X_cal, cov):
        self.calibrate(X_cal, cov)
        preds = self.predict(X)
        return np.where((self.error_predict(X)) < self.theta, 1, 0)

