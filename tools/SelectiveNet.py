import sklearn.preprocessing
import torch.utils.data
import torch.optim as optim
from tools.utils import *
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd


class SelectiveNetRegressor(ClassifierMixin, BaseEstimator):
    """ """

    def __init__(
        self,
        body_dict: dict,
        model_type: str = "TabFTTransformer",
        coverage: float = 0.99,
        seed: int = 42,
    ):
        """
        :param body:
        :param model_type:
        :param coverage:
        :param seed:
        """
        self.body_dict = body_dict
        self.model_type = model_type
        self.coverage = coverage
        self.seed = seed
        self.thetas = None
        self.model = None
        self.scaler = None

    def fit(
        self,
        X,
        y,
        scaler="minmax",
        epochs=50,
        bsize=32,
        shuffle=True,
        device="cuda:0",
        optimizer_name="SGD",
        nesterov=False,
        lr=1e-2,
        wd=5e-4,
        momentum_sgd=0.99,
        td=True,
        alpha=0.5,
        lamda=32,
        verbose=True,
    ):
        """

        :param X: numpy.ndarray or pandas.DataFrame
        :param y: numpy.ndarray or pandas.Series
        :param scaler: str
        :param epochs: int
        :param bsize: int
        :param shuffle: bool
        :param device: str
        :param optimizer_name: str
        :param nesterov: bool
        :param lr: float
        :param wd: float
        :param momentum_sgd: float
        :param td: bool
        :param alpha: float
        :param lamda: int
        :return:
        """
        if self.model is not None:
            print("Model already fitted")
        else:
            if type(X) == np.array:
                cols = ["X{}".format(el) for el in range(X.shape[1])]
                df = pd.DataFrame(np.c_[X, y], columns=cols + ["TARGET"])
                atts = cols
                self.atts = atts
            elif type(X) == np.ndarray:
                cols = ["X{}".format(el) for el in range(X.shape[1])]
                df = pd.DataFrame(np.c_[X, y], columns=cols + ["TARGET"])
                atts = cols
                self.atts = atts
            elif type(X) == pd.DataFrame:
                atts = [el for el in X.columns]
                df = pd.concat([X, y], axis=1)
                df.columns = atts + ["TARGET"]
                self.atts = atts
            df.reset_index(inplace=True)
            if torch.cuda.is_available():
                self.device = device
            else:
                self.device = "cpu"
                print("Cuda is not available. The device is set to cpu.")
            if scaler == "minmax":
                self.scaler = sklearn.preprocessing.MinMaxScaler()
                self.scaler.fit(df["TARGET"].values.reshape(-1, 1))
                df["TARGET"] = self.scaler.transform(df["TARGET"].values.reshape(-1, 1))
            else:
                raise NotImplementedError("Other scaler not yet implented.")
            training_set = TabularDataset(df, atts, pred=False)
            set_seed(self.seed)
            train_dl = torch.utils.data.DataLoader(
                training_set, batch_size=bsize, shuffle=shuffle
            )
            self.x_num = training_set.x_num.shape[1]
            try:
                self.cat_dim = (training_set.x_cat.max(dim=0).values + 1).tolist()
            except:
                self.cat_dim = []
            model = tabular_model(
                self.model_type, self.x_num, self.cat_dim, "selnet", self.body_dict
            )
            if nesterov:
                optimizer = getattr(optim, optimizer_name)(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    nesterov=nesterov,
                    momentum=momentum_sgd,
                )
            else:
                optimizer = getattr(optim, optimizer_name)(
                    model.parameters(), lr=lr, weight_decay=wd
                )
            model.train()
            self.model = train(
                model,
                self.device,
                epochs,
                optimizer,
                "selnet",
                train_dl,
                lamda=lamda,
                alpha=alpha,
                coverage=self.coverage,
                td=td,
                verbose=verbose
            )

    def predict_conf(self, X):
        """
        :param X:
        :return:
        """
        if type(X) == np.array:
            cols = ["X{}".format(el) for el in range(X.shape[1])]
            atts = cols
            df = pd.DataFrame(X, columns=atts)
        elif type(X) == np.ndarray:
            cols = ["X{}".format(el) for el in range(X.shape[1])]
            atts = cols
            df = pd.DataFrame(X, columns=atts)
            self.atts = atts
        elif type(X) == pd.DataFrame:
            atts = [el for el in X.columns]
            df = pd.DataFrame(X, columns=atts)
            self.atts = atts
        test_set = TabularDataset(df, atts, pred=True)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
        return predict_conf(self.model, self.device, test_dl, "selnet")

    def predict(self, X, rescale=True, conf=True):
        """

        :param X: np.array or pandas.DataFrame.
                The X features to predict.
        :param rescale: bool
                If True scales back to original scale the target predictions. The default is True.
        :param conf: bool
                If True, it provides also the selective head confidence. The default is True.
        :return: np.array or tuple of arrays
                y_hat and (if conf is True) y_conf
        """
        if type(X) == np.array:
            cols = ["X{}".format(el) for el in range(X.shape[1])]
            atts = cols
            df = pd.DataFrame(X, columns=atts)
        elif type(X) == np.ndarray:
            cols = ["X{}".format(el) for el in range(X.shape[1])]
            atts = cols
            df = pd.DataFrame(X, columns=atts)
            self.atts = atts
        elif type(X) == pd.DataFrame:
            atts = [el for el in X.columns]
            df = pd.DataFrame(X, columns=atts)
            self.atts = atts
        test_set = TabularDataset(df, atts, pred=True)
        test_dl = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
        assert (self.scaler is not None, "The model is not yet fitted")
        if rescale == True:
            y_hat = predict(self.model, self.device, test_dl, "selnet")
            if conf == True:
                y_conf = predict_conf(self.model, self.device, test_dl, "selnet")
                return self.scaler.inverse_transform(y_hat.reshape(-1, 1)), y_conf
            else:
                return self.scaler.inverse_transform(y_hat.reshape(-1, 1))
        else:
            y_hat = predict(self.model, self.device, test_dl, "selnet")
            if conf == True:
                y_conf = predict_conf(self.model, self.device, test_dl, "selnet")
                return y_hat, y_conf
            else:
                return y_hat
