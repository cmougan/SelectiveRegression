import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class TabularDataset(Dataset):
    def __init__(
        self,
        dataset: str or pd.DataFrame,
        atts: list,
        root: str = "data/clean",
        target: str = "TARGET",
        set: str = "train",
        pred: bool = False,
        split: bool = False,
        test_perc: float = 0.1,
        device: str = "cuda:0",
    ):
        """

        :param dataset: str or Pandas.DataFrame
        :param atts: list
                The names of features.
        :param root:
                The path to data. Ignored if dataset is a Pandas.DataFrame.
        :param target:
                The name of target variable. The default is 'TARGET'.
        :param set:
                The set to which we are referring. Ignored if dataset is a Pandas DataFrame.
        :param pred: bool
                It determines whether the target variable is seen or not. The default is False.
        :param device:
        """
        super(TabularDataset, self).__init__()
        if type(dataset) == str:
            self.path = os.path.join(root, dataset, dataset + "_" + set + ".csv")
            self.df = pd.read_csv(self.path)
        elif type(dataset) == pd.DataFrame:
            self.df = dataset
        self.cat_atts = list(
            self.df[atts].select_dtypes(include=["object", "category"]).columns
        )
        self.cont_atts = list(
            self.df[atts].select_dtypes(exclude=["object", "category"]).columns
        )
        for col in self.cat_atts:
            self.df[col] = self.df[col].astype("category").cat.codes
        for col in self.cont_atts:
            self.df[col] = self.df[col].astype(float)
        self.pred = pred
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"
            print("Cuda is not available. The device is set to cpu.")
        self.x_num = (
            torch.from_numpy(self.df[self.cont_atts].values).float().to(self.device)
        )
        self.x_cat = (
            torch.from_numpy(self.df[self.cat_atts].values).to(self.device).long()
        )
        self.data = torch.cat([self.x_num, self.x_cat], dim=1)
        if self.pred is False:
            self.y = torch.from_numpy(self.df[target].values).to(self.device)
            self.targets = self.df[target].values
            self.classes = np.unique(self.df[target])

    def __getitem__(self, index):
        if self.pred:
            return self.x_num[index], self.x_cat[index]
        else:
            return self.x_num[index], self.x_cat[index], self.y[index], index

    def __len__(self):
        return self.data.shape[0]


class ImgFolder(ImageFolder):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, index
