import torch
import rtdl
from torch.nn import Module
from torch import nn
import math


class TabCatResNet(Module):
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 1,
        n_blocks: int = 2,
        d_main: int = 3,
        d_hidden: int = 4,
        dropout_first=0.25,
        dropout_second=0.00,
    ):
        """

        :param d_in: int
                the number of continuous variables
        :param cat_cardinalities:
                the
        :param d_token:
        :param d_out:
        :param n_blocks: the number of Blocks
        :param d_main: the input size (or, equivalently, the output size) of each Block
        :param d_hidden: the output size of the first linear layer in each Block
        :param dropout_first: the dropout rate of the first dropout layer in each Block.
        :param dropout_second: the dropout rate of the second dropout layer in each Block.
        """
        super(TabCatResNet, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        if self.cat_cardinalities != []:
            self.CAT = rtdl.CategoricalFeatureTokenizer(
                self.cat_cardinalities, self.d_token, True, "uniform"
            )
        self.n_blocks = n_blocks
        self.d_main = d_main
        self.d_hidden = d_hidden
        self.dropout_first = dropout_first
        self.dropout_second = dropout_second
        self.d_out = d_out
        self.resnet = rtdl.ResNet.make_baseline(
            d_in=self.d_in + len(self.cat_cardinalities) * self.d_token,
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout_first,
            dropout_second=self.dropout_second,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_num, x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x_cat = self.CAT(x_cat)
            x = torch.cat([x_cat.flatten(1, -1)], dim=1)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = torch.cat([x_num], dim=1)
        x = self.resnet(x)
        return x


class TabFTTransformer(Module):
    def __init__(
        self,
        d_in: int,
        cat_cardinalities: list,
        d_token: int = 192,  # as the default for FTTransformer in Gorishniy et al. 2021
        d_out: int = 1,
        n_blocks: int = 3,
        attention_dropout: float = 0.2,
        ffn_d_hidden: int = 256,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0,
    ):
        """

        :param d_in:
        :param cat_cardinalities:
        :param d_token:
        :param d_out:
        :param n_blocks:
        :param attention_dropout:
        :param ffn_d_hidden:
        :param ffn_dropout:
        :param residual_dropout:
        """

        super(TabFTTransformer, self).__init__()
        self.d_in = d_in
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token

        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.d_out = d_out
        self.ftt = rtdl.FTTransformer.make_baseline(
            n_num_features=self.d_in,
            cat_cardinalities=self.cat_cardinalities,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            d_out=self.d_out,
        )

    def forward(self, x_num, x_cat):
        if (self.cat_cardinalities != []) & (self.d_in > 0):
            x = self.ftt(x_num, x_cat)
        elif (self.cat_cardinalities != []) & (self.d_in == 0):
            x = self.ftt(None, x_cat)
        elif (self.cat_cardinalities == []) & (self.d_in > 0):
            x = self.ftt(x_num, None)
        return x


class HeadSelectiveNet(Module):
    """
    Module for the head implementation of Selective Net.
    This is intended to substitute the head structure of rtdl modules
    """

    def __init__(
        self,
        d_in: int = 128,
        d_out: int = 2,
        batch_norm: str = "batch_norm",
        main_body: str = "resnet",
        pre_norm: bool = True,
    ):
        super(HeadSelectiveNet, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.batch_norm = batch_norm
        self.main_body = main_body
        self.dense_class = torch.nn.Linear(self.d_in, self.d_out)
        self.dense_selec_1 = torch.nn.Linear(self.d_in, int(self.d_in / 2))
        # if the model is VGG-based we apply an additional linear layer as in SelNet paper
        if self.main_body == "VGG":
            self.first_layer = nn.Sequential(
                nn.Linear(self.d_in, self.d_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.d_in),
                nn.Dropout(0.5),
            )
        # if the model is a resnet-based or a transformer we apply normalization before the head layers
        # depending on the model we change the normalization
        # if (self.main_body in ["resnet", "transformer"]) and (
        #     self.batch_norm == "batch_norm"
        # ):
        if (pre_norm) and (self.batch_norm == "batch_norm"):
            self.pre_norm = torch.nn.BatchNorm1d(self.d_in)
        elif (pre_norm) and (self.batch_norm == "layer_norm"):
            self.pre_norm = torch.nn.LayerNorm(self.d_in)
        else:
            self.pre_norm = None
        # depending on the model we change the normalization
        if self.batch_norm == "batch_norm":
            self.batch_norm = torch.nn.BatchNorm1d(int(self.d_in / 2))
            # if the model is a resnet-based or a transformer we apply normalization before the head layers
        elif self.batch_norm == "layer_norm":
            self.batch_norm = torch.nn.LayerNorm(int(self.d_in / 2))
        self.dense_selec_2 = torch.nn.Linear(int(self.d_in / 2), 1)
        self.dense_auxil = torch.nn.Linear(self.d_in, self.d_out)

    def forward(self, x):
        if self.main_body == "transformer":
            x = x[:, -1]
        if self.main_body == "VGG":
            x = self.first_layer(x)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        h = self.dense_class(x)
        # h = torch.nn.functional.sigmoid(h)
        g = self.dense_selec_1(x)
        g = torch.nn.functional.relu(g)
        g = self.batch_norm(g)
        g = self.dense_selec_2(g)
        g = torch.sigmoid(g)
        a = self.dense_auxil(x)
        # a = torch.nn.functional.softmax(a, dim=1)
        hg = torch.cat([h, g], 1)
        return hg, a
