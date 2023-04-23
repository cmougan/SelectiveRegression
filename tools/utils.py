import torch.utils.data
from tqdm import tqdm
import random
import copy
from tools.modules import *
from tools.losses import *
from tools.datasets import *

cfg = {
    "D": [
        64,
        0.3,
        64,
        "M",
        128,
        0.4,
        128,
        "M",
        256,
        0.4,
        256,
        0.4,
        256,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        512,
        0.4,
        512,
        0.4,
        512,
        "M",
        0.5,
    ]
}


# set seed function
def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2**32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(f"Random seed {seed} has been set.")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def buildSelNet(
    model_type: str,
    body_params: dict,
    head_params: dict,
):
    """
    :param model_type: str
            The main body type. Possible choices are:
            - 'FTResnet'
            - 'CATResnet'
            - 'FTTransformer'
            - 'VGG16'
            - 'VGG16bn'
    :param body_params:
    :param head_params:
    :return:
    """
    if model_type == "TabResnet":
        model = TabCatResNet(**body_params)
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "resnet"
            ), "Check the head is configured for a ResNet architecture"
        else:
            head_params["main_body"] = "resnet"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.resnet.blocks[-1].linear_second.out_features
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["d_in"] = model.resnet.blocks[-1].linear_second.out_features
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadSelectiveNet(**head_params)
        model.resnet.head = head
    elif model_type == "TabFTTransformer":
        model = TabFTTransformer(**body_params)
        if "main_body" in head_params.keys():
            assert (
                head_params["main_body"] == "transformer"
            ), "Check the head is configured for a transformer architecture"
        else:
            head_params["main_body"] = "transformer"
        if "d_in" in head_params.keys():
            assert (
                head_params["d_in"]
                == model.ftt.transformer.blocks[-1].ffn_normalization.normalized_shape[
                    0
                ]
            ), "Check the input of the head corresponds to the last one of the transformer"
        else:
            head_params["d_in"] = model.ftt.transformer.blocks[
                -1
            ].ffn_normalization.normalized_shape[0]
        if "batch_norm" in head_params.keys():
            assert (
                head_params["batch_norm"] == "layer_norm"
            ), "Check the head has set a layer_norm parameter"
        else:
            head_params["batch_norm"] = "layer_norm"
        if "pre_norm" in head_params.keys():
            assert (
                head_params["pre_norm"] == True
            ), "Check the input of the head corresponds to the last one of the neural network block"
        else:
            head_params["pre_norm"] = True
        head = HeadSelectiveNet(**head_params)
        model.ftt.transformer.head = head
    return model


def get_datatype(training_set):
    """
    :param training_set:
    :return:
    """
    if hasattr(training_set, "classes"):
        if type(training_set) == ImgFolder:
            tab = False
        else:
            tab = True
    elif hasattr(training_set, "datasets"):
        if type(training_set.datasets[1]) == ImgFolder:
            tab = False
        elif type(training_set.datasets[1]) == TabularDataset:
            tab = True
    elif hasattr(training_set, "dataset"):
        if hasattr(training_set.dataset, "classes"):
            if type(training_set.dataset) == ImgFolder:
                tab = False
            elif type(training_set.dataset) == TabularDataset:
                tab = True
        elif hasattr(training_set.dataset, "datasets"):
            if type(training_set.dataset.datasets[1]) == ImgFolder:
                tab = False
            elif type(training_set.dataset.datasets[1]) == TabularDataset:
                tab = True
    else:
        tab = None
    return tab


def train(
    model: nn.Module,
    device: str,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: str,
    train_dl: torch.utils.data.DataLoader,
    lamda: int = 32,
    alpha: float = 0.5,
    coverage: float = 0.9,
    beta: float = 0.01,
    td: bool = True,
    gamma: float = 0.5,
    epochs_lr: list = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299],
    verbose: bool = True,
    seed: int = 42,
):
    """
    :param model:
    :param device:
    :param epochs:
    :param optimizer:
    :param criterion:
    :param train_dl:
    :param lamda:
    :param alpha:
    :param coverage:
    :param beta:
    :param pretrain:
    :param reward:
    :param td:
    :param gamma:
    :param epochs_lr:
    :param momentum:
    :param verbose:
    :param seed:
    :return:
    """
    model.train()
    model.to(device)
    n = len(train_dl.dataset)
    tabular = get_datatype(train_dl.dataset)
    set_seed(seed)
    for epoch in range(1, epochs + 1):
        if td:
            if epoch in epochs_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= gamma
                    print("\n: lr now is: {}\n".format(param_group["lr"]))
        running_loss = 0
        if verbose:
            if tabular:
                with tqdm(train_dl, unit="batch") as tbatch:
                    for i, batch in enumerate(tbatch):
                        tbatch.set_description(
                            "Epoch {} - dev {}".format(epoch, device)
                        )
                        x_num, x_cat, y, indices = batch
                        x_num, x_cat, y = (
                            x_num.to(device),
                            x_cat.to(device),
                            y.float().to(device),
                        )
                        if len(y) == 1:
                            pass
                        else:
                            optimizer.zero_grad()
                            if "selnet" in criterion:
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                    print(
                                        "\n target coverage is {} \n".format(coverage)
                                    )
                                hg, aux = model.forward(x_num, x_cat)
                                loss1 = MSE_loss_selective(
                                    y, hg, lamda=lamda, c=coverage
                                )
                                loss2 = MSE_loss(y, aux)
                                loss = (alpha * loss1) + ((1 - alpha) * loss2)
                                if criterion == "selnet_em":
                                    loss3 = entropy_loss(hg[:, :-1])
                                    loss += beta * loss3
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            r = torch.cuda.memory_reserved(0)
                            a = torch.cuda.memory_allocated(0)
                            f = (r - a) / (1024**2)
                            tbatch.set_postfix(
                                loss=loss.item(),
                                average_loss=running_loss / (i + 1),
                                memory=f,
                            )
            else:
                with tqdm(train_dl, unit="batch") as tbatch:
                    for i, batch in enumerate(tbatch):
                        tbatch.set_description(
                            "Epoch {} - dev {}".format(epoch, device)
                        )
                        x, y, indices = batch
                        x, y = x.to(device), y.float().to(device)
                        if len(y) == 1:
                            pass
                        else:
                            optimizer.zero_grad()
                            if criterion in ["selnet", "selnet_em"]:
                                if (epoch == 1) & (i == 0):
                                    print("\n criterion is {} \n".format(criterion))
                                hg, aux = model.forward(x)
                                loss1 = MSE_loss_selective(
                                    y, hg, lamda=lamda, c=coverage
                                )
                                loss2 = MSE_loss(y, aux)
                                loss = (alpha * loss1) + ((1 - alpha) * loss2)
                                if criterion == "selnet_em":
                                    loss3 = entropy_loss(hg[:, :-1])
                                    loss += beta * loss3
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                            r = torch.cuda.memory_reserved(0)
                            a = torch.cuda.memory_allocated(0)
                            f = (r - a) / (1024**2)
                            tbatch.set_postfix(
                                loss=loss.item(),
                                average_loss=running_loss / (i + 1),
                                memory=f,
                            )
        else:
            if tabular:
                for i, batch in enumerate(train_dl):
                    x_num, x_cat, y, indices = batch
                    x_num, x_cat, y = (
                        x_num.to(device),
                        x_cat.to(device),
                        y.float().to(device),
                    )
                    if len(y) == 1:
                        pass
                    else:
                        optimizer.zero_grad()
                        if "selnet" in criterion:
                            if (epoch == 1) & (i == 0):
                                print("\n criterion is {} \n".format(criterion))
                            hg, aux = model.forward(x_num, x_cat)
                            loss1 = MSE_loss_selective(y, hg, lamda=lamda, c=coverage)
                            loss2 = MSE_loss(y, aux)
                            loss = (alpha * loss1) + ((1 - alpha) * loss2)
                            if criterion == "selnet_em":
                                loss3 = entropy_loss(hg[:, :-1])
                                loss += beta * loss3
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if (epoch % 50 == 0) & (i == len(train_dl) - 1):
                            print(running_loss / len(train_dl))
            else:
                for i, batch in enumerate(train_dl):
                    x, y, indices = batch
                    x, y = x.to(device), y.float().to(device)
                    if len(y) == 1:
                        pass
                    else:
                        optimizer.zero_grad()
                        if criterion in ["selnet", "selnet_em"]:
                            if (epoch == 1) & (i == 0):
                                print("\n criterion is {} \n".format(criterion))
                            hg, aux = model.forward(x)
                            loss1 = MSE_loss_selective(y, hg, lamda=lamda, c=coverage)
                            loss2 = MSE_loss(y, aux)
                            loss = (alpha * loss1) + ((1 - alpha) * loss2)
                            if criterion == "selnet_em":
                                loss3 = entropy_loss(hg[:, :-1])
                                loss += beta * loss3
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if (epoch % 50 == 0) & (i == len(train_dl)):
                            print(running_loss / len(train_dl))

    return model


def predict(
    model: nn.Module,
    device: str,
    dataloader: torch.utils.data.DataLoader,
    meta: str,
):
    model.eval()
    model.to(device)
    y_hat_ = []
    tab = get_datatype(dataloader.dataset)
    assert (
        dataloader.dataset.pred == True,
        "The Dataloader is not properly configured",
    )
    for i, batch in enumerate(dataloader):
        x_num, x_cat = batch
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        if "selnet" in meta:
            hg, aux = model(x_num, x_cat)
            y_hat_batch = hg[:, 0].detach().cpu().numpy().reshape(-1, 1)
        y_hat_.append(y_hat_batch)
    # TODO: implement for images (?)

    # elif tab == False:
    #     print("predicting")
    #     for i, batch in enumerate(dataloader):
    #         if i % 10 == 0:
    #             print("batch {} out of {}".format(i, len(dataloader)))
    #         x = batch
    #         x = x.to(device)
    #         if "selnet" in meta:
    #             hg, aux = model(x)
    #             y_hat_batch = (hg[:, :-1]
    #                 .detach()
    #                 .cpu()
    #                 .numpy().reshape(-1,1)
    #             )
    #         y_hat_.append(y_hat_batch)
    return np.vstack(y_hat_)


def predict_conf(
    model: nn.Module, device: str, dataloader: torch.utils.data.DataLoader, meta: str
):
    model.eval()
    model.to(device)
    sel_ = []
    tab = get_datatype(dataloader.dataset)
    # if tab:
    for i, batch in enumerate(dataloader):
        x_num, x_cat = batch
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        if "selnet" in meta:
            hg, aux = model(x_num, x_cat)
            sel_batch = hg[:, -1].detach().cpu().numpy().reshape(-1, 1)
        sel_.append(sel_batch)
    # TODO: implement for images (?)

    # elif tab == False:
    #     for i, batch in enumerate(dataloader):
    #         x, y, indices = batch
    #         x, y = x.to(device), y.float().to(device)
    #         if "selnet" in meta:
    #             hg, aux = model(x)
    #             if meta in ["selnet", "selnet_em"]:
    #                 sel_batch = hg[:, -1].detach().cpu().numpy().reshape(-1, 1)
    #             elif meta in ["selnet_em_sr", "selnet_sr"]:
    #                 sel_batch = (
    #                     torch.max(
    #                         torch.nn.functional.softmax(hg[:, :-1], dim=1), dim=1
    #                     )[0]
    #                     .detach()
    #                     .cpu()
    #                     .numpy()
    #                     .reshape(-1, 1)
    #                 )
    #         sel_.append(sel_batch)
    return np.vstack(sel_).flatten()


def tabular_model(model_type, x_num, cat_dim, meta, body_dict):
    """
    Function to build a tabular neural network
    :param model_type:
    :param train_set:
    :param meta:
    :param body_dict:
    :return:
    """
    body = copy.deepcopy(body_dict)
    body["d_in"] = x_num
    body["cat_cardinalities"] = cat_dim
    body["d_out"] = 1
    if model_type == "TabFTTransformer":
        if meta in ["selnet", "selnet_em"]:
            head = {
                "main_body": "transformer",
                "d_out": 1,
                "pre_norm": True,
            }
            model = buildSelNet(
                model_type="TabFTTransformer", body_params=body, head_params=head
            )
        else:
            model = TabFTTransformer(**body)
    elif model_type == "TabResnet":
        if meta in ["selnet", "selnet_em"]:
            head = {"main_body": "resnet", "d_out": 1, "pre_norm": True}
            model = buildSelNet(
                model_type="TabResnet", body_params=body, head_params=head
            )
        else:
            model = TabCatResNet(**body)
    else:
        raise NotImplementedError("The model architecture is not implemented yet.")
    return model
