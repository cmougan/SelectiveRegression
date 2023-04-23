import torch
import torch.nn.functional as F


def MSE_loss(y_true, outputs):
    """
    :param y_true: Pytorch Tensor
        The tensor with actual labaels.
    :param outputs: Pytorch Tensor
        The predicted values from auxiliary head.
    :return: Pytorch tensor
        The cross entropy loss for auxiliary head.
    """
    if len(y_true.shape) == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    crit = torch.nn.MSELoss()
    loss = crit(outputs, y_true)
    return loss


def MSE_loss_selective(y_true, hg, lamda=32, c=0.9):
    """

    :param y_true:
    :param hg:
    :param lamda:
    :param c:
    :return:
    """
    # hg = hg.float()
    n_batch = hg.shape[0]
    if len(y_true.shape) == 1:
        y_true = torch.unsqueeze(y_true, dim=-1)
    if c == 1:
        selected = n_batch
    else:
        selected = torch.sum(hg[:, -1]) + 0.000001
    selection = torch.unsqueeze(hg[:, -1], dim=-1)
    pred = torch.unsqueeze(hg[:, 0], dim=1)
    ind_loss = ((pred - y_true) ** 2) * selection
    f_l = (
        torch.sum(ind_loss) / (selected)
        + lamda * (max(0, c - (selected / n_batch))) ** 2
    )
    return f_l


def entropy_loss(output):
    """

    :param output:
    :return:
    """
    el = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
    loss = -1.0 * (el.sum())
    return loss
