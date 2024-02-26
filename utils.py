import torch
from torch import nn


class UncertaintyLoss(nn.Module):
    def __init__(self):
        """
        initialize loss weights.
        """
        super(UncertaintyLoss, self).__init__()
        params = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, loss_list):
        """
        forward pass of loss.
        :param loss_list: the provided loss list.
        :return: the task-uncertainty loss.
        """
        loss_sum = 0
        loss1, loss2, loss3 = loss_list
        loss1 = loss1 / (self.params[0] ** 2) + torch.log(self.params[0])
        loss2 = loss2 / (self.params[1] ** 2) + torch.log(self.params[1])
        loss3 = loss3 / (2 * (self.params[2] ** 2)) + torch.log(self.params[2])
        loss_sum += loss1 + loss2 + loss3
        return loss_sum
