import torch
from torch import nn
import torch.distributed as dist


class UncertaintyLoss(nn.Module):
    def __init__(self):
        """
        initialize loss weights.
        """
        super(UncertaintyLoss, self).__init__()
        params = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], requires_grad=True)
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


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size
        len = z_i.size(1)
        dim = z_i.size(-1)

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        total_loss = 0

        for index in range(0, z.size(1)):
            output = z[:, index, :]

            sim = self.similarity_f(output.unsqueeze(1), output.unsqueeze(0)) / self.temperature

            sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
            sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

            # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N

            total_loss += loss

        return total_loss / z.size(1)