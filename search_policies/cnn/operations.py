# define some operations
import IPython
import torch.nn.functional as F
import torch.nn as nn
import torch


def gumbel_softmax(logits, temperature, gumbel_dist):
    # IPython.embed()
    y = logits + gumbel_dist.sample(logits.size()).squeeze().cuda()
    return F.softmax(y / temperature, dim=-1)


class WSBNFull(nn.Module):

    def __init__(self, num_possible_inputs, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Based on WSBN from NAO, but extend to running mean and var.

        :param num_possible_inputs:
        :param num_features:
        :param eps:
        :param momentum:
        :param affine:
        """
        super(WSBNFull, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.ParameterList(
                [nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for idx in range(num_possible_inputs):
            self.register_buffer('running_mean_{}'.format(idx), torch.zeros(num_features))
            self.register_buffer('running_var_{}'.format(idx), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_possible_inputs):
            getattr(self, f'running_mean_{i}').fill_(1)
            # print(f'running_mean_{i} init ', getattr(self, f'running_mean_{i}').mean())
            getattr(self, f'running_var_{i}').zero_()
            # print(f'running_var_{i} init ', getattr(self, f'running_var_{i}').mean())
            if self.affine:
                self.weight[i].data.fill_(1)
                self.bias[i].data.zero_()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True
        # diff = getattr(self, f'running_mean_{x_id}').sum().item()
        return F.batch_norm(
            x, getattr(self, f'running_mean_{x_id}'), getattr(self, f'running_var_{x_id}'),
            self.weight[x_id], self.bias[x_id],
            training, self.momentum, self.eps)
        # diff -= getattr(self, f'running_mean_{x_id}').sum().item()
        # return o

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__))
