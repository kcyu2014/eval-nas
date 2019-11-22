"""
Collection of CNN utils, that are duplicated by others.

"""

from collections import namedtuple

import torch

Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def clip_grad_norm(grad_tensors, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    Modify from the original ones, just to clip grad directly.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        grad_tensors (Iterable[Tensor] or Tensor): an iterable of grad Tensors
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    grad_tensors = list(filter(lambda p: p is not None, grad_tensors))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == 'inf':
        total_norm = max(p.data.abs().max() for p in grad_tensors)
    else:
        total_norm = 0
        for p in grad_tensors:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in grad_tensors:
            p.data.mul_(clip_coef)
    return total_norm


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res