from argparse import Namespace

from torchvision.datasets import CIFAR10
import numpy as np
import torch
import utils

DEFAULT_ARGS=Namespace(
    dataset='cifar10',
    data='data',
    cutout=None,
    cutout_length=None,
    train_portion=0.9,
    batch_size=256,
    evaluate_batch_size=256,
)


def load_dataset(args=DEFAULT_ARGS):
    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_length if args.cutout else None)
        train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        test_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.evaluate_batch_size,
            shuffle=False, pin_memory=True, num_workers=2)
    else:
        raise NotImplementedError("Temporary not used.")
    return train_queue, valid_queue, test_queue