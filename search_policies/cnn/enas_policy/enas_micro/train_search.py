import os
import sys
import time
import random
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

import search_policies.cnn.utils
from .data.data import get_loaders
import search_policies.cnn.enas_policy.enas_micro.utils as utils
from .micro_child import CNN
from .micro_controller import Controller


def get_enas_microcnn_parser():
    # backup, the same as original ENAS code. should not modify.
    parser = argparse.ArgumentParser("enas-cifar-10")
    parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=160, help='batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--child_lr_max', type=float, default=0.05)
    parser.add_argument('--child_lr_min', type=float, default=0.0005)
    parser.add_argument('--child_lr_T_0', type=int, default=10)
    parser.add_argument('--child_lr_T_mul', type=int, default=2)
    parser.add_argument('--child_num_layers', type=int, default=6)
    parser.add_argument('--child_out_filters', type=int, default=20)
    parser.add_argument('--child_num_branches', type=int, default=5)
    parser.add_argument('--child_num_cells', type=int, default=5)
    parser.add_argument('--child_use_aux_heads', type=bool, default=False)

    parser.add_argument('--controller_lr', type=float, default=0.0035)
    parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
    parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
    parser.add_argument('--controller_train_steps', type=int, default=30)

    parser.add_argument('--lstm_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--lstm_keep_prob', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=5.0)

    parser.add_argument('--entropy_weight', type=float, default=0.0001)
    parser.add_argument('--bl_dec', type=float, default=0.99)
    return parser


# args = parser.parse_args()

# args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# CIFAR_CLASSES = 10
# baseline = None
# epoch = 0

# transfered into ENASMicroCNNSearchPolicy
# def main():
#     if not torch.cuda.is_available():
#         logging.info('no gpu device available')
#         sys.exit(1)
#
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.cuda.set_device(args.gpu)
#     torch.backends.cudnn.benchmark = True
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     logging.info('gpu device = %d' % args.gpu)
#     logging.info("args = %s", args)
#
#     model = CNN(args)
#     model.cuda()
#
#     controller = Controller(args)
#     controller.cuda()
#     baseline = None
#
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         args.child_lr_max,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay,
#     )
#
#     controller_optimizer = torch.optim.Adam(
#         controller.parameters(),
#         args.controller_lr,
#         betas=(0.1,0.999),
#         eps=1e-3,
#     )
#
#     train_loader, reward_loader, valid_loader = get_loaders(args)
#
#     scheduler = utils.LRScheduler(optimizer, args)
#
#     for epoch in range(args.epochs):
#         lr = scheduler.update(epoch)
#         logging.info('epoch %d lr %e', epoch, lr)
#
#         # training
#         train_acc = train(train_loader, model, controller, optimizer)
#         logging.info('train_acc %f', train_acc)
#
#         train_controller(reward_loader, model, controller, controller_optimizer)
#
#         # validation
#         valid_acc = infer(valid_loader, model, controller)
#         logging.info('valid_acc %f', valid_acc)
#
#         utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_loader, model, controller, optimizer, args):
    total_loss = search_policies.cnn.utils.AverageMeter()
    total_top1 = search_policies.cnn.utils.AverageMeter()

    for step, (data, target) in enumerate(train_loader):
        if args.debug and step > 10:
            logging.warning("Breaking after 10 steps")
            break

        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        controller.eval()
        dag, _, _ = controller()

        logits, _ = model(data, dag)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, total_loss.avg, total_top1.avg)

    return total_top1.avg


def train_controller(reward_loader, model, controller, controller_optimizer, baseline, args):
    total_loss = search_policies.cnn.utils.AverageMeter()
    total_reward = search_policies.cnn.utils.AverageMeter()
    total_entropy = search_policies.cnn.utils.AverageMeter()

    #for step, (data, target) in enumerate(reward_loader):
    for step in range(300):
        if args.debug and step > 10:
            logging.warning("Debugging here break after 10.")
            break
        data, target = reward_loader.next_batch()
        model.eval()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        controller.train()
        dag, log_prob, entropy = controller()

        with torch.no_grad():
            logits, _ = model(data, dag)
            reward = utils.accuracy(logits, target)[0]

        if args.entropy_weight is not None:
            reward += args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = log_prob * (reward - baseline)
        loss = loss.sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)

        if step % args.report_freq == 0:
            #logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
            logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, baseline.item())
            #tensorboard.add_scalar('controller/loss', loss, epoch)
            #tensorboard.add_scalar('controller/reward', reward, epoch)
            #tensorboard.add_scalar('controller/entropy', entropy, epoch)
    return baseline


def infer(valid_loader, model, controller, args, return_perf=False):
    total_loss = search_policies.cnn.utils.AverageMeter()
    total_top1 = search_policies.cnn.utils.AverageMeter()
    model.eval()
    controller.eval()
    archs_pool = []
    archs_perf = []

    with torch.no_grad():
        for step in range(10):
            data, target = valid_loader.next_batch()
            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            logits, _ = model(data, dag)
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            #if step % args.report_freq == 0:
            logging.info('valid %03d %e %f', step, loss.item(), prec1.item())
            logging.info('normal cell %s', str(dag[0]))
            logging.info('reduce cell %s', str(dag[1]))
            archs_pool.append(dag)
            archs_perf.append(prec1.item())

    if return_perf:
        sorted_indices = np.argsort(archs_perf)[::-1]
        archs_pool = [archs_pool[i] for i in sorted_indices]
        archs_perf = [archs_perf[i] for i in sorted_indices]
        return total_top1.avg, archs_pool, archs_perf
    else:
        return total_top1.avg


