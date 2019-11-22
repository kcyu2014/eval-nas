import IPython
from torch import nn as nn

import search_policies.cnn.utils
from search_policies.cnn.darts_policy import utils as darts_utils
from .train_search_procedure import _summarize_shared_train


def fairnas_train_model_v1(train_queue, valid_queue, model, criterion, optimizer, lr, args, architect,
                        topology_sampler=None,
                        op_sampler=None,
                        ):
    """
    Implement the
    :param train_queue:
    :param valid_queue:
    :param model:
    :param criterion:
    :param optimizer:
    :param lr:
    :param args:
    :param architect:
    :param topology_sampler:
    :param op_sampler:
    :return:
    """
    # sampler is the very different.
    objs = search_policies.cnn.utils.AverageMeter()
    top1 = search_policies.cnn.utils.AverageMeter()
    top5 = search_policies.cnn.utils.AverageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        if args.debug:
            if step > 10:
                print("Break after 10 batch")
                break
        # For each batch, sample a topology architecture first.
        if topology_sampler:
            model = topology_sampler(model, architect, args)

        n = input.size(0)
        input = input.cuda().requires_grad_()
        target = target.cuda()

        # this is a fairly simple step function logic. update the architecture in each step, before updating the
        # weight itself.
        if architect:
            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.requires_grad_().cuda()
            target_search = target_search.cuda()
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        loss = 0.0
        logits = 0.0
        total_model = 0
        optimizer.zero_grad()
        for _model in iter(op_sampler(model, architect, args)):
            # IPython.embed(header='check logits.')
            _logits, _ = _model(input)
            logits += _logits
            _loss = criterion(_logits, target)
            _loss.backward()
            loss += _loss.item()
            total_model += 1

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = search_policies.cnn.utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss / total_model, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            _summarize_shared_train(step, objs.avg, objs.avg, top1.avg, top5.avg, lr)

    return top1.avg, objs.avg


def fairnas_train_model_v2(train_queue, valid_queue, model, criterion, optimizer, lr, args, architect,
                        topology_sampler=None,
                        op_sampler=None,
                        ):
    """
    FairNAS with topology and operation fairness.

    :param train_queue:
    :param valid_queue:
    :param model:
    :param criterion:
    :param optimizer:
    :param lr:
    :param args:
    :param architect:
    :param topology_sampler:
    :param op_sampler:
    :return:
    """
    # sampler is the very different.
    assert topology_sampler is not None, 'must pass a topology sampler here.'
    objs = search_policies.cnn.utils.AverageMeter()
    top1 = search_policies.cnn.utils.AverageMeter()
    top5 = search_policies.cnn.utils.AverageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        if args.debug:
            if step > 10:
                print("Break after 10 batch")
                break
        # For each batch, sample a topology architecture first
        # if topology_sampler:
        for topo_model in iter(topology_sampler(model, architect, args)):
            n = input.size(0)
            input = input.cuda().requires_grad_()
            target = target.cuda()

            # this is a fairly simple step function logic. update the architecture in each step, before updating the
            # weight itself.
            if architect:
                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_queue))
                input_search = input_search.requires_grad_().cuda()
                target_search = target_search.cuda()
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            loss = 0.0
            logits = 0.0
            total_model = 0
            optimizer.zero_grad()
            for _model in iter(op_sampler(topo_model, architect, args)):
                # IPython.embed(header='check logits.')
                _logits, _ = _model(input)
                logits += _logits
                _loss = criterion(_logits, target)
                _loss.backward()
                loss += _loss.item()
                total_model += 1

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1, prec5 = search_policies.cnn.utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss / total_model, n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                _summarize_shared_train(step, objs.avg, objs.avg, top1.avg, top5.avg, lr)

    return top1.avg, objs.avg

