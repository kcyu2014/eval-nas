
"""
Contains the training procedures.

Usually can be captured as a function, independently from the sampling procedure.

"""
import logging

import torch
import torch.nn as nn
import search_policies.cnn.darts_policy.utils as darts_utils
import search_policies.cnn.nao_policy.utils as nao_utils
import search_policies.cnn.utils
import utils


def _summarize_shared_train(curr_step, total_loss, raw_total_loss, acc=0, acc_5=0, lr=0.0, epoch_steps=1, writer=None):
    """Logs a set of training steps."""
    cur_loss = utils.to_item(total_loss) / epoch_steps
    cur_raw_loss = utils.to_item(raw_total_loss) / epoch_steps

    logging.info(f'| step {curr_step:3d} '
                 f'| lr {lr:4.2f} '
                 f'| raw loss {cur_raw_loss:.2f} '
                 f'| loss {cur_loss:.2f} '
                 f'| acc {acc:8.2f}'
                 f'| acc-5 {acc_5: 8.2f}')

    # Tensorboard
    if writer is not None:
        writer.scalar_summary('shared/loss',
                              cur_loss,
                              epoch_steps)
        writer.scalar_summary('shared/accuracy',
                              acc,
                              epoch_steps)


def darts_train_model(train_queue, valid_queue, model, criterion, optimizer, lr, args, architect, sampler=None):
    objs = search_policies.cnn.utils.AverageMeter()
    top1 = search_policies.cnn.utils.AverageMeter()
    top5 = search_policies.cnn.utils.AverageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        # IPython.embed()
        # sampler
        if args.debug and step > 10:
            logging.warning('Testing only. Break after 10 batches.')
            break

        if sampler:
            model = sampler(model, architect, args)
            # print("Activate sampler")

        n = input.size(0)
        # TODO check if this is correct, to put the data on cuda?
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

        optimizer.zero_grad()
        logits, _ = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = search_policies.cnn.utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            _summarize_shared_train(step, objs.avg, objs.avg, top1.avg, top5.avg, lr)

    return top1.avg, objs.avg


def darts_model_validation(valid_queue, model, criterion, args, verbose=True):
    objs = search_policies.cnn.utils.AverageMeter()
    top1 = search_policies.cnn.utils.AverageMeter()
    top5 = search_policies.cnn.utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break
            input = input.cuda()
            target = target.cuda()
            # target = target.cuda(async=True)
            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = search_policies.cnn.utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0 and step > 0 and verbose:
                logging.info('valid | step %03d | loss %e | acc %f | acc-5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def nao_train_model(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion, args):
    """
    training model procedure in NAO training search.
    :param train_queue:
    :param model:
    :param optimizer:
    :param global_step:
    :param arch_pool: architecture pooling defined in NAO
    :param arch_pool_prob: architecture prob.
    :param criterion:
    :param args: argument
    :return:
    """
    utils = nao_utils
    objs = search_policies.cnn.utils.AverageMeter()
    top1 = search_policies.cnn.utils.AverageMeter()
    top5 = search_policies.cnn.utils.AverageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits, aux_logits = model(input, arch, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if (step+1) % 100 == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch[0] + arch[1])))

    return top1.avg, objs.avg, global_step


def nao_model_validation(valid_queue, model, arch_pool, criterion):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            # for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()

            logits, _ = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)

            prec1, prec5 = nao_utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1.data / 100)

            if (i + 1) % 100 == 0:
                logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', ' '.join(map(str, arch[0] + arch[1])), loss,
                             prec1, prec5)

    return valid_acc_list


def nao_model_validation_nasbench(valid_queue, model, arch_pool, criterion):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            # for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()

            logits, _ = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)

            prec1, prec5 = nao_utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1.data / 100)

            if (i + 1) % 100 == 0:
                logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', ' '.join(map(str, arch[0] + arch[1])), loss,
                             prec1, prec5)

    return valid_acc_list


