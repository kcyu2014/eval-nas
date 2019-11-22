import logging
import operator
import os
import shutil

import gc
import sys

import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.utils

import search_policies.cnn.utils
import utils as project_utils
import torchvision.datasets as dset

from collections import namedtuple, deque, OrderedDict
from tensorboardX import SummaryWriter

from search_policies.cnn.cnn_general_search_policies import CNNSearchPolicy
from search_policies.cnn.enas_policy.enas_micro.data.data import RepeatedDataLoader, get_loaders
from search_policies.cnn.procedures.evaluate_procedure import evaluate_enas_with_dag
from search_policies.cnn.search_space.nas_bench.util import change_model_spec
from visualization.process_data import tensorboard_summarize_list

from .enas_micro.micro_controller import Controller as MicroController
from .enas_micro.micro_child import CNN as ENASWSCNN
from .models.model_search_nasbench import NasBenchNetSearchENAS
from .models.controller import ControllerNASbench as MicroControllerNasbench
from ..search_space.nas_bench.model import NasBenchNet
from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import ModelSpec_v2
from ..random_policy.nasbench_weight_sharing_policy import NasBenchWeightSharingPolicy
from ..procedures.train_search_procedure import nao_model_validation_nasbench
import search_policies.cnn.enas_policy.utils_for_nasbench as enas_nasbench_utils
import search_policies.cnn.enas_policy.enas_micro.utils as enas_utils
from search_policies.cnn.enas_policy.enas_micro.train_search import train, train_controller, infer

Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class ENASMicroCNNSearchPolicy(CNNSearchPolicy):
    """Wrapper of the original ENAS search space. """
    def __init__(self, args, enas_args):
        super(ENASMicroCNNSearchPolicy, self).__init__(args, sub_dir_path='enas-search-{}'.format(args.seed))
        self.baseline = None
        self.enas_args = enas_args
    # override these models.
    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
        pass
    def op_sampler(self, model, architect, args):
        pass
    def random_sampler(self, model, architect, args):
        pass
    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        pass

    def run(self):
        args = self.enas_args
        args.debug = self.args.debug
        project_utils.torch_random_seed(args.seed)
        logging.info('gpu device = %d' % args.gpu)
        logging.info("args = %s", args)
        model = ENASWSCNN(args)
        model.cuda()
        controller = MicroController(args)
        controller.cuda()
        baseline = None
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        controller_optimizer = torch.optim.Adam(
            controller.parameters(),
            args.controller_lr,
            betas=(0.1, 0.999),
            eps=1e-3,
        )
        train_loader, reward_loader, valid_loader, test_loader = get_loaders(args)
        scheduler = enas_utils.LRScheduler(optimizer, args)
        for epoch in range(args.epochs):
            lr = scheduler.update(epoch)
            logging.info('epoch %d lr %e', epoch, lr)
            # training
            train_acc = train(train_loader, model, controller, optimizer, args)
            logging.info('train_acc %f', train_acc)
            baseline = train_controller(reward_loader, model, controller, controller_optimizer, baseline, args)
            # validation
            valid_acc = infer(valid_loader, model, controller, args)
            logging.info('valid_acc %f', valid_acc)

            if epoch % 30 == 0:
                # evaluation
                logging.info('Evaluation at epoch {}'.format(epoch))
                valid_acc, arch_pool, arch_perf = evaluate_enas_with_dag(model, controller,
                                                                         test_loader, args, return_perf=True)
                with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as f:
                    with open(os.path.join(self.exp_dir, 'arch_pool.perf'), 'w') as fb:
                        for arch, perf in zip(arch_pool, arch_perf):
                            # logging.info("normal cell {}".format(arch[0].tolist()))
                            # logging.info("reduce cell {}".format(arch[1].tolist()))
                            # logging.info("accuracy {}".format(perf))
                            f.write('[{}, {}] \n'.format(arch[0].tolist(), arch[1].tolist()))
                            fb.write('{}\n'.format(perf))
                # backup the results
                shutil.copy(os.path.join(self.exp_dir, 'arch_pool'),
                            os.path.join(self.exp_dir, 'arch_pool') + '.{}'.format(epoch))
                shutil.copy(os.path.join(self.exp_dir, 'arch_pool.perf'),
                            os.path.join(self.exp_dir, 'arch_pool.perf') + '.{}'.format(epoch))
                enas_utils.save(model, os.path.join(self.exp_dir, 'weights.pt'))
        # best choosed dag
        bestdag, _, _ = controller.forward()
        return -1, bestdag


class ENASNasBenchSearch(NasBenchWeightSharingPolicy):

    top_K_complete_evaluate = 200
    baseline = None
        
    """
    
    Becasue ENAS has a feature, that only sampler samples the new architecture (in this sence, very similar to 
    NAO sampler, i.e. NAO-infer = ENAS sample.
    SO I could keep NAO's code not touch, only change the logic of training such sampler.
    Then, during evaluation, use the child-arch-pool as the same.
    Only difference is, training child network is by sampling but not random an arch pool model.
    Arch pool is solely for evaluation.
    
    """
    def __init__(self, args):
        return super(ENASNasBenchSearch, self).__init__(args, full_dataset=False)

    # EMAS_search_config
    def wrap_enas_search_config(self):
        for k, v in self.args.enas_search_config.__dict__.items():
            self.args.__dict__[k] = v

    # The same as NAO one.
    def initialize_run(self):
        """
        TODO This is the same as NAO one.
        :return:
        """
        args = self.args
        utils = project_utils
        if not self.args.continue_train:
            self.sub_directory_path = 'WeightSharingNasBenchNetRandom-{}_SEED_{}'.format(self.args.save, self.args.seed)
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            utils.create_exp_dir(self.exp_dir)

        if self.args.visualize:
            self.viz_dir_path = utils.create_viz_dir(self.exp_dir)

        if self.args.tensorboard:
            self.tb_dir = self.exp_dir
            tboard_dir = os.path.join(self.args.tboard_dir, self.sub_directory_path)
            self.writer = SummaryWriter(tboard_dir)

        if self.args.debug:
            torch.autograd.set_detect_anomaly(True)

        self.nasbench = self.search_space.nasbench

        # Set logger.
        self.logger = utils.get_logger(
            "train_search",
            file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')),
            level=logging.INFO if not args.debug else logging.DEBUG
        )
        logging.info(f"setting random seed as {args.seed}")
        utils.torch_random_seed(args.seed)
        logging.info('gpu number = %d' % args.gpus)
        logging.info("args = %s", args)

        criterion = nn.CrossEntropyLoss().cuda()
        eval_criterion = nn.CrossEntropyLoss().cuda()
        self.eval_loss = eval_criterion

        train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_length if args.cutout else None)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.enas_search_config.ratio * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.enas_search_config.child_eval_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.evaluate_batch_size,
            shuffle=False, pin_memory=True, num_workers=8)

        repeat_valid_loader = RepeatedDataLoader(valid_queue)
        return train_queue, valid_queue, test_queue, repeat_valid_loader, criterion, eval_criterion

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        args = self.args
        # over ride the model_fn
        self.train_fn = None
        self.eval_fn = nao_model_validation_nasbench

        if self.args.search_space == 'nasbench':
            self.model_fn = NasBenchNetSearchENAS
            self.fixmodel_fn = NasBenchNet
            model = self.model_fn(args)
            utils = enas_nasbench_utils
            enas = MicroControllerNasbench(args=args)
        else:
            utils = enas_utils
            self.model_fn = ENASWSCNN
            self.fixmodel_fn = None
            model = self.model_fn(args)

            enas = MicroController(args)

        enas = enas.cuda()
        logging.info("ENAS RNN sampler param size = %fMB", project_utils.count_parameters_in_MB(enas))
        self.controller = enas

        if args.gpus > 0:
            if self.args.gpus == 1:
                model = model.cuda()
                self.parallel_model = model
            else:
                self.model = model
                self.parallel_model = nn.DataParallel(self.model).cuda()
                # IPython.embed(header='checking replicas and others.')
        else:
            self.parallel_model = model
        # rewrite the pointer
        model = self.parallel_model
        logging.info("param size = %fMB", project_utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )

        controller_optimizer = torch.optim.Adam(
            enas.parameters(),
            args.controller_lr,
            betas=(0.1, 0.999),
            eps=1e-3,
        )

        # scheduler as Cosine.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.child_lr_min)
        return model, optimizer, scheduler, enas, controller_optimizer

    def run(self):
        # process UTILS and args
        args = self.args
        self.utils = enas_nasbench_utils if args.search_space == 'nasbench' else enas_utils
        utils = self.utils
        self.enas_search_config = args.enas_search_config
        self.wrap_enas_search_config()

        # build the functions.
        train_queue, valid_queue, test_queue, repeat_valid_queue, train_criterion, eval_criterion = self.initialize_run()
        model, optimizer, scheduler, enas, enas_optimizer = self.initialize_model()
        fitness_dict = {}
        self.child_arch_pool = deque()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.controller_optimizer = enas_optimizer
        # Train child model
        child_arch_pool = self.child_arch_pool
        # IPython.embed(header='Test sampler and map.')
        step = 0
        # Begin the full loop
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, step = self.child_train(train_queue, model, optimizer, step, enas, train_criterion)
            if self.writer:
                self.writer.add_scalar(f'train/loss', train_obj, epoch)
                self.writer.add_scalar(f'train/top_1_acc', train_acc, epoch)
            logging.info('train_acc %f', train_acc)
            # train the controller
            if epoch % args.controller_train_every == 0:
                self.controller_train(repeat_valid_queue, model, enas, enas_optimizer, epoch)

            # end of normal training loop.
            if not epoch % args.save_every_epoch == 0:
                continue
            # Saving the current performance of arch pool into the list.
            # Evaluate seed archs
            valid_accuracy_list = self.child_valid(repeat_valid_queue, model, child_arch_pool, eval_criterion)

            # Output archs and evaluated error rate
            old_archs = child_arch_pool
            old_archs_perf = valid_accuracy_list
            old_archs_sorted_indices = np.argsort(old_archs_perf)[::-1]
            old_archs = [old_archs[i] for i in old_archs_sorted_indices]
            old_archs_perf = [old_archs_perf[i] for i in old_archs_sorted_indices]
            with open(os.path.join(self.exp_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
                with open(os.path.join(self.exp_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                    with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as fa_latest:
                        with open(os.path.join(self.exp_dir, 'arch_pool.perf'), 'w') as fp_latest:
                            for arch, perf in zip(old_archs, old_archs_perf):
                                arch = self.process_archname(arch)
                                fa.write('{}\n'.format(arch))
                                fa_latest.write('{}\n'.format(arch))
                                fp.write('{}\n'.format(perf))
                                fp_latest.write('{}\n'.format(perf))

            # Generate new archs for robust evaluation.
            new_archs = []
            max_step_size = 50
            predict_step_size = 0
            enas.eval()
            while len(new_archs) < args.controller_new_arch:
                predict_step_size += 1
                logging.info('Generate new architectures with step size %d', predict_step_size)

                for _ in range(args.controller_sample_batch):
                    _, _, _, arch = enas()
                    if arch not in new_archs:
                        self.child_arch_pool_append(arch)
                        new_archs.append(arch)
                    if len(new_archs) >= args.controller_new_arch:
                        break
                logging.info('%d new archs generated now', len(new_archs))
                if predict_step_size > max_step_size:
                    break
            # IPython.embed()
            num_new_archs = len(new_archs)
            logging.info("Generate %d new archs", num_new_archs)
            # replace bottom archs
            random_archs = self.generate_new_arch(args.controller_random_arch)
            for a in random_archs:
                self.child_arch_pool_append(a)

            logging.info("Totally %d architectures now to train", len(self.child_arch_pool))
            with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as f:
                for arch in self.child_arch_pool:
                    f.write('{}\n'.format(self.process_archname(arch)))
            # save model
            project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
            fitness_dict = self.evaluate(epoch, test_queue,
                                         fitnesses_dict=fitness_dict,
                                         arch_pool=child_arch_pool,
                                         train_queue=train_queue,
                                         criterion=eval_criterion)

            self.save_results(epoch, rank_details=True)
            # IPython.embed()
        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, return the best of training.
        # IPython.embed(header="Pause for nothing.")
        fitness_dict = self.evaluate(epoch, test_queue,
                                     fitnesses_dict=fitness_dict,
                                     arch_pool=child_arch_pool,
                                     train_queue=train_queue,
                                     criterion=eval_criterion)
        project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
        self.save_results(epoch, rank_details=True)
        ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
        best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
        return best_id, self.nasbench_model_specs[best_id]

    def child_train(self, train_queue, model, optimizer, global_step, enas, criterion):
        utils = self.utils
        objs = search_policies.cnn.utils.AverageMeter()
        top1 = search_policies.cnn.utils.AverageMeter()
        top5 = search_policies.cnn.utils.AverageMeter()
        model.train()
        for step, (input, target) in enumerate(train_queue):
            if self.args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break
            input = input.cuda().requires_grad_()
            target = target.cuda()

            optimizer.zero_grad()

            # sample an arch to train
            _, _, _, arch = enas()
            self.child_arch_pool_append(arch)
            arch_l = arch
            arch = self.process_arch(arch)
            logits, aux_logits = model(input, arch, global_step, bn_train=False)
            global_step += 1
            loss = criterion(logits, target)
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, target)
                loss += 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.child_grad_bound)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if (step + 1) % 100 == 0:
                logging.info('Train %03d loss %e top1 %f top5 %f', step + 1, objs.avg, top1.avg, top5.avg)
                logging.info('Arch: %s', self.process_archname(arch_l))

        return top1.avg, objs.avg, global_step

    def child_valid(self, valid_queue, model, arch_pool, criterion):
        valid_acc_list = []
        with torch.no_grad():
            model.eval()
            for i, arch in enumerate(arch_pool):
                # for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = valid_queue.next_batch()
                inputs = inputs.cuda()
                targets = targets.cuda()
                arch_l = arch
                arch = self.process_arch(arch)
                logits, _ = model(inputs, arch, bn_train=True)
                loss = criterion(logits, targets)

                prec1, prec5 = self.utils.accuracy(logits, targets, topk=(1, 5))
                valid_acc_list.append(prec1.data / 100)

                if (i + 1) % 100 == 0:
                    logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', self.process_archname(arch_l),
                                 loss, prec1, prec5)
        return valid_acc_list

    def controller_train(self, train_queue, model, controller, optimizer, epoch):
        baseline = self.baseline
        args = self.args
        utils = self.utils

        total_loss = search_policies.cnn.utils.AverageMeter()
        total_reward = search_policies.cnn.utils.AverageMeter()
        total_entropy = search_policies.cnn.utils.AverageMeter()
        model.eval()
        controller.train()

        for step in range(args.controller_train_steps):
            optimizer.zero_grad()
            loss = 0
            if args.debug and step > 3:
                print("debug, break after 3 controller step")
                break
            for ind in range(args.controller_num_aggregate):
                data, target = train_queue.next_batch()
                n = data.size(0)
                data = data.cuda()
                target = target.cuda()
                _, log_prob, entropy, arch = controller()
                arch_l = arch
                arch = self.process_arch(arch)
                with torch.no_grad():
                    logits, _ = model(data, arch)
                    reward = utils.accuracy(logits, target)[0]

                if args.entropy_weight is not None:
                    reward += args.entropy_weight * entropy

                log_prob = torch.sum(log_prob)
                if baseline is None:
                    baseline = reward
                baseline -= (1 - args.bl_dec) * (baseline - reward)

                _loss = log_prob * (reward - baseline)
                loss = loss + _loss.sum()
                total_reward.update(reward.item(), n)
                total_entropy.update(entropy.item(), n)
                total_loss.update(_loss.item(), n)

            loss = loss / args.controller_num_aggregate
            loss.backward()
            optimizer.step()

            if step % args.report_freq == 0:
                # logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
                logging.info('controller epoch %03d step %03d loss %e reward %f baseline %f',
                             epoch, step, total_loss.avg, total_reward.avg, baseline.item())

        if self.writer:
            self.writer.add_scalar('controller/loss', total_loss.avg, epoch)
            self.writer.add_scalar('controller/reward', total_reward.avg, epoch)
            self.writer.add_scalar('controller/entropy', total_entropy.avg, epoch)

        return total_loss.avg, total_reward.avg, total_entropy.avg

    def process_arch(self,arch):
        """ return model spec for nasbench. else return itself. """
        if self.args.search_space == 'nasbench':
            matrix, ops = enas_nasbench_utils.parse_arch_to_model_spec_matrix_op(arch, self.args.child_num_cells)
            model_spec = ModelSpec_v2(matrix, ops)
            return model_spec
        else:
            return arch

    def process_archname(self, arch):
        if self.args.search_space == 'nasbench':
            return ' '.join(map(str, arch))
        else:
            return ' '.join(map(str, arch[0] + arch[1]))

    def generate_new_arch(self, num_new):
        num_ops = 3 if self.args.search_space == 'nasbench' else 5
        child_arch_pool = self.utils.generate_arch(num_new, self.args.child_num_cells, num_ops)
        return child_arch_pool

    def evaluate(self, epoch, data_source, arch_pool=None, fitnesses_dict=None, train_queue=None, criterion=None):
        """
        Full evaluation of all possible models.
        :param epoch:
        :param data_source:
        :param fitnesses_dict: Store the model_spec_id -> accuracy
        :return:
        """

        fitnesses_dict = fitnesses_dict or {}
        total_avg_acc = 0
        total_avg_obj = 0

        # rank dict for the possible solutions
        model_specs_rank = {}
        model_specs_rank_before = {}
        queries = {}
        # as backup
        ind = 0
        eval_result = {}
        # let us sample 200 architecture to evaluate. # just keep the top K.
        clean_arch_pool = self.clean_arch_pool(arch_pool)[:self.top_K_complete_evaluate]

        while ind < len(clean_arch_pool):
            # get this id
            if self.args.debug and ind > 10:
                break

            arch = clean_arch_pool[ind]
            new_model_spec = self.process_arch(clean_arch_pool[ind])
            ind += 1  # increment this.

            try:
                model_spec_id = self.nasbench_hashs.index(new_model_spec.hash_spec())
            except Exception as e:
                logging.error(e)
                continue

            query = {'test accuracy':self.search_space.nasbench.perf_rank[model_spec_id]}

            # selecting the current subDAG in our DAG to train
            change_model_spec(self.parallel_model, new_model_spec)
            # Reset the weights.
            # evaluate before train
            self.logger.info('evaluate the model spec id: {}'.format(model_spec_id))
            _avg_val_acc, _avg_val_acc5, _avg_val_obj = self.child_test(data_source, self.parallel_model, arch, criterion=criterion)
            eval_result[model_spec_id] = _avg_val_acc, _avg_val_obj

            logging.info("Query: {}".format(query))
            # update the total loss.
            total_avg_acc += _avg_val_acc
            total_avg_obj += _avg_val_obj

            # saving the particle fit in our dictionaries
            fitnesses_dict[model_spec_id] = _avg_val_acc
            ms_hash = self.nasbench_hashs[model_spec_id]
            model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
                                              model_spec_id)
            queries[ms_hash] = query
            gc.collect()

        # save the ranking, according to their GENOTYPE but not particle id
        rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))

        self.ranking_per_epoch[epoch] = rank_gens
        self.eval_result[epoch] = eval_result
        # IPython.embed(header="Check evaluation result")

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                             f'hash: {elem[0]}, pos {pos}')

        if self.writer:
            # process data into list.
            accs_after, objs_after = zip(*eval_result.values())
            tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
            tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)

        return fitnesses_dict

    def process_nasbench(self):
        super(ENASNasBenchSearch, self).process_nasbench(only_hash=False)

    def child_test(self, test_queue, model, arch, criterion, verbose=True):
        utils = self.utils
        objs = search_policies.cnn.utils.AverageMeter()
        top1 = search_policies.cnn.utils.AverageMeter()
        top5 = search_policies.cnn.utils.AverageMeter()
        model.eval()
        # arch_l = arch
        arch = self.process_arch(arch)
        with torch.no_grad():
            for step, (input, target) in enumerate(test_queue):
                if self.args.debug:
                    if step > 10:
                        print("Break after 10 batch")
                        break
                input = input.cuda()
                target = target.cuda()
                logits, _ = model(input, arch, bn_train=False)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.args.report_freq == 0 and step > 0 and verbose:
                    logging.info('test | step %03d | loss %e | acc %f | acc-5 %f', step, objs.avg, top1.avg,
                                 top5.avg)

        return top1.avg, top5.avg, objs.avg

    def clean_arch_pool(self, arch_pool):
        new_arch_pool = []
        for i in arch_pool:
            if not i in new_arch_pool:
                new_arch_pool.append(i)
        return new_arch_pool

    def child_arch_pool_append(self, arch):
        """
        Append child arch pool. without duplication.
        :param arch: list of int [0, ,1,2,3...]
        :return:
        """
        if arch in self.child_arch_pool:
            self.child_arch_pool.remove(arch)
        self.child_arch_pool.append(arch)
        if len(self.child_arch_pool) > self.top_K_complete_evaluate:
            old_arch = self.child_arch_pool.popleft()
            logging.debug("Pop arch {} from pool".format(old_arch))


class ENASNasBenchGroundtruthPolicy(ENASNasBenchSearch):

    def child_train(self, train_queue, model, optimizer, global_step, enas, criterion):
        for step, (_, _) in enumerate(train_queue):
            if self.args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break

            # sample an arch to train
            _, _, _, arch_l = enas()
            try:

                arch = self.process_arch(arch_l)
                r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
            except ValueError as e:
                continue
            self.child_arch_pool_append(arch_l)
            global_step += 1

        return 0., 0., global_step

    def child_test(self, test_queue, model, arch, criterion, verbose=True):
        try:
            if not hasattr(arch, 'hash_spec'):
                arch = self.process_arch(arch)
            r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
        except ValueError as e:
            return 0., 0., 0.
        valid_acc = self.search_space.nasbench.perf_rank[r][0]
        return valid_acc, 0., 0.

    def child_valid(self, valid_queue, model, arch_pool, criterion):
        valid_acc_list = []

        for i, arch in enumerate(arch_pool):
            # for step, (inputs, targets) in enumerate(valid_queue):
            arch_l = arch
            arch = self.process_arch(arch)
            try:
                r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
                valid_acc = self.search_space.nasbench.perf_rank[r][0]
                valid_acc_list.append(valid_acc)
            except ValueError as e:
                valid_acc_list.append(0.0)

        return valid_acc_list

    def controller_train(self, train_queue, model, controller, optimizer, epoch):
        """ Override this controller, modify the ground-truth """
        baseline = self.baseline
        args = self.args
        utils = self.utils

        total_loss = search_policies.cnn.utils.AverageMeter()
        total_reward = search_policies.cnn.utils.AverageMeter()
        total_entropy = search_policies.cnn.utils.AverageMeter()
        model.eval()
        controller.train()

        for step in range(args.controller_train_steps):
            optimizer.zero_grad()
            loss = 0
            if args.debug and step > 3:
                print("debug, break after 3 controller step")
                break
            for ind in range(args.controller_num_aggregate):
                n = 1
                _, log_prob, entropy, arch = controller()
                arch_l = arch
                arch = self.process_arch(arch)
                try:
                    r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
                except Exception as e:
                    logging.error(e)
                    continue
                reward = self.search_space.nasbench.perf_rank[r][0]

                if args.entropy_weight is not None:
                    reward += args.entropy_weight * entropy

                log_prob = torch.sum(log_prob)
                if baseline is None:
                    baseline = reward
                baseline -= (1 - args.bl_dec) * (baseline - reward)

                _loss = log_prob * (reward - baseline)
                loss = loss + _loss.sum()
                total_reward.update(reward.item(), n)
                total_entropy.update(entropy.item(), n)
                total_loss.update(_loss.item(), n)

            loss = loss / args.controller_num_aggregate
            loss.backward()
            optimizer.step()

            if step % args.report_freq == 0:
                # logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
                logging.info('controller epoch %03d step %03d loss %e reward %f baseline %f',
                             epoch, step, total_loss.avg, total_reward.avg, baseline.item())

        if self.writer:
            self.writer.add_scalar('controller/loss', total_loss.avg, epoch)
            self.writer.add_scalar('controller/reward', total_reward.avg, epoch)
            self.writer.add_scalar('controller/entropy', total_entropy.avg, epoch)

        # evaluate this controller.
        # IPython.embed()
        controller.eval()
        eval_controller_results = OrderedDict()
        for i in range(200):
            _, _, _, arch = controller()
            arch = self.process_arch(arch)
            try:
                r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
                if r not in eval_controller_results.keys():
                    eval_controller_results[r] = 1
                else:
                    eval_controller_results[r] += 1
            except Exception as e:
                logging.error(e)
                continue

        logging.info('End of epoch: controller evaluation {}'.format(len(eval_controller_results.keys())))
        self.save_arch_pool_performance(list(eval_controller_results.keys()),
                                        list(eval_controller_results.values()),
                                        prefix='enas')

        return total_loss.avg, total_reward.avg, total_entropy.avg
