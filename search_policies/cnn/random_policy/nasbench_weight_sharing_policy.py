"""
This is a test case to isolate the search space.

# Reference: Single Path One-Shot Neural Architecture Search with Uniform Sampling
# NOTE: Release this code after, build as fast as possible.
# Goal is to develop this and test on NasBench-101, small search space.
# Then to other search space. but now all the experiment is  NASBench-101 centric.
"""

import os
import gc
import logging
import operator

import IPython
import shutil

import numpy as np
import torch
from functools import partial
from collections import namedtuple, OrderedDict, deque

import utils
from search_policies.cnn.cnn_general_search_policies import CNNSearchPolicy
from search_policies.cnn.enas_policy.enas_micro.data.data import RepeatedDataLoader
from search_policies.cnn.search_space.nas_bench.nasbench_search_space import NASbenchSearchSpace, \
    NasBenchSearchSpaceLinear, NasBenchSearchSpaceSubsample, NasBenchSearchSpaceICLRInfluenceWS

from search_policies.cnn import model as model_module
import search_policies.cnn.procedures as procedure_ops
from search_policies.cnn.search_space.nas_bench.util import change_model_spec
from search_policies.cnn.utils import AverageMeter
from visualization.process_data import tensorboard_summarize_list


Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class NasBenchWeightSharingPolicy(CNNSearchPolicy):
    r""" Class to train NAS-Bench with weight sharing in general.

    This support all variation of NASBench space.
    Use this to build a simple network trainer and ranking, to facilitate other usage.

    Super class this to support many other things.
    """
    # top_K_complete_evaluate = 200
    trained_model_spec_ids = []
    # evaluate_model_spec_ids = []        # ids to evaluate.
    # sample_step_for_evaluation = 2      # Evaluate sample steps.
    eval_result = OrderedDict()

    # defining the model spec placeholder.
    model_spec = None
    model_spec_id = None

    ## This belongs to interaction for now, should be removed later.
    @property
    def nasbench_model_specs(self):
        return self.search_space.nasbench_hashs

    @property
    def nasbench_hashs(self):
        return self.search_space.nasbench_hashs

    @property
    def evaluate_model_spec_ids(self):
        return self.search_space.evaluate_model_spec_ids

    def evaluate_model_spec_id_pool(self):
        # shrink the eval pool.
        return self.search_space.evaluate_model_spec_id_pool()

    def model_spec_by_id(self, mid):
        return self.search_space.nasbench_model_specs[mid]

    def random_sampler(self, model, architect, args):
        """
        random sampler and update the model scenario
        :param model:
        :param architect:
        :param args:
        :return:
        """
        rand_spec_id, rand_spec = self.search_space.random_topology()
        self.model_spec_id = rand_spec_id
        self.model_spec = rand_spec
        new_model = change_model_spec(model, rand_spec)
        # this is saved per sample.
        self.trained_model_spec_ids.append(rand_spec_id)
        return new_model

    def op_sampler(self, model, architect, args):
        """
        Sample operation from model, used mainly for FairNAS procedure.
        :param model:
        :param architect:
        :param args:
        :return:
        """
        spec = self.model_spec
        ops = spec.ops
        avail_ops = self.search_space.available_ops
        try:
            op_vs_choice = np.tile(np.arange(len(avail_ops)), (len(ops)-2, 1))
            op_vs_choice = np.apply_along_axis(np.random.permutation, 1, op_vs_choice).transpose()
            for i in range(len(avail_ops)):
                new_ops = [avail_ops[ind] for ind in op_vs_choice[i]]
                spec.ops = ['input',] + new_ops + ['output']
                yield change_model_spec(model, spec)
        except ValueError as e:
            logging.warning(f'Op sampler: received exception {e}, return the original model without any op sampling.')
            yield model

    def __init__(self, args, full_dataset=False):
        """ Init and ad this. """
        super(NasBenchWeightSharingPolicy, self).__init__(
            args=args,
            sub_dir_path='{}_SEED_{}'.format(args.supernet_train_method, args.seed)
        )
        self.args = args

        if args.search_space == 'nasbench':
            self.model_fn = model_module.NasBenchNetSearch
            self.search_space = NASbenchSearchSpace(args, full_dataset=full_dataset)
        elif args.search_space == 'nasbench_linear':
            self.model_fn = model_module.NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceLinear(args)
        elif args.search_space == 'nasbench_subspace':
            self.model_fn = model_module.NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceSubsample(args)
        elif args.search_space == 'nasbench_iclr_wsinfluence':
            self.model_fn = model_module.NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceICLRInfluenceWS(args)
        else:
            raise NotImplementedError("Other search space not supported at this moment.")

        self.counter = 0

    # Search happens here.
    def run(self):
        """
        Procedure of training. This run describes the entire training procedure.
        :return:
        """
        train_queue, valid_queue, test_queue, criterion = self.initialize_run()
        args = self.args
        model, optimizer, scheduler = self.initialize_model()
        fitness_dict = {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        logging.info(">> Begin the search with supernet method :".format(args.supernet_train_method))

        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]

            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})

            # validation
            valid_acc, valid_obj = self.validate_model(model, valid_queue, self.model_spec_id, self.model_spec)
            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')

            if not self.check_should_save(epoch):
                continue
            # evaluate process.
            self.save_duplicate_arch_pool('valid', epoch)
            fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue)
            utils.save_checkpoint(model, optimizer, self.running_stats, self.exp_dir)
            self.save_results(epoch, rank_details=True)

        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, return the best of training.
        ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
        best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
        return best_id, self.search_space.nasbench_model_specs[best_id]

    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        # this is flaw, let me do another evaluating on all possible models.

        # compute all possible batches
        complete_valid_queue = data_source
        _valid_queue = []
        nb_batch_per_model, nb_models, valid_model_pool = self.\
            search_space.validate_model_indices(len(complete_valid_queue))
        total_valid_acc = 0.
        total_valid_obj = 0.

        valid_accs = OrderedDict()
        valid_objs = OrderedDict()

        for step, val_d in enumerate(complete_valid_queue):
            if self.args.debug:
                if step > 10:
                    logging.debug("Break after 10 step in validation step.")
                    break

            _valid_queue.append(val_d)
            if step % nb_batch_per_model == 0 and step > 0:
                _id = valid_model_pool[min(int(step / nb_batch_per_model), nb_models - 1)]
                current_model = change_model_spec(current_model, self.search_space.topologies[_id])
                current_model.eval()
                _valid_acc, _valid_obj = self.eval_fn(_valid_queue, current_model, self._loss)
                # logging.info(f"model id {valid_model_pool[_id]} acc {_valid_acc} loss {_valid_obj}")
                # update the metrics
                total_valid_acc += _valid_acc
                total_valid_obj += _valid_obj
                # store the results
                valid_accs[_id] = _valid_acc
                valid_objs[_id] = _valid_obj
                _valid_queue = []

        self.save_arch_pool_performance(archs=list(valid_accs.keys()), perfs=list(valid_accs.values()), prefix='valid')
        return total_valid_acc / nb_models, total_valid_obj/ nb_models

    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None, model_spec_id_pool=None):
        """
        Full evaluation of all possible models.
        :param epoch:
        :param data_source:
        :param fitnesses_dict: Store the model_spec_id -> accuracy
        :return:
        """
        # Make sure this id pool is not None.
        model_spec_id_pool = model_spec_id_pool or self.evaluate_model_spec_id_pool()

        rank_gens, eval_result = procedure_ops.evaluate_procedure.evaluate_normal(
            self, self.parallel_model, fitnesses_dict, model_spec_id_pool, data_source, change_model_spec,
            self._loss
        )

        self.ranking_per_epoch[epoch] = rank_gens
        self.eval_result[epoch] = eval_result

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                             f'hash: {elem[0]}, pos {pos}')

        # save the eval arch pool.
        archs = [elem[1].geno_id for elem in rank_gens]
        perfs = [elem[1].valid_acc for elem in rank_gens]
        self.save_arch_pool_performance(archs, perfs, prefix='eval')
        self.save_duplicate_arch_pool(prefix='eval', epoch=epoch)
        self.search_space.eval_model_spec_id_rank(archs, perfs)

        if self.writer:
            # process data into list.
            accs_after, objs_after = zip(*eval_result.values())
            tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
            tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)

        return fitnesses_dict

    def save_results(self, epoch, rank_details=True):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, sparse_kdt=True, percentile=True, random_kdt=True)

    def save_duplicate_arch_pool(self, prefix, epoch):
        f_pool = os.path.join(self.exp_dir, f'{prefix}_arch_pool')
        f_perf = os.path.join(self.exp_dir, f'{prefix}_arch_pool.perf')
        if os.path.exists(f_pool):
            shutil.copy(f_pool, f_pool + '.{}'.format(epoch))
        if os.path.exists(f_perf):
            shutil.copy(f_perf, f_perf + '.{}'.format(epoch))

    def save_arch_pool_performance(self, archs, perfs, prefix='valid'):
        old_archs_sorted_indices = np.argsort(perfs)[::-1]
        old_archs = [archs[i] for i in old_archs_sorted_indices]
        old_archs_perf = [perfs[i] for i in old_archs_sorted_indices]
        with open(os.path.join(self.exp_dir, f'{prefix}_arch_pool'), 'w') as fa_latest:
            with open(os.path.join(self.exp_dir, f'{prefix}_arch_pool.perf'), 'w') as fp_latest:
                for arch_id, perf in zip(old_archs, old_archs_perf):
                    arch_id = self.search_space.process_archname_by_id(arch_id)
                    fa_latest.write('{}\n'.format(arch_id))
                    fp_latest.write('{}\n'.format(perf))



class NasBenchNetOneShotPolicy(NasBenchWeightSharingPolicy):
    """
    This is implemented with NAO implementation, i.e. add this arch pool idea.

    Only support nasbench search space for now. Add more later.

    Use this to build a simple network trainer and ranking, to facilitate other usage.

    """
    trained_model_spec_ids = []
    eval_result = OrderedDict()

    def __init__(self, args):
        if args.search_policy in ['fairnas', 'spos']:
            args.supernet_train_method = args.search_policy
        super(NasBenchNetOneShotPolicy, self).__init__(args=args)
        # initialize eval pool. TODO Check if this cause the trouble, yes it is causing trouble now.
        self.search_space.evaluate_model_spec_ids = deque()
        arch_ids = self.search_space.random_eval_ids((args.num_intermediate_nodes - 1) * 40)
        for arch_id in arch_ids:
            self.search_space.eval_model_spec_id_append(arch_id)
        logging.info("Initial pool size {}".format(self.search_space.evaluate_model_spec_id_pool()))

    def run(self):
        """
        Difference with super.run() is, it will change the eval pool by random sampling some new architecture to replace
        the old ones.
        :return:
        """
        train_queue, valid_queue, test_queue, criterion = self.initialize_run()
        repeat_valid_queue = RepeatedDataLoader(valid_queue)
        args = self.args
        model, optimizer, scheduler = self.initialize_model()
        fitness_dict = {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        logging.info(">> Begin the search with supernet method: {}".format(args.supernet_train_method))
        logging.info("Always setting BN-train to True!")
        for epoch in range(args.epochs):
            args.current_epoch = epoch
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            # training for each epoch.
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})

            # do this after pytorch 1.1.0
            scheduler.step()

            # validation, compare to the traditional, we only evaluate the eval arch pool in this step.
            validate_accuracies, valid_acc, valid_obj = self.child_valid(
                model, repeat_valid_queue, self.evaluate_model_spec_id_pool(), criterion)

            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')

            if not self.check_should_save(epoch):
                continue
            self.save_duplicate_arch_pool('valid', epoch)
            logging.info("Evaluating and save the results.")
            utils.save_checkpoint(model, optimizer, self.running_stats, self.exp_dir)
            logging.info("Totally %d architectures now to evaluate", len(self.evaluate_model_spec_id_pool()))
            # evaluate steps.
            fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue)
            # Generate new archs for robust evaluation.
            # replace bottom archs
            num_new_archs = self.search_space.replace_eval_ids_by_random(args.controller_random_arch)
            logging.info("Generate %d new archs", num_new_archs)
            self.save_results(epoch, rank_details=True)

        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, return the best of training.
        ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
        best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
        return best_id, self.nasbench_model_specs[best_id]

    def save_results(self, epoch, rank_details=False):
        # IPython.embed(header='Checking save results json problem')
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
            'top-k-left': list(self.search_space.evaluate_model_spec_id_pool())
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, sparse_kdt=True, percentile=True, random_kdt=True)

    def child_valid(self, model, valid_queue, arch_pool, criterion):
        valid_acc_list = []
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        logging.info("num valid arch {}".format(len(arch_pool)))
        with torch.no_grad():
            model.eval()
            for i, arch in enumerate(arch_pool):
                # for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = valid_queue.next_batch()
                inputs = inputs.cuda()
                targets = targets.cuda()
                n = inputs.size(0)
                arch_l = arch
                model = change_model_spec(model, self.search_space.topologies[arch])
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                valid_acc_list.append(prec1.data / 100)

                if (i + 1) % 100 == 0:
                    logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f',
                                 self.search_space.process_archname_by_id(arch_l),
                                 loss, prec1, prec5)

        self.save_arch_pool_performance(arch_pool, valid_acc_list, prefix='valid')
        return valid_acc_list, objs.avg, top1.avg

    @staticmethod
    def _compute_kendall_tau(ranking_per_epoch, compute_across_time=False):
        return NasBenchWeightSharingPolicy._compute_kendall_tau(ranking_per_epoch, compute_across_time=False)

