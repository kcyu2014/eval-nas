import logging
import os
from collections import deque
from functools import partial

import torch
from scipy.stats import kendalltau
from torch import nn as nn

import utils as project_utils
from search_policies.cnn.darts_policy import utils_for_nasbench as darts_nasbench_utils, utils as darts_utils
from search_policies.cnn.darts_policy.architect import DartsArchitect
from search_policies.cnn.darts_policy.darts_search_policy import DARTSNasBenchSearch
from search_policies.cnn.darts_policy.model_search_nasbench import NasBenchNetSearchDarts
from search_policies.cnn.darts_policy.model_search_nasbench_fbnet import NasBenchNetSearchFBNet
from search_policies.cnn.search_space.nas_bench.model import NasBenchNet
from search_policies.cnn.procedures import darts_train_model, darts_model_validation


class FBNetNasBenchSearch(DARTSNasBenchSearch):

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        args = self.args

        if self.args.search_space == 'nasbench':
            self.model_fn = NasBenchNetSearchFBNet
            self.fixmodel_fn = NasBenchNet
            model = self.model_fn(args)
            utils = darts_nasbench_utils
        else:
            raise NotImplementedError("Not supported")
        # finialize model update
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

        darts = DartsArchitect(model, args=args)
        model = self.parallel_model
        # logging.info("DARTS param size = %fMB", utils.count_parameters_in_MB(darts))
        self.train_fn = partial(darts_train_model, args=args, architect=darts, sampler=None)
        self.eval_fn = partial(darts_model_validation, args=args, verbose=True)
        self.controller = darts

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        # scheduler as Cosine.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.learning_rate_min)
        return model, optimizer, scheduler, darts, None

    def wrap_fbnet_search_config(self):
        for k, v in self.darts_search_config.__dict__.items():
            self.args.__dict__[k] = v

    def run(self):
        # process UTILS and args
        args = self.args
        self.utils = darts_nasbench_utils if args.search_space == 'nasbench' else darts_utils
        # using the darts configs. just add another thing
        self.darts_search_config = args.fbnet_search_config
        self.wrap_fbnet_search_config()

        # build the functions.
        train_queue, valid_queue, test_queue, repeat_valid_queue, train_criterion, eval_criterion = self.initialize_run()
        model, optimizer, scheduler, darts, darts_optimizer = self.initialize_model()

        self.child_arch_pool = deque()
        self.optimizer = optimizer
        self.scheduler = scheduler

        child_arch_pool = self.child_arch_pool
        # Begin the full loop
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            self.model.epoch = epoch  # update epoch for temperature
            logging.info('epoch %d lr %e gumbel_softmax temp %e', epoch, lr, self.model.temperature())
            # summarize darts training.
            ids, sample_nums = self.sample_and_summarize(10)

            # DARTS train
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, train_criterion, optimizer, lr)
            logging.info(f'Train at epoch {epoch} | loss: {train_obj:8.2f} | top_1_acc: {train_acc:8.2f}')

            # evaluate:
            valid_acc, valid_obj = self.eval_fn(valid_queue, model, eval_criterion)
            logging.info(f'Valid at epoch {epoch} | loss: {valid_obj:8.2f} | top_1_acc: {valid_acc:8.2f}')

            if self.writer:
                self.writer.add_scalar(f'train/loss', train_obj, epoch)
                self.writer.add_scalar(f'train/top_1_acc', train_acc, epoch)
                self.writer.add_scalar(f'valid/loss', valid_obj, epoch)
                self.writer.add_scalar(f'valid/top_1_acc', valid_acc, epoch)

            # end of normal training loop.
            if not epoch % args.save_every_epoch == 0:
                continue
            # Saving the current performance of arch pool into the list.
            # Evaluate seed archs
            logging.info("="*89)
            logging.info('Evalutation at epoch {}'.format(epoch))

            # Output archs and evaluated error rate
            old_archs = child_arch_pool
            with open(os.path.join(self.exp_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
                for arch in old_archs:
                    arch = self.process_archname(arch)
                    fa.write('{}\n'.format(arch))
            logging.info("Complete arch pool {}".format(old_archs))
            # save model
            project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
            # compute kd_tau directly here.
            # IPython.embed(header="compute kdtau here")
            r1 = ids
            r2 = list(reversed(sorted(r1)))
            kdtau = kendalltau(r1, r2)
            self.ranking_per_epoch[epoch] = r1
            self.eval_result[epoch] = kdtau
            logging.info("=" * 89)
            logging.info('Kendall tau at epoch {} is {}'.format(epoch, kdtau))
            logging.info('Top 10 sampled id {}'.format(ids[:10]))
            logging.info('Top 10 sampled times {}'.format(sample_nums[:10]))
            logging.info("Alphas: {}".format(self.model.arch_parameters()))
            logging.info("=" * 89)
            self.save_results(epoch, rank_details=False)

        project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
        best_id = ids[0]
        return best_id, self.nasbench_model_specs[best_id]