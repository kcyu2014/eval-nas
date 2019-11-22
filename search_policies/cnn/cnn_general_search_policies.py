import logging
from abc import ABC, abstractmethod
import random
from collections import OrderedDict
from functools import partial
from operator import itemgetter

import IPython
from scipy.stats import kendalltau
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10

import utils
from search_policies.cnn.search_space.nas_bench.util import compute_sparse_kendalltau, sort_hash_perfs, \
    compute_percentile
from search_policies.cnn.search_space.search_space_utils import CNNSearchSpace
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
# after import torch, add this
from visualization.cnn_best_configs import BEST_RANK_BY_MODEL_HASH
from visualization.plot_rank_change import process_rank_data_nasbench
from visualization.process_data import *

torch.backends.cudnn.benchmark = True
import torch.backends.cudnn as cudnn
from . import procedures as procedure_ops


class CNNSearchPolicy(ABC):
    """
    Search policy for CNN model.

    """

    @property
    def epoch(self):
        if 'epoch' in self.running_stats.keys():
            return self.running_stats['epoch']
        else:
            return 0

    @property
    def ranking_per_epoch(self):
        if 'ranking_per_epoch' in self.running_stats.keys():
            return self.running_stats['ranking_per_epoch']
        else:
            logging.info("Initializing one")
            self.running_stats['ranking_per_epoch'] = OrderedDict()
            return self.running_stats['ranking_per_epoch']

    def __init__(self, args, sub_dir_path=None):
        super(CNNSearchPolicy, self).__init__()

        self.args = args

        # initialize path and logger
        if not self.args.continue_train:
            self.sub_directory_path = sub_dir_path or '{}_SEED_{}'.format(self.args.supernet_train_method,
                                                                          self.args.seed)
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            utils.create_exp_dir(self.exp_dir)
            utils.save_json(args, self.exp_dir + '/args.json')
        if self.args.visualize:
            self.viz_dir_path = utils.create_viz_dir(self.exp_dir)

        if self.args.tensorboard:
            self.tb_dir = self.exp_dir
            tboard_dir = os.path.join(self.args.tboard_dir, self.sub_directory_path)
            self.writer = SummaryWriter(tboard_dir)

        if self.args.debug:
            torch.autograd.set_detect_anomaly(True)

        # Set logger and directory.
        self.logger = utils.get_logger(
            "train_search",
            file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')),
            level=logging.INFO if not args.debug else logging.DEBUG
        )

        # Random seed should be set once the Policy is created.
        logging.info(f"setting random seed as {args.seed}")
        utils.torch_random_seed(args.seed)
        logging.info('gpu number = %d' % args.gpus)
        logging.info("args = %s", args)

        # metrics to track
        # self.ranking_per_epoch = OrderedDict()
        self.search_space = None # store the search space.
        self.model = None # store the model
        self.model_fn = None
        self.running_stats = OrderedDict() # store all running status.

        # to log the training results.
        self.logging_fn = self.logging_at_epoch
        if args.supernet_train_method in ['darts', 'spos']:
            """ Fundamental baseline training methods
            sample 1 architecture per batch
            train supernet
            Conv op has maximum possible filter channels (== output size of cell)
            Random a chunk of it.
            """
            train_fn = procedure_ops.darts_train_model
            self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler)
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
        elif args.supernet_train_method == 'fairnas':
            """
            Extend darts training method with FairNas strategy. It is not possible to use directly the FairNAS,
            but we can extend it into 2 method.
            """
            train_fn = procedure_ops.fairnas_train_model_v1
            self.train_fn = partial(train_fn, args=self.args, architect=None,
                                    topology_sampler=self.random_sampler,
                                    op_sampler=self.op_sampler
                                    )
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
        else:
            pass

    @abstractmethod
    def random_sampler(self, model, architect, args):
        pass

    @abstractmethod
    def op_sampler(self, model, architect, args):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        pass

    @abstractmethod
    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
        pass

    def cpu(self):
        self.model = self.model.cpu()

    def initialize_run(self, sub_dir_path=None):

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self._loss = criterion
        tr, va, te = self.load_dataset()
        return tr, va, te, criterion

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        args = self.args
        model = self.model_fn(args)
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

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        # scheduler as Cosine.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        return model, optimizer, scheduler

    def load_dataset(self):
        args = self.args
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

    @staticmethod
    def next_batches(dataloader, num_batches):
        queue = []
        _batch_count = 0
        for data in dataloader:
            _batch_count += 1
            queue.append(data)
            if _batch_count > num_batches:
                # process iteration
                break
        return queue

    @staticmethod
    def _compute_kendall_tau(ranking_per_epoch, compute_across_time=False):
        """
        Compute Kendall tau given the ranking per epochs.

        :param ranking_per_epoch:
        :param compute_across_time: True for ranking-per-epoch always fixed, False for dynamic list of models.
        :return: kd_tau dict{epoch_key: KendallTau}
        """
        if compute_across_time:
            # Compute Kendall tau for every epochs and save them into result.
            epoch_keys = [k for k in reversed(ranking_per_epoch.keys())]
            epoch_keys.insert(0, 10000000)
            kd_tau = {}
            for ind, k in enumerate(epoch_keys[:-1]):
                elem = []
                if ind == 0:
                    # Sort the ground-truth ranking
                    p = sorted([elem[1] for elem in ranking_per_epoch[epoch_keys[ind + 1]]], key=itemgetter(3))
                    rank_1 = np.array([elem.geno_id for elem in p], dtype=np.uint)
                else:
                    rank_1 = np.array([elem[1].geno_id for elem in ranking_per_epoch[k]], dtype=np.uint)
                for j in epoch_keys[ind + 1:]:
                    rank_2 = np.array([elem[1].geno_id for elem in ranking_per_epoch[j]], dtype=np.uint)
                    elem.append(kendalltau(rank_1, rank_2))
                kd_tau[k] = elem
            logging.info("Latest Kendall Tau (ground-truth vs {}): {}".format(epoch_keys[1], kd_tau[10000000][0]))
            return kd_tau, kd_tau[10000000][0].correlation
        else:
            # Dynamic ranking per epoch size, thus only compute the KDT against the final ranking.
            epoch_keys = [k for k in reversed(ranking_per_epoch.keys())]
            kd_tau = {}
            # only sort across the ground-truth
            for ind, k in enumerate(epoch_keys):
                p = sorted([elem[1] for elem in ranking_per_epoch[k]], key=itemgetter(3))
                rank_gt = np.array([elem.geno_id for elem in p], dtype=np.uint)
                rank_2 = np.array([elem[1].geno_id for elem in ranking_per_epoch[k]], dtype=np.uint)
                kd_tau[k] = kendalltau(rank_gt, rank_2)

            kd_tau[10000000] = kd_tau[epoch_keys[0]]
            logging.info("Latest Kendall Tau (ground-truth vs {}): {}".format(epoch_keys[0], kd_tau[epoch_keys[0]][0]))
            return kd_tau, kd_tau[epoch_keys[0]][0]

    def _save_ranking_results(self, save_data, epoch,
                              prefix=None,
                              compute_kdt_before=False,
                              sparse_kdt=True, sparse_kdt_threshold=1e-3,
                              percentile=True, percentile_top_K=(3, 5, 10, 20),
                              random_kdt=True, random_kdt_numrepeat=5, random_kdt_num_archs=(10, 20, 50, 100)):
        """
        Save the ranking results if necessary.

        13.09.2019: Adding the sparse kendall tau, percentile, random kendall tau.

        :param save_data:
        :param epoch:
        :param prefix: Prefix to the tensorboard scalar.
        :param compute_kdt_before: True to compute the kendall tau for additional evaluation approachs.
        :param sparse_kdt: True to compute the sparse kendall tau based on the GT accuracy.
        :param percentile: True to compute the top K model percentile.
        :param percentile_top_K: Number of top K architectures for percentile.
        :param random_kdt: True to compute the random K architectures's kendall tau.
        :param random_kdt_numrepeat: Number of repeated times for this random kdt
        :param random_kdt_num_archs: Number of architectures for random kdt
        :return: None
        """
        prefix_name = lambda prefix, name: name if prefix is None else f'{prefix}-{name}'

        try:
            fname = prefix_name(prefix, f'rank_change-{epoch}.pdf')
            fig = process_rank_data_nasbench(save_data, os.path.join(self.exp_dir, fname))
            if self.writer:
                self.writer.add_figure(tag=fname.split('.')[0].replace('_','-'), figure=fig, global_step=epoch)
        except Exception as e:
            logging.warning(e)

        try:
            ranking_per_epoch = save_data['ranking_per_epoch']
        except KeyError as e:
            logging.warning("save_data parsed into _save_ranking_results expect having key ranking_per_epoch"
                            "; got {}. Using self.ranking_per_epoch instead".format(save_data.keys()))
            ranking_per_epoch = self.ranking_per_epoch

        # Compute Kendall tau for every epochs and save them into result.
        # IPython.embed()
        kd_tau, kd_tau_report = self._compute_kendall_tau(ranking_per_epoch)
        save_data['kendaltau'] = kd_tau

        if compute_kdt_before and hasattr(self, 'ranking_per_epoch_before'):
            kd_tau_before, kd_tau_before_report = self._compute_kendall_tau(self.ranking_per_epoch_before)
            save_data['kendaltau_before'] = kd_tau_before

        if self.writer is not None:
            p = sorted([elem[1] for elem in ranking_per_epoch[epoch]], key=itemgetter(2))
            tensorboard_summarize_list(
                [e[0] for e in p], self.writer, prefix_name(prefix, 'eval_acc'), epoch, ascending=False
            )
            tensorboard_summarize_list(
                [e[1] for e in p], self.writer, prefix_name(prefix, 'eval_obj'), epoch, ascending=True
            )
            self.writer.add_scalar(prefix_name(prefix, 'eval_kendall_tau'), kd_tau_report, epoch)
            if compute_kdt_before and hasattr(self, 'ranking_per_epoch_before'):
                self.writer.add_scalar(prefix_name(prefix, 'eval_kendall_tau/original'), kd_tau_before_report, epoch)

        # add these and collect writer keys
        if any([sparse_kdt, percentile, random_kdt]):

            data = ranking_per_epoch[epoch]
            # ranking by valid accuracy
            model_ids = [elem[1][3] for elem in data]
            model_perfs = [elem[1][0] for elem in data]
            model_ids, model_perfs = sort_hash_perfs(model_ids, model_perfs)
            model_gt_perfs = self.search_space.query_gt_perfs(model_ids)
            sorted_indices = np.argsort(model_perfs)[::-1]
            sorted_model_ids = [model_ids[i] for i in sorted_indices]

            # IPython.embed(header='checking the saving here.')
            add_metrics = {}
            if sparse_kdt:
                if not isinstance(sparse_kdt_threshold, (tuple, list)):
                    sparse_kdt_threshold = [sparse_kdt_threshold]
                for th in sparse_kdt_threshold:

                    kdt = compute_sparse_kendalltau(model_ids, model_perfs, model_gt_perfs,
                                                                   threshold=th)
                    add_metrics[prefix_name(prefix, f'eval_kendall_tau/sparse_{th}')] = kdt.correlation

            if percentile:
                for top_k in percentile_top_K:
                    res = compute_percentile(sorted_model_ids,
                                             self.search_space.num_architectures,
                                             top_k,
                                             verbose=self.args.debug)
                    mname = prefix_name(prefix, 'percentile')
                    add_metrics[f'{mname}/min_{top_k}'] = res.min()
                    add_metrics[f'{mname}/median_{top_k}'] = np.median(res)
                    add_metrics[f'{mname}/max_{top_k}'] = res.max()
                logging.info("{} of top {}: {} - {} - {}".format(
                    mname,
                    top_k, res.min(), np.median(res), res.max()))

            if random_kdt:
                for subsample in random_kdt_num_archs:
                    if subsample > len(sorted_model_ids):
                        continue
                    kdt_final = []
                    for _ in range(random_kdt_numrepeat):
                        sub_model_indices = sorted(
                            np.random.choice(np.arange(0, len(sorted_model_ids)), subsample, replace=False).tolist())
                        sub_model_ids = [sorted_model_ids[i] for i in sub_model_indices]
                        kdt = kendalltau(sub_model_ids, list(reversed(sorted(sub_model_ids))))
                        kdt_final.append(kdt.correlation)

                    kdt_final = np.asanyarray(kdt_final, dtype=np.float)
                    mname = prefix_name(prefix, 'eval_kendall_tau')
                    add_metrics[f'{mname}/random_{subsample}_min'] = kdt_final.min()
                    add_metrics[f'{mname}/random_{subsample}_max'] = kdt_final.max()
                    add_metrics[f'{mname}/random_{subsample}_mean'] = kdt_final.mean()

                    logging.info("Random subsample {} archs: kendall tau {} ({},{})".format(
                        subsample, kdt_final.mean(), kdt_final.min(), kdt_final.max()))

            # end of additioanl metrics
            if self.writer:
                for k, v in add_metrics.items():
                    self.writer.add_scalar(k, v, epoch)

        return save_data

    def _save_results(self, save_data, epoch, rank_details=False, filename='result.json', **kwargs):
        if rank_details:
            save_data = self._save_ranking_results(save_data, epoch, **kwargs)

        utils.save_json(save_data, os.path.join(self.exp_dir, filename))
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.export_scalars_to_json(os.path.join(self.exp_dir, 'tb_scalars.json'))

    def save_results(self, epoch, rank_details=False):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details)

    def check_should_save(self, epoch):
        """
        invoke the evaluate step, this is also used to update epoch information.
        :param epoch:
        :return:
        """
        self.running_stats['epoch'] = epoch
        if self.args.extensive_save and epoch > 50:
            return any([(epoch - i) % self.args.save_every_epoch == 0 for i in range(3)])
        return epoch % self.args.save_every_epoch == 0

    # logging for normal one.
    def logging_at_epoch(self, acc, obj, epoch, keyword, display_dict=None):
        message = f'{keyword} at epoch {epoch} | loss: {obj:8.2f} | top_1_acc: {acc:8.2f}'
        if display_dict:
            for k, v in display_dict.items():
                message += f' | {k}: {v} '

        logging.info(message)
        self.running_stats['epoch'] = epoch
        if self.writer:
            self.writer.add_scalar(f'{keyword}/loss', obj, epoch)
            self.writer.add_scalar(f'{keyword}/top_1_acc', acc, epoch)
            if display_dict:
                for k, v in display_dict.items():
                    self.writer.add_scalar(f'{keyword}/{k}', v, epoch)

    def logging_maml(self, acc, obj, epoch, keyword, **kwargs):
        """ specifically for MAML procedures"""
        if isinstance(acc, tuple) and len(acc) == 2:
            self.logging_at_epoch(acc[0], obj[0], epoch, keyword + '_task', **kwargs)
            self.logging_at_epoch(acc[1], obj[1], epoch, keyword + '_meta', **kwargs)
        else:
            return self.logging_at_epoch(acc, obj, epoch, keyword, **kwargs)

