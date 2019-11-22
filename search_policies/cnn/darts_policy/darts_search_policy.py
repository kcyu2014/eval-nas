import logging
import operator
import os
import shutil

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils
from scipy.stats import kendalltau

import utils as project_utils
import torchvision.datasets as dset

from collections import namedtuple, deque
from tensorboardX import SummaryWriter

from search_policies.cnn.cnn_general_search_policies import CNNSearchPolicy
from search_policies.cnn.random_policy import NasBenchWeightSharingPolicy
from .architect import DartsArchitect
from .model_search_nasbench import NasBenchNetSearchDarts

from ..search_space.nas_bench.model import NasBenchNet
from ..procedures.train_search_procedure import darts_model_validation, darts_train_model
import search_policies.cnn.darts_policy.utils_for_nasbench as darts_nasbench_utils
import search_policies.cnn.darts_policy.utils as darts_utils
from .model_search import Network as DARTSWSNetwork
from .architect import Architect
# import search_policies.cnn.procedures as procedure_ops


Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class DARTSMicroCNNSearchPolicy(CNNSearchPolicy):

    def __init__(self, args, darts_args):
        super(DARTSMicroCNNSearchPolicy, self).__init__(args, sub_dir_path='darts-search-{}'.format(args.seed))
        self.darts_args = darts_args
        # self.train_fn = partial(procedure_ops.darts_train_model, args=self.args, architect=None, sampler=None)
        # self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)

    def run(self):
        args = self.args
        utils = project_utils
        criterion = nn.CrossEntropyLoss().cuda()
        eval_criterion = nn.CrossEntropyLoss().cuda()

        train_queue, valid_queue, test_queue = self.load_dataset()
        model = DARTSWSNetwork(args.init_channels, 10, args.layers, criterion)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        architect = Architect(model, args)

        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

            print(torch.softmax(model.alphas_normal, dim=-1))
            print(torch.softmax(model.alphas_reduce, dim=-1))

            # training
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model,
                                                 architect=architect,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 lr=lr)
            logging.info('train_acc %f', train_acc)

            # validation
            valid_acc, valid_obj = self.eval_fn(test_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            utils.save_json(genotype, os.path.join(self.exp_dir, 'arch_pool.json'))
            if epoch % 30 == 0:
                shutil.copy(os.path.join(self.exp_dir, 'arch_pool.json'),
                            os.path.join(self.exp_dir, 'arch_pool') + '.{}.json'.format(epoch))
                darts_utils.save(model, os.path.join(self.exp_dir, 'weights.pt'))

        # best choosed dag
        bestdag = model.genotype()
        return -1, bestdag

    # disabling random sampler.
    def random_sampler(self, model, architect, args):
        return model

    # override these models.
    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
        pass

    def op_sampler(self, model, architect, args):
        pass

    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        pass


class DARTSNasBenchSearch(NasBenchWeightSharingPolicy):
    top_K_complete_evaluate = 200
    top_K_num_sample = 1000
    exp_name = "DARTSNasBench"
    """

    Becasue darts has a feature, that only sampler samples the new architecture (in this sence, very similar to 
    NAO sampler, i.e. NAO-infer = darts sample.
    SO I could keep NAO's code not touch, only change the logic of training such sampler.
    Then, during evaluation, use the child-arch-pool as the same.
    Only difference is, training child network is by sampling but not random an arch pool model.
    Arch pool is solely for evaluation.

    """

    def wrap_darts_search_config(self):
        for k, v in self.args.darts_search_config.__dict__.items():
            self.args.__dict__[k] = v

    # The same as NAO one.
    def initialize_run(self):
        """

        :return:
        """
        args = self.args
        utils = project_utils
        if not self.args.continue_train:
            self.sub_directory_path = '{}-{}_SEED_{}'.format(self.exp_name,self.args.save, self.args.seed)
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
        self.train_loss = criterion
        
        train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_length if args.cutout else None)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.darts_search_config.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.evaluate_batch_size,
            shuffle=False, pin_memory=True, num_workers=2)

        return train_queue, valid_queue, test_queue, None, criterion, eval_criterion

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        args = self.args

        if self.args.search_space == 'nasbench':
            self.model_fn = NasBenchNetSearchDarts
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

    def run(self):
        # process UTILS and args
        args = self.args
        self.utils = darts_nasbench_utils if args.search_space == 'nasbench' else darts_utils
        utils = self.utils
        self.darts_search_config = args.darts_search_config
        self.wrap_darts_search_config()

        # build the functions.
        train_queue, valid_queue, test_queue, repeat_valid_queue, train_criterion, eval_criterion = self.initialize_run()
        model, optimizer, scheduler, darts, darts_optimizer = self.initialize_model()
        fitness_dict = {}

        self.child_arch_pool = deque()
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Train child model
        child_arch_pool = self.child_arch_pool
        # IPython.embed(header='Test sampler and map.')
        step = 0
        # Begin the full loop
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
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

    def save_results(self, epoch, rank_details=False):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details)

    def process_arch(self, arch):
        if self.args.search_space == 'nasbench':
            # matrix, ops = darts_nasbench_utils.parse_arch_to_model_spec_matrix_op(arch, self.args.child_num_cells)
            # model_spec = ModelSpec_v2(matrix, ops)
            # return model_spec
            return self.nasbench_model_specs[arch]
        else:
            return arch

    def process_archname(self, arch):
        if self.args.search_space == 'nasbench':
            return ' '.join(map(str, [arch, self.nasbench_hashs[arch]]))
        else:
            return ' '.join(map(str, arch[0] + arch[1]))

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

        # save the ranking, according to their GENOTYPE but not particle id
        rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))

        self.ranking_per_epoch[epoch] = rank_gens
        self.eval_result[epoch] = eval_result
        # IPython.embed(header="Check evaluation result")

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                             f'hash: {elem[0]}, pos {pos}')

        return fitnesses_dict

    def process_nasbench(self):
        super(DARTSNasBenchSearch, self).process_nasbench(only_hash=True)

    def clean_arch_pool(self, arch_pool):
        new_arch_pool = []
        for i in arch_pool:
            if not i in new_arch_pool:
                new_arch_pool.append(i)
        return new_arch_pool

    def child_arch_pool_append(self, arch):
        # arch should be the hash
        hash = arch
        if isinstance(arch, int):
            pass
        else:
            arch = self.nasbench_hashs.index(hash)

        if arch in self.child_arch_pool:
            self.child_arch_pool.remove(arch)
        self.child_arch_pool.append(arch)
        if len(self.child_arch_pool) > self.top_K_complete_evaluate:
            old_arch = self.child_arch_pool.popleft()
            logging.debug("Pop arch {} from pool".format(old_arch))

    def sample_and_summarize(self, top_k=10):
        hashs, sample_nums = self.model.summary(
            self.top_K_num_sample)
        ids = []
        for h in hashs:
            try:
                d = self.nasbench_hashs.index(h)
                ids.append(d)
                self.child_arch_pool_append(d)
            except ValueError as e:
                pass
        ids = ids[:self.top_K_complete_evaluate]
        sample_nums = sample_nums[:self.top_K_complete_evaluate]
        logging.info('Top {} Confidence: \n{} \n{}'.format(top_k,ids[:top_k], sample_nums[:top_k]))
        return ids, sample_nums
