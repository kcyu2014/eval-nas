# # Implement the one-shot architecture search, newer version...
# # Reference: Single Path One-Shot Neural Architecture Search with Uniform Sampling
# # NOTE: Release this code after, build as fast as possible.
# # Goal is to develop this and test on NasBench-101, small search space.
# # Then to other search space. but now all the experiment is  NASBench-101 centric.
#
# import logging
# import operator
# import os
# from collections import OrderedDict
#
# from functools import partial
#
# import numpy as np
#
# import torch
# import torch.nn as nn
# import torch.utils
# import torchvision.datasets as dset
# from tensorboardX import SummaryWriter
#
# from search_policies.cnn.search_space.nas_bench.sampler import obtain_full_model_spec
# from search_policies.cnn import model as model_module
# import search_policies.cnn.procedures as procedure_ops
# import utils
# from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import NASBench_v2
# from search_policies.cnn.utils import Rank
# from visualization.plot_rank_change import process_rank_data_nasbench
# from visualization.process_data import tensorboard_summarize_list
# import gc
#
#
# class WeightSharingNasBenchNetSearch(CNNSearchPolicy):
#     """
#     Implementation based on DARTS train_search.py.
#         To compare with NAO in future?
#     This is a general search policy on NASbench.
#
#     TO implement:
#         - NAO
#         - ENAS
#         - DARTS
#         - Evolutionary
#         - EA + RL (MoreNAS, FBNas)
#
#     Use this to build a simple network trainer and ranking, to facilitate other usage.
#
#     """
#
#     trained_model_spec_ids = []
#     evaluate_model_spec_ids = []        # ids to evaluate.
#     sample_step_for_evaluation = 2      # Evaluate sample steps.
#     # eval_model_spec_per_epochs = None
#     eval_result = OrderedDict()
#
#     # defining search space
#     nasbench_model_specs = []
#     nasbench_hashs = []
#
#     def __init__(self, args):
#         super(WeightSharingNasBenchNetSearch, self).__init__(args=args)
#         self.args = args
#         self.sample_step_for_evaluation = 2 if not args.debug else 30
#
#         if args.search_space == 'nasbench':
#             self.model_fn = model_module.NasBenchNetSearch
#             self.search_space = None # Add this later.
#             args.model_spec = obtain_full_model_spec(args.num_intermediate_nodes + 2)
#             # initialize nasbench dataset for data analysis.
#             self.process_nasbench()
#         else:
#             raise NotImplementedError("Other search space not supported at this moment.")
#
#         if args.supernet_train_method == 'darts':
#             """ Fundamental baseline training methods
#             sample 1 architecture per batch
#             train supernet
#             Conv op has maximum possible filter channels (== output size of cell)
#             Random a chunk of it.
#             """
#             train_fn = procedure_ops.darts_train_model
#             self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler)
#             self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
#         elif args.supernet_train_method == 'fairnas':
#             """
#             Extend darts training method with FairNas strategy. It is not possible to use directly the FairNAS,
#             but we can extend it into 2 method.
#             """
#             train_fn = procedure_ops.fairnas_train_model_v1
#             self.train_fn = partial(train_fn, args=self.args, architect=None,
#                                     topology_sampler=self.random_sampler,
#                                     op_sampler=self.op_sampler
#                                     )
#             self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
#         elif args.supernet_train_method == 'nao':
#             self.train_fn = procedure_ops.nao_train_model
#             self.eval_fn = procedure_ops.nao_model_validation
#         else:
#             raise NotImplementedError
#
#         self.counter = 0
#
#     def initialize_run(self):
#         args = self.args
#         if not self.args.continue_train:
#             self.sub_directory_path = 'WeightSharingNasBenchNetRandom-{}_SEED_{}'.format(self.args.save, self.args.seed)
#             self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
#             utils.create_exp_dir(self.exp_dir)
#         if self.args.visualize:
#             self.viz_dir_path = utils.create_viz_dir(self.exp_dir)
#
#         if self.args.tensorboard:
#             self.tb_dir = self.exp_dir
#             tboard_dir = os.path.join(self.args.tboard_dir, self.sub_directory_path)
#             self.writer = SummaryWriter(tboard_dir)
#
#         if self.args.debug:
#             torch.autograd.set_detect_anomaly(True)
#
#         # Set logger.
#         self.logger = utils.get_logger(
#             "train_search",
#             file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')),
#             level=logging.INFO if not args.debug else logging.DEBUG
#         )
#         logging.info(f"setting random seed as {args.seed}")
#         utils.torch_random_seed(args.seed)
#         logging.info('gpu number = %d' % args.gpus)
#         logging.info("args = %s", args)
#
#         criterion = nn.CrossEntropyLoss()
#         criterion = criterion.cuda()
#         self._loss = criterion
#
#         train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_length if args.cutout else None)
#         train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
#         test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
#
#         num_train = len(train_data)
#         indices = list(range(num_train))
#         split = int(np.floor(args.train_portion * num_train))
#
#         train_queue = torch.utils.data.DataLoader(
#             train_data, batch_size=args.batch_size,
#             sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#             pin_memory=True, num_workers=2)
#
#         valid_queue = torch.utils.data.DataLoader(
#             train_data, batch_size=args.batch_size,
#             sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
#             pin_memory=True, num_workers=2)
#
#         test_queue = torch.utils.data.DataLoader(
#             test_data, batch_size=args.evaluate_batch_size,
#             shuffle=False, pin_memory=True, num_workers=8)
#
#         # IPython.embed()
#         return train_queue, valid_queue, test_queue, criterion
#
#     def initialize_model(self):
#         """
#         Initialize model, may change across different model.
#         :return:
#         """
#         args = self.args
#         model = self.model_fn(args)
#         if args.gpus > 0:
#             if self.args.gpus == 1:
#                 model = model.cuda()
#                 self.parallel_model = model
#             else:
#                 self.model = model
#                 self.parallel_model = nn.DataParallel(self.model).cuda()
#                 # IPython.embed(header='checking replicas and others.')
#         else:
#             self.parallel_model = model
#         # rewrite the pointer
#         model = self.parallel_model
#
#         logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
#
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             args.learning_rate,
#             momentum=args.momentum,
#             weight_decay=args.weight_decay)
#
#         # scheduler as Cosine.
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, float(args.epochs), eta_min=args.learning_rate_min)
#
#         return model, optimizer, scheduler
#
#     def _change_model_spec(self, model, spec):
#         m = self._call_model_function(model)
#         m.change_model_spec(spec)
#         return model
#
#     def _call_model_function(self, model):
#         if isinstance(model, nn.DataParallel):
#             return model.module
#         else:
#             return model
#
#     def process_nasbench(self, only_hash=True):
#         # initialize run here.
#         # load the nasbench dataset
#         v = self.args.num_intermediate_nodes + 2
#         # TODO merge this into CNNSearchSpace, because for another search space, such ranking is different.
#         self.nasbench = NASBench_v2(os.path.join(self.args.data, 'nasbench/nasbench_only108.tfrecord'),
#                                     config=f'v{v}_e9_op3', only_hash=only_hash)
#         self.nasbench_hashs, self.nasbench_model_specs = self.nasbench.model_hash_rank(full_spec=True)
#         self.rank_by_mid = [i for i in range(0, len(self.nasbench_hashs))]
#         self.evaluate_model_spec_ids = [i for i in range(0, len(self.nasbench_hashs), self.sample_step_for_evaluation)]
#
#     def op_sampler(self, model, architect, args):
#         # build a sampler, returns iterator over m == num_ops architectures with
#         # same topology connection.
#         spec = self.model_spec
#         ops = spec.ops
#
#         avail_ops = self.nasbench.available_ops
#         # tidy up the ops id.
#         try:
#             op_vs_choice = np.tile(np.arange(len(avail_ops)), (len(ops)-2, 1))
#             op_vs_choice = np.apply_along_axis(np.random.permutation, 1, op_vs_choice).transpose()
#             # print('sampled op choice ', op_vs_choice)
#             for i in range(len(avail_ops)):
#                 new_ops = [avail_ops[ind] for ind in op_vs_choice[i]]
#                 spec.ops = ['input',] + new_ops + ['output']
#                 # print("sampled new ops", new_ops)
#                 yield self._change_model_spec(model, spec)
#         except ValueError as e:
#             # if len(ops) - 2 == 0, return this.
#             yield model
#
#     def random_sampler(self, model, architect, args):
#         # rand_spec = obtain_random_spec(args.num_intermediate_nodes + 2)
#         rand_spec_id = int(np.random.randint(0, len(self.nasbench_hashs)))
#         rand_spec = self.nasbench_model_specs[rand_spec_id]
#         # overwrite the current spec and id.
#         self.model_spec_id = rand_spec_id
#         self.model_spec = rand_spec
#
#         # if isinstance(model, nn.DataParallel):
#         #     new_model = model.module.change_model_spec(rand_spec)
#         # else:
#         #     new_model = model.change_model_spec(rand_spec)
#         new_model = self._change_model_spec(model, rand_spec)
#         # this is saved per sample.
#         self.trained_model_spec_ids.append(rand_spec_id)
#         # IPython.embed(header="Check if the model archi change")
#         return new_model
#
#     def run(self):
#         train_queue, valid_queue, test_queue, criterion = self.initialize_run()
#         args = self.args
#         model, optimizer, scheduler = self.initialize_model()
#         fitness_dict = {}
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#
#         for epoch in range(args.epochs):
#             scheduler.step()
#             lr = scheduler.get_lr()[0]
#             logging.info('epoch %d lr %e', epoch, lr)
#
#             # genotype = model.model_spec()
#             # logging.info('current model_spec = %s', genotype)
#
#             # training for each epoch.
#             train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, criterion, optimizer, lr)
#             logging.info(f'Train at epoch {epoch} | loss: {train_obj:8.2f} | top_1_acc: {train_acc:8.2f}')
#
#             # validation
#             valid_acc, valid_obj = self.validate_model(model, valid_queue, self.model_spec_id, self.model_spec)
#             logging.info(f'Valid at epoch {epoch} | loss: {valid_obj:8.2f} | top_1_acc: {valid_acc:8.2f}')
#
#             if self.writer:
#                 self.writer.add_scalar(f'train/loss', train_obj, epoch)
#                 self.writer.add_scalar(f'train/top_1_acc', train_acc, epoch)
#                 self.writer.add_scalar(f'valid/loss', valid_obj, epoch)
#                 self.writer.add_scalar(f'valid/top_1_acc', valid_acc, epoch)
#
#             if epoch % self.args.save_every_epoch == 0:
#                 fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue)
#                 utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
#                 self.save_results(epoch, rank_details=True)
#
#         # add later, return the model specs that is evaluated across the time.
#         # Process the ranking in the end, return the best of training.
#         ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
#         best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
#         return best_id, self.nasbench_model_specs[best_id]
#
#     def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
#         # this is flaw, let me do another evaluating on all possible models.
#
#         # compute all possible batches
#         complete_valid_queue = data_source
#         nb_models = len(self.nasbench_hashs)
#         nb_batch_per_model = max(len(complete_valid_queue) // nb_models, 1)
#         _valid_queue = []
#         valid_model_order = np.random.choice(range(nb_models), nb_models, False)
#
#         if nb_models > len(complete_valid_queue):
#             valid_model_order = valid_model_order[:len(complete_valid_queue)]
#             nb_models = len(complete_valid_queue)
#         total_valid_acc = 0.
#         total_valid_obj = 0.
#
#         for step, val_d in enumerate(complete_valid_queue):
#             _valid_queue.append(val_d)
#             if step % nb_batch_per_model == 0 and step > 0:
#                 _id = valid_model_order[min(int(step / nb_batch_per_model), nb_models - 1)]
#                 current_model = self._change_model_spec(current_model, self.nasbench_model_specs[_id])
#                 current_model.eval()
#                 _valid_acc, _valid_obj = self.eval_fn(_valid_queue, current_model, self._loss)
#                 # logging.info(f"model id {valid_model_order[_id]} acc {_valid_acc} loss {_valid_obj}")
#                 # update the metrics
#                 total_valid_acc += _valid_acc
#                 total_valid_obj += _valid_obj
#                 _valid_queue = []
#         return total_valid_acc / nb_models, total_valid_obj/ nb_models
#
#     def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
#         """
#         Full evaluation of all possible models.
#         :param epoch:
#         :param data_source:
#         :param fitnesses_dict: Store the model_spec_id -> accuracy
#         :return:
#         """
#         nb = self.args.neweval_num_train_batches
#         assert nb > 0
#         if not train_queue:
#             raise ValueError("New evaluation scheme requires a training queue.")
#
#         fitnesses_dict = fitnesses_dict or {}
#         total_avg_acc = 0
#         total_avg_obj = 0
#
#         # rank dict for the possible solutions
#         model_specs_rank = {}
#         model_specs_rank_before = {}
#
#         # as backup
#         ind = 0
#         eval_result = {}
#
#         while ind < len(self.evaluate_model_spec_ids):
#             # model spec id to test.
#             # computing the genotype of the next particle
#             # recover the weights
#             model_spec_id = self.evaluate_model_spec_ids[ind]
#             ind += 1  # increment this.
#             new_model_spec = self.nasbench_model_specs[model_spec_id]
#
#             # selecting the current subDAG in our DAG to train
#             self._change_model_spec(self.parallel_model, new_model_spec)
#             # Reset the weights.
#             # evaluate before train
#             self.logger.info('evaluate the model spec id: {}'.format(model_spec_id))
#             _avg_val_acc, _avg_val_obj = self.eval_fn(data_source, self.parallel_model,
#                                                       criterion=self._loss, verbose=False)
#             eval_result[model_spec_id] = _avg_val_acc, _avg_val_obj
#             # IPython.embed(header=f'Model {model_spec_id}: finish and move to next one, check GPU release')
#             # update the total loss.
#             total_avg_acc += _avg_val_acc
#             total_avg_obj += _avg_val_obj
#
#             # saving the particle fit in our dictionaries
#             fitnesses_dict[model_spec_id] = _avg_val_acc
#             ms_hash = self.nasbench_hashs[model_spec_id]
#             model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
#                                              self.rank_by_mid[model_spec_id])
#             # manual collect the non-used graphs.
#             gc.collect()
#
#         # save the ranking, according to their GENOTYPE but not particle id
#         rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))
#         rank_gens_before = sorted(model_specs_rank_before.items(), key=operator.itemgetter(1))
#         # hash to positions mapping, before training
#         rank_gens_before_pos = {elem[0]: pos for pos, elem in enumerate(rank_gens_before)}
#
#         self.ranking_per_epoch[epoch] = rank_gens
#         self.eval_result[epoch] = eval_result
#
#         self.logger.info('VALIDATION RANKING OF PARTICLES')
#         for pos, elem in enumerate(rank_gens):
#             self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
#                              f'hash: {elem[0]}, pos {pos} vs orig pos {rank_gens_before_pos[elem[0]]}')
#
#         if self.writer:
#             # process data into list.
#             accs_after, objs_after = zip(*eval_result.values())
#             tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
#             tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)
#
#         return fitnesses_dict
#
#     def save_results(self, epoch, rank_details=False):
#         save_data = {
#             'ranking_per_epoch': self.ranking_per_epoch,
#             'trained_model_spec_per_steps': self.trained_model_spec_ids,
#         }
#         # for other to overwrite.
#         return self._save_results(save_data, epoch, rank_details)
#
#
# class WeightSharingNasBenchNewEval(WeightSharingNasBenchNetSearch):
#     """
#     Testing the idea of, before evaluation, train a few batches
#     """
#     eval_result = {}
#     ranking_per_epoch_before = OrderedDict()
#
#     def __init__(self, args):
#         super(WeightSharingNasBenchNewEval, self).__init__(args)
#         self.eval_train_fn = partial(procedure_ops.darts_train_model, args=self.args, architect=None, sampler=None)
#
#     def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
#         """
#         Full evaluation of all possible models.
#         :param epoch:
#         :param data_source:
#         :param fitnesses_dict: Store the model_spec_id -> accuracy
#         :return:
#         """
#         nb = self.args.neweval_num_train_batches
#         assert nb > 0
#         if not train_queue:
#             raise ValueError("New evaluation scheme requires a training queue.")
#
#         fitnesses_dict = fitnesses_dict or {}
#         total_avg_acc = 0
#         total_avg_obj = 0
#
#         # rank dict for the possible solutions
#         model_specs_rank = {}
#         model_specs_rank_before = {}
#
#         # make sure the backup weights on CPU, do not occupy the space
#         backup_weights = self.parallel_model.cpu().state_dict()
#         self.parallel_model.cuda()
#
#         _train_iter = enumerate(train_queue)    # manual iterate the data here.
#         _train_queue = []
#         # as backup
#         ind = 0
#         eval_before_train = {}
#         eval_after_train = {}
#
#
#
#         while ind < len(self.evaluate_model_spec_ids):
#             # model spec id to test.
#             # computing the genotype of the next particle
#             # recover the weights
#             try:
#                 model_spec_id = self.evaluate_model_spec_ids[ind]
#                 ind += 1  # increment this.
#                 new_model_spec = self.nasbench_model_specs[model_spec_id]
#
#                 # selecting the current subDAG in our DAG to train
#                 self._change_model_spec(self.parallel_model, new_model_spec)
#
#                 # Reset the weights.
#                 logging.debug('Resetting parallel model weights ...')
#                 self.parallel_model.load_state_dict(backup_weights)
#                 self.parallel_model.cuda() # make sure this is on GPU.
#                 # IPython.embed(header=f'Model {model_spec_id}: before eval, == checked')
#                 # evaluate before train
#                 _avg_val_acc_before, _avg_val_obj_before = self.eval_fn(data_source, self.parallel_model,
#                                                                         criterion=self._loss, verbose=False)
#                 eval_before_train[model_spec_id] = _avg_val_acc_before, _avg_val_obj_before
#
#                 # TODO process data, make a static function to reuse this.
#                 _batch_count = 0
#                 for batch_id, data in _train_iter:
#                     _batch_count += 1
#                     _train_queue.append(data)
#                     if _batch_count > nb:
#                         # process iteration
#                         break
#
#                 # IPython.embed(header=f'Model {model_spec_id}: finish before eval, == checked')
#                 logging.debug('Train {} batches for model_id {} before eval'.format(_batch_count, model_spec_id))
#                 lr = self.scheduler.get_lr()[0]
#                 org_train_acc, org_train_obj = self.eval_fn(_train_queue, self.parallel_model,
#                                                             criterion=self._loss, verbose=self.args.debug)
#                 # IPython.embed(header=f'finish Model {model_spec_id}: validate train batches')
#                 # only train the specific parallel model, do not sample a new one.
#                 train_acc, train_obj = self.eval_train_fn(
#                     _train_queue, None, self.parallel_model, self._loss, self.optimizer, lr)
#
#                 # clean up the train queue completely.
#                 for d in _train_queue:
#                     del d
#                 _train_queue = []  # clean the data, destroy the graph.
#
#                 logging.debug('-> Train acc {} -> {} | train obj {} -> {} '.format(
#                     org_train_acc, train_acc, org_train_obj, train_obj))
#                 # IPython.embed(header=f'Model {model_spec_id}: finish training == checked 1916MB')
#
#                 self.logger.info('evaluate the model spec id: {}'.format(model_spec_id))
#                 _avg_val_acc, _avg_val_obj = self.eval_fn(data_source, self.parallel_model,
#                                                           criterion=self._loss, verbose=False)
#                 eval_after_train[model_spec_id] = _avg_val_acc, _avg_val_obj
#                 logging.info('eval acc {} -> {} | eval obj {} -> {}'.format(
#                     _avg_val_acc_before, _avg_val_acc, _avg_val_obj_before, _avg_val_obj
#                 ))
#                 # IPython.embed(header=f'Model {model_spec_id}: finish and move to next one, check GPU release')
#                 # update the total loss.
#                 total_avg_acc += _avg_val_acc
#                 total_avg_obj += _avg_val_obj
#
#                 # saving the particle fit in our dictionaries
#                 fitnesses_dict[model_spec_id] = _avg_val_acc
#                 ms_hash = self.nasbench_hashs[model_spec_id]
#                 model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
#                                                  self.rank_by_mid[model_spec_id])
#                 model_specs_rank_before[ms_hash] = Rank(_avg_val_acc_before, _avg_val_obj_before, model_spec_id,
#                                                         self.rank_by_mid[model_spec_id])
#                 # manual collect the non-used graphs.
#                 gc.collect()
#
#             except StopIteration as e:
#                 _train_iter = enumerate(train_queue)
#                 logging.debug("Run out of train queue, {}, restart ind {}".format(e, ind-1))
#                 ind = ind - 1
#
#         # IPython.embed(header="Checking the results.")
#         # save the ranking, according to their GENOTYPE but not particle id
#         rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))
#         rank_gens_before = sorted(model_specs_rank_before.items(), key=operator.itemgetter(1))
#         # hash to positions mapping, before training
#         rank_gens_before_pos = {elem[0]: pos for pos, elem in enumerate(rank_gens_before)}
#
#         self.ranking_per_epoch[epoch] = rank_gens
#         self.ranking_per_epoch_before[epoch] = rank_gens_before
#         self.eval_result[epoch] = (eval_before_train, eval_after_train)
#
#         self.logger.info('VALIDATION RANKING OF PARTICLES')
#         for pos, elem in enumerate(rank_gens):
#             self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
#                              f'hash: {elem[0]}, pos {pos} vs orig pos {rank_gens_before_pos[elem[0]]}')
#
#         if self.writer:
#             # process data into list.
#             accs_before, objs_before = zip(*eval_before_train.values())
#             accs_after, objs_after = zip(*eval_after_train.values())
#             tensorboard_summarize_list(accs_before, writer=self.writer, key='neweval_before/acc', step=epoch, ascending=False)
#             tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
#             tensorboard_summarize_list(objs_before, writer=self.writer, key='neweval_before/obj', step=epoch)
#             tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)
#
#         return fitnesses_dict
#
#     def save_results(self, epoch, rank_details=False):
#         save_data = {
#             'ranking_per_epoch': self.ranking_per_epoch,
#             'rank_per_epoch_before': self.ranking_per_epoch_before,
#             'trained_model_spec_per_steps': self.trained_model_spec_ids,
#             'eval_result': self.eval_result
#         }
#         # for other to overwrite.
#         return self._save_results(save_data, epoch, rank_details)
#
#     def _save_ranking_results(self, save_data, epoch):
#         """
#         Save the ranking results if necessary.
#         :param save_data:
#         :param epoch:
#         :return:
#         """
#         # Plot the ranking data
#         fig = process_rank_data_nasbench(save_data, os.path.join(self.exp_dir, 'rank_change-{}.pdf'.format(epoch)))
#
#         # Compute Kendall tau for every epochs and save them into result.
#         kd_tau = self._compute_kendall_tau(self.ranking_per_epoch)
#         kd_tau_before = self._compute_kendall_tau(self.ranking_per_epoch_before)
#         save_data['kendaltau'] = kd_tau
#         save_data['kendaltau_before'] = kd_tau_before
#
#         if self.writer is not None:
#             # p: list or Rank(acc, obj, geno_id, gt_rank)
#             p = sorted([elem[1] for elem in self.ranking_per_epoch[epoch]], key=operator.itemgetter(2))
#             tensorboard_summarize_list([e[0] for e in p], self.writer, 'eval_acc', epoch, ascending=False)
#             tensorboard_summarize_list([e[1] for e in p], self.writer, 'eval_obj', epoch, ascending=True)
#             self.writer.add_scalar('eval_kendall_tau', kd_tau[10000000][0].correlation, epoch)
#             self.writer.add_scalar('eval_kendall_tau_before', kd_tau_before[10000000][0].correlation, epoch)
#             self.writer.add_figure(tag='rank-diff', figure=fig, global_step=epoch)
#         return save_data
