# implement the evaluation query of a trained model.
import copy
import logging
import random
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
from scipy.stats import kendalltau
import glob

from search_policies.cnn.search_space.api import CNNSearchSpace
from search_policies.cnn.search_space.nas_bench.model_search import NasBenchNetSearch
from search_policies.cnn.search_space.nas_bench.nasbench_api_v2 import ModelSpec_v2
from search_policies.cnn.utils import AverageMeter, accuracy


def _summarize_shared_train(curr_step, loss,
                            task_loss,
                            acc=0,
                            task_acc=0):
    print(f'arch_id {curr_step:3d} '
          f'| valid {loss:.2f} '
          f'({acc:8.2f}) '
          f'| eval {task_loss:.2f} '
          f'|({task_acc: 8.4f}) '
          )


def load_nasbench_supernet_and_proceed(path, args):
    net = NasBenchNetSearch(
        args,
    )
    state_dict = torch.load(path)
    print(state_dict)
    if args.gpus > 1:
        net = DataParallel(net)
    net.load_state_dict(state_dict)
    return net


def mutate_spec(old_spec, nasbench, mutation_rate=1.0, num_vertices=5, num_ops=3):
    """Computes a valid mutated spec from the old_spec."""
    while True:
        new_matrix = copy.deepcopy(old_spec.original_matrix)
        new_ops = copy.deepcopy(old_spec.original_ops)

        # In expectation, V edges flipped (note that most end up being pruned).
        edge_mutation_prob = mutation_rate / num_vertices
        for src in range(0, num_vertices - 1):
            for dst in range(src + 1, num_vertices):
                if random.random() < edge_mutation_prob:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        # In expectation, one op is resampled.
        op_mutation_prob = mutation_rate / num_ops
        for ind in range(1, num_vertices - 1):
            if random.random() < op_mutation_prob:
                available = [o for o in nasbench.available_ops if o != new_ops[ind]]
                new_ops[ind] = random.choice(available)

        new_spec = ModelSpec_v2(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return new_spec


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def evaluate_single_model(model, criterion, valid_queue, debug=False, verbose=False, report_freq=50):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if debug:
                if step > 10:
                    print("Break after 10 batch")
                    break
            input = input.cuda()
            target = target.cuda()
            # target = target.cuda(async=True)
            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % report_freq == 0 and step > 0 and verbose:
                logging.info('valid | step %03d | loss %e | acc %f | acc-5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def run_evolution_search_space_with_supernet(
        supernet,
        search_space,
        eval_queue,
        test_queue,
        max_evaluate_models=100,
        population_size=50,
        tournament_size=10,
        mutation_rate=1.0,
        fitness_dict=None,
        eval_fitness_dict=None,
):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    if not isinstance(search_space, CNNSearchSpace):
        raise NotImplementedError("only support CNNSearchSpace")

    times, best_valids, best_tests = [0.0], [0.0], [0.0]
    population = []   # (validation, spec) tuples
    time_spent = 0

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    print("Evaluation algorithms start to build the population...")
    for _ in range(population_size):

        spec_id, spec = search_space.random_topology()
        supernet.module.change_model_spec(model_spec=spec)
        if fitness_dict and spec_id in fitness_dict.keys():
            acc, obj = fitness_dict[spec_id]
            test_acc, test_obj = eval_fitness_dict[spec_id]
        else:
            acc, obj = evaluate_single_model(supernet, torch.nn.CrossEntropyLoss(), eval_queue, verbose=True)
            test_acc, test_obj = evaluate_single_model(supernet, torch.nn.CrossEntropyLoss(), test_queue, verbose=True)
            if fitness_dict:
                fitness_dict[spec_id] = acc, obj
                eval_fitness_dict[spec_id] = test_acc, test_obj
        time_spent += 1
        times.append(time_spent)
        population.append((acc, spec))
        _summarize_shared_train(spec_id, obj, test_obj, acc, test_acc)
        if acc > best_valids[-1]:
            best_valids.append(acc)
            best_tests.append(acc)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])

        if time_spent > max_evaluate_models:
            break

    print("Evaluation algorithm built population size {}".format(time_spent))
    print("Start evolution...")
    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_spec(best_spec, search_space.nasbench, mutation_rate)
        new_spec_id = search_space.hashs.index(new_spec.hash_spec())
        supernet.module.change_model_spec(model_spec=new_spec)

        if fitness_dict and new_spec_id in fitness_dict.keys():
            acc, obj = fitness_dict[new_spec_id]
            test_acc, test_obj = eval_fitness_dict[new_spec_id]
        else:
            acc, obj = evaluate_single_model(supernet, torch.nn.CrossEntropyLoss(), eval_queue, verbose=True)
            test_acc, test_obj = evaluate_single_model(supernet, torch.nn.CrossEntropyLoss(), test_queue, verbose=True)
            if fitness_dict:
                fitness_dict[new_spec_id] = acc, obj
                eval_fitness_dict[new_spec_id] = test_acc, test_obj

        _summarize_shared_train(new_spec_id, obj, test_obj, acc, test_acc)
        # In regularized evolution, we kill the oldest individual in the population.
        population.append((acc, new_spec))
        population.pop(0)

        if acc > best_valids[-1]:
            best_valids.append(acc)
            best_tests.append(test_acc)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
        time_spent += 1
        times.append(time_spent)
        if time_spent > max_evaluate_models:
            break

    return times, best_valids, best_tests


def run_random_search(
        supernet,
        search_space,
        eval_queue,
        test_queue,
        max_evaluate_models=100,
):
    return run_evolution_search_space_with_supernet(
        supernet,
        search_space,
        eval_queue,
        test_queue,
        max_evaluate_models=max_evaluate_models,
        population_size=max_evaluate_models,
        tournament_size=0,
        mutation_rate=0.0
    )


# Compare the mean test accuracy along with error bars.
def plot_data(data, color, label, gran=1, max_budget=100):
    """Computes the mean and IQR fixed time steps."""
    xs = range(0, max_budget + 1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = gran
    while cur < max_budget + 1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and
                   data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat] - 1]
            prev_test = data[repeat][2][pointers[repeat] - 1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][2][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_test - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)


def compute_kdt(pred_rank):
    gt_rank = list(reversed(sorted(pred_rank)))
    return kendalltau(gt_rank, pred_rank).correlation


def manual_rechecking_kendall_tau_for_two_folders(target, baseline, verbose=False, load_perf=False, report=True):
    pool1 = [epoch for epoch in glob.glob(target + '/*SEED*/eval_arch_pool.*') if 'perf' not in epoch]
    pool2 = [epoch for epoch in glob.glob(baseline + '/*SEED*/eval_arch_pool.*') if 'perf' not in epoch]
    if load_perf:
        perf_files_1 = [epoch for epoch in glob.glob(target + '/*SEED*/eval_arch_pool.*') if 'perf'  in epoch]
        perf_files_2 = [epoch for epoch in glob.glob(baseline + '/*SEED*/eval_arch_pool.*') if 'perf' in epoch]

    print('pool 1 ',len(pool1),'  pool 2 ', len(pool2)) 
    p1 = [int(epoch.split('eval_arch_pool.')[1]) for epoch in pool1]
    p2 = [int(epoch.split('eval_arch_pool.')[1]) for epoch in pool2]
    epoch_eval_dict1 = {}
    epoch_eval_dict2 = {}
    for ind, epoch in enumerate(p1):
        if epoch in p2:
            with open(pool1[ind], 'r') as f1:
                with open(pool2[p2.index(epoch)], 'r') as f2:
                    a = f1.readlines()
                    b = f2.readlines()
                    epoch_eval_dict1[epoch] = [int(r.split(',')[0]) for r in a]
                    epoch_eval_dict2[epoch] = [int(r.split(',')[0]) for r in b]
            if load_perf:
                with open(perf_files_1[ind], 'r') as f1:
                    with open(perf_files_2[p2.index(epoch)], 'r') as f2:
                        a = f1.readlines()
                        b = f2.readlines()
                        epoch_eval_dict1[f'perf_{epoch}'] = [float(r.split(',')[0]) for r in a]
                        epoch_eval_dict2[f'perf_{epoch}'] = [float(r.split(',')[0]) for r in b]

    sorted_keys = sorted(list([k for k in epoch_eval_dict2.keys() if not isinstance(k, str)]))
    if report:
        for epoch in sorted_keys:
            comparable = set(epoch_eval_dict1[epoch]) == set(epoch_eval_dict2[epoch])
            print('Epoch ', epoch, ' kendall tau: \tbaseline\t', compute_kdt(epoch_eval_dict2[epoch]),
                  '\trankloss\t',compute_kdt(epoch_eval_dict1[epoch]), 'comparable ', comparable)
            if not comparable and verbose:
                print('baseline eval archs', len(epoch_eval_dict2[epoch]))
                print('target eval archs ', len(epoch_eval_dict1[epoch]))
                print("difference archs ", set(epoch_eval_dict1[epoch]) - set(epoch_eval_dict2[epoch]))

    return epoch_eval_dict1, epoch_eval_dict2

    
def check_how_many_archs_are_always_the_same(epoch_arch_dict):
    unique_set = None
    for epoch in epoch_arch_dict.keys():
        if unique_set is None:
            unique_set = set(epoch_arch_dict[epoch])
        unique_set.intersection_update(set(epoch_arch_dict[epoch]))
    print('Archs always in eval pool number: ',len(unique_set))
    return list(unique_set)

import numpy as np
def make_test_rank_file(hashs, perfs, hash_perfs, test=0, verbose=False ):
    sorted_indices = np.argsort(perfs)
    s_h, s_p = [hashs[i] for i in sorted_indices], [perfs[i] for i in sorted_indices]
    return s_h, s_p
