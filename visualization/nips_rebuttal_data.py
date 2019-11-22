from operator import itemgetter
from scipy.stats import kendalltau



def clean_arch_pool_with_performance(pool, perf):
    new_arch_pool = dict()
    new_arch_perf = dict()
    for a, p in zip(pool, perf):
        if a not in new_arch_pool.keys():
            new_arch_pool[a] = 1
            new_arch_perf[a] = [p]
        else:
            new_arch_pool[a] += 1
            new_arch_perf[a].append(p)
    return new_arch_pool, new_arch_perf


def compute_arch_pool_final_performance(hashs, arch_pool_perf, nasbench, seed, top_k=5, sample_kdt=20):
    pool, perf = clean_arch_pool_with_performance(hashs, arch_pool_perf)
    perf_avg = {k: sum(a) / len(a) for k, a in perf.items()}

    # print(pool)
    # print(perf_avg)
    final_result = list(reversed(sorted(perf_avg.items(), key=itemgetter(1))))
    best_3_model_hash = [a[0] for a in final_result][:top_k]
    
    # this is the selected ones, but not the entire list.
    gt_perf = []
    gt_perf_std = []
    for h in best_3_model_hash:
        try:
            query = nasbench.query_hash(h)
            gt_perf.append(query['validation_accuracy'])
            gt_perf_std.append(query['validation_accuracy_std'])
        except Exception as e:
            print(e)
            pass

    # process Kendall tau
    hash_rank = nasbench.model_hash_rank()
    final_hashs = [h for h, _ in final_result]
    sample_step = max(int(len(final_hashs) / sample_kdt), 1)

    # only sample 20 times.
    final_hashs = [final_hashs[i] for i in range(0, len(final_hashs), sample_step)]
    estimate_rank = list(range(len(final_hashs)))
    gt_rank = []

    for ind, fh in enumerate(final_hashs):
        try:
            r = hash_rank.index(fh)
            gt_rank.append(r)
        except:
            estimate_rank.remove(ind)
            # pass
    # print(gt_rank)
    # print("Kendall Tau computation ranking")
    # print(gt_rank)
    # print(estimate_rank)
    # process rank_compute for kdtau
    gt_rank_compute = [gt_rank.index(g) for g in reversed(sorted(gt_rank))]

    # print(gt_rank_compute)
    kdt = kendalltau(gt_rank_compute, estimate_rank)
    print(f"Evaluating result for {seed}: Top {top_k} model accuracy {gt_perf[:top_k]} "
          f"and std {gt_perf_std[:top_k]}. Kendall Tau {kdt}.")
    return gt_perf[:top_k], gt_perf_std[:top_k], kdt