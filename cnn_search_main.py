import glob

import IPython
import time
import torch

from search_policies.cnn import cnn_search_configs as configs
# IPython.embed()
import os
import pickle

from search_policies.cnn.cnn_search_policy_factory import CNNCellSearchPolicyFactory

from utils import create_exp_dir, create_dir, DictAttr
import search_policies.cnn.search_space.search_space_utils as search_space
import random


args = configs.parser.parse_args()

args_dict = DictAttr(vars(args))

print(f">++++ USING PYTORCH VERSION {torch.__version__} ++++++")

# creating the main directory
if args.main_path is None:
    if args.model_spec_hash is not None:
        # evaluate code should be called.
        args.main_path = 'modelspec_{}-nodes_{}-SEED{}_{}-cuda10{}'.format(
            args.model_spec_hash,
            args.num_intermediate_nodes,
            args.seed_range_start,
            args.seed_range_end,
            torch.cuda.get_device_name(0).replace(' ', '-').replace('(', '').replace(')', ''),
        )
    else:
        args.main_path = 'SEED{}_{}-cuda10{}'.format(
            args.seed_range_start,
            args.seed_range_end,
            torch.cuda.get_device_name(0).replace(' ', '-').replace('(', '').replace(')', ''),
        )
    args.main_path = os.path.join(args.test_dir, args.main_path)
    print(">=== Create main path ====")
    # Copying all the path py file into the folder.
    create_exp_dir(args.main_path, scripts_to_save=glob.glob('*.py') + glob.glob('search_policies'))
else:
    print(">=== USING Existing MAIN PATH ====")

print(args.main_path)

# creating the tensorboard directory
tboard_path = os.path.join(args.main_path, args.tboard_dir)
args.tboard_dir = tboard_path
create_dir(tboard_path)

# list for the final dictionary to be used for dataframe analysis
model_spec_ids = []
model_specs = []
validation_acc = []
seeds = []

range_tests = [value for value in range(args.seed_range_start, args.seed_range_end)]

all_evaluations_dicts = {}

for value in range_tests:
    # enumerate all testing seeds.

    temp_eval_dict = {}
    # modifying the configs
    args.seed = value
    # set the seed here
    random.seed(args.seed)

    print(f">===== Experiment with SEED {value} =====")
    try:
        if args.model_spec_hash is not None:
            chosen_mspec_id, chosen_mspec = search_space.get_fixed_model_spec(args)
        else:
            # creating the search strategy object from the Search Policy Factory
            search_policy = CNNCellSearchPolicyFactory.factory(args)

            print(">===== Start the search process ... =====")
            # getting the best architecture from the chosen policy
            chosen_mspec_id, chosen_mspec = search_policy.run()
            print(">===== Finish search, delete the search policy to release all GPU memory =====")

        if args.evaluate_after_search:
            print(">===== Start the evaluation process ... =====")
            # initializing the Evaluation routine to evaluate the chosen genotype
            raise NotImplementedError("Not yet implemented for evaluation from after search. ")

            # training from scratch the chosen genotype
            train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, test_loss, test_acc = evaluation_phase.run()

            temp_eval_dict['train_losses'] = train_loss_list
            temp_eval_dict['train_accs'] = train_acc_list
            temp_eval_dict['valid_losses'] = valid_loss_list
            temp_eval_dict['valid_accs'] = valid_acc_list
            temp_eval_dict['test_loss'] = test_loss
            temp_eval_dict['test_acc'] = test_acc

            # saving the metrics dict in the genotypes dictionary
            all_evaluations_dicts[value] = temp_eval_dict

            # saving infos regarding the ranking of the genotypes
            seeds.append(value)
            model_spec_ids.append(chosen_mspec_id)
            model_specs.append(chosen_mspec.recurrent)
            validation_acc.append(valid_acc_list[-1])
        else:
            model_spec_ids.append(chosen_mspec_id)
            model_specs.append(chosen_mspec)

    except Exception as e:
        # Handle the exception and save all necessary data.
        print(e)
        print('ERROR ENCOUNTERED IN THIS SEARCH, MOVING ON NEXT SEARCH')

        rank_dataframe_dict = {'seeds': seeds,
                               'gen_ids': model_spec_ids,
                               'genotypes': model_specs,
                               'last_valid_acc': validation_acc
                               }

        rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                         args.evaluation_seed,
                                                                                                         chosen_mspec_id))
        if args.evaluate_after_search:
            with open(rank_save_path, 'wb') as handle:
                pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                                   args.evaluation_seed,
                                                                                                                   chosen_mspec_id))

            with open(seeds_save_path, 'wb') as handle:
                pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raise
            # continue

# saving with pickle our dictionaries #
rank_dataframe_dict = {'seeds': seeds,
                       'gen_ids': model_spec_ids,
                       'genotypes': model_specs,
                       'last_valid_acc': validation_acc
                       }

rank_save_path = os.path.join(args.main_path, 'rank_dataframe_dict_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                 args.evaluation_seed,
                                                                                                 chosen_mspec_id))
print(rank_dataframe_dict)

if args.evaluate_after_search:
    with open(rank_save_path, 'wb') as handle:
        pickle.dump(rank_dataframe_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    seeds_save_path = os.path.join(args.main_path, 'seeds_metrics_dictionary_{}_eval_{}_geno_id_{}'.format(args.seed_range_start,
                                                                                                           args.evaluation_seed,
                                                                                                           chosen_mspec_id))

    with open(seeds_save_path, 'wb') as handle:
        pickle.dump(all_evaluations_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)





