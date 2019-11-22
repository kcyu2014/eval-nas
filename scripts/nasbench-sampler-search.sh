#!/usr/bin/env bash
# Template file for submit jobs or testing with interactive jobs.

PYTHON=python
PARTITION=gpu
SUBMIT_FILE='slurm-submit.sh'
DAYS=20
EXPERIMENT="cnn-nasbench"

if [ $1 = "debug" ]; then

policy='nao'
SUB_EXP="cifar_nasbench_${policy}"
node=5
$PYTHON cnn_search_main.py --epochs=4 \
              --save_every_epoch=1 \
              --search_space nasbench \
              --search_policy $policy \
              --num_intermediate_nodes ${node} \
              --nasbenchnet_vertex_type ${policy}_ws \
              --supernet_train_method darts \
              --seed_range_start 1268 \
              --seed_range_end 1269 \
              --batch_size 16 \
              --evaluate_batch_size 16 \
              --tensorboard \
              --gpus 4 \
              --test_dir experiments/debug/${EXPERIMENT}_${SUB_EXP}-normal-test1-node${node} \
              --debug
fi


if [ $1 = "search" ]; then

    # Experiments extensively on four policys.
    # Note that this intermediate_nodes = node (in paper) - 2.
    NUM_SEED_TO_USE=10  # reduce this number to save time.
    declare -a arr1=("2" "3" "4" "5")
    declare -a arr2=("enas" "nao" "darts" "fbnet")

    for ((k=0;k<${NUM_SEED_TO_USE};k=k+1));do
    SEED=$((1268 + $k))
    SEED_END=$((1268 + $k + 1))
    for ((i=0;i<${#arr1[@]};i=i+1)); do
        node=${arr1[i]}
        for((j=0;j<${#arr2[@]};j=j+1));do
            policy=${arr2[j]}
            SUB_EXP="cifar_nasbench_${policy}"
            cmd="$PYTHON cnn_search_main.py --epochs=200 \
              --save_every_epoch=20 \
              --search_space nasbench \
              --search_policy ${policy} \
              --num_intermediate_nodes ${node} \
              --nasbenchnet_vertex_type ${policy}_ws \
              --supernet_train_method darts \
              --seed_range_start ${SEED} \
              --seed_range_end ${SEED_END} \
              --batch_size 256 \
              --evaluate_batch_size 64 \
              --tensorboard \
              --gpus $GPU \
              --test_dir experiments/${EXPERIMENT}/${SUB_EXP}/baseline-new-node${node}-${SEED} \
              > logs/${EXPERIMENT}/${SUB_EXP}_baseline-new-node${node}-${SEED}.log \
              2>&1"
            cmdALL="$cmd"
            bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS
        done
    done
    done
fi
