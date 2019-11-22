#!/usr/bin/env bash
# Template file for submit jobs or testing with interactive jobs.

PYTHON=python

EXPERIMENT="cnn-fullspace"
SUB_EXP="cifar_original_enas"
SUBMIT_FILE='slurm-submit.sh'


if [ $1 = "search" ]; then

  if [ $2 = "debug" ]; then
  $PYTHON cnn_search_main.py --epochs=150 \
                --search_space original \
                --search_policy enas \
                --seed_range_start 1270 \
                --seed_range_end 1271 \
                --test_dir experiments/debug/${EXPERIMENT}_${SUB_EXP}

  else
    declare -a arr1=("1268" "1269" "1271" "1270" "1272" "1273" "1274" "1275" "1276" "1277")
    declare -a arr2=("1269" "1270" "1272" "1271" "1273" "1274" "1275" "1276" "1277" "1278")
    for ((i=0;i<${#arr1[@]};i=i+1)); do
        start=${arr1[i]}
        end=${arr2[i]}
        EXPNAME=${start}-${end}
        cmd="$PYTHON cnn_search_main.py --epochs=150 \
          --search_space original \
          --search_policy enas \
          --num_intermediate_nodes 5 \
          --seed_range_start ${start} \
          --seed_range_end ${end} \
          --test_dir experiments/${EXPERIMENT}/${SUB_EXP}/$EXPNAME \
          > logs/${EXPERIMENT}/${SUB_EXP}_$EXPNAME.log \
           2>&1"
        cmdALL="echo '$cmd' && $cmd"
        bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS
    done
   fi

elif [ $1 == "train_from_scratch" ]; then
  SEED=1268
  SUB_EXP="ENAS-train-from-scratch"
  if [ "$2" = "debug" ]; then
    $PYTHON train_cifar.py --epochs=10 \
              --batch_size 96 \
              --init_channels 36 \
              --layers 20 \
              --policy enas \
              --arch ${SEED} \
              --save debug/${EXPERIMENT}_${SUB_EXP} \
              --auxiliary \
              --cutout
  else

  start=1268
  end=1278

  for ((i=$start;i<${end};i=i+1)); do
      SEED=${i}
      EXPNAME="final-model-seed_${SEED}"
      cmd="$PYTHON train_cifar.py --epochs=600 \
            --batch_size 96 \
            --eval_batch_size 64 \
            --init_channels 36 \
            --layers 20 \
            --policy enas \
            --arch ${SEED} \
            --save ${EXPERIMENT}/${SUB_EXP}/${EXPNAME} \
            --auxiliary \
            --cutout \
        > logs/${EXPERIMENT}/${SUB_EXP}_$EXPNAME.log \
         2>&1"
      cmdALL="echo '$cmd' && $cmd"
      bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS
  done

  fi

fi