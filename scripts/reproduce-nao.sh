#!/usr/bin/env bash
# Template file for submit jobs or testing with interactive jobs.

PYTHON=python

EXPERIMENT="cnn-fullspace"
SUB_EXP="cifar_original_nao"
SUBMIT_FILE='slurm-submit.sh'



if [ $1 == "train_from_scratch" ]; then
  SEED=1268
  SUB_EXP="NAO-train-from-scratch"
  if [ "$2" = "debug" ]; then
    $PYTHON train_cifar.py --epochs=10 \
              --batch_size 96 \
              --init_channels 36 \
              --layers 20 \
              --policy nao \
              --arch ${SEED} \
              --save debug/${EXPERIMENT}_${SUB_EXP} \
              --auxiliary \
              --cutout
  fi
  if [ "$2" -eq 1 ]; then
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
                  --policy nao \
                  --arch ${SEED} \
                  --save ${EXPERIMENT}/${SUB_EXP}/${EXPNAME} \
                  --auxiliary \
                  --cutout \
              > logs/${EXPERIMENT}/${SUB_EXP}_$EXPNAME.log \
               2>&1"
            cmdALL="echo '$cmd' && $cmd"
            bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS
        done

  else
      exit
  fi

fi