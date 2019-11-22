#!/usr/bin/env bash

# Try to submit job to SLURM cluster from local computer.
# Please change the variable here
# Environment things.
USER=YOUR_USERNAME
HOST=TO_SET_A_SLURM_HOST
WORK_DIR=PATH_TO_YOUR_WORKING_DIR
SUBMISSION_DIR=SUBMISSION_DIR_UNDER_WORK_DIR


######## SSH to the code and deploying. #####################

RUN_ID=0
GPU=1
PARTITION=gpu
DAYS=6

# Overwrite Configs accordingly.
[[ ! -z "$2" ]] && export RUN_ID=$2
[[ ! -z "$3" ]] && export GPU=$3
[[ ! -z "$4" ]] && export PARTITION=$4
[[ ! -z "$5" ]] && export DAYS=$5

file=/tmp/$USER-job-run-${RUN_ID}.sh
if [ "$4" = "quadro" ]; then
    CMD="sbatch -p ${PARTITION} -w vcl-gpu4 --gres=gpu:${GPU} -c 6 -t ${DAYS}-0 -o ${SUBMISSION_DIR}/slurm-%N.%j.out -e ${SUBMISSION_DIR}/slurm-%N.%j.err ${file}"
else
    CMD="sbatch -p ${PARTITION} --gres=gpu:${GPU} -c 6 -t ${DAYS}-0 -o ${SUBMISSION_DIR}/slurm-%N.%j.out -e ${SUBMISSION_DIR}/slurm-%N.%j.err ${file}"
fi

# Use SSH to login slurm deployment server, and write the file for execution.
ssh $USER@${HOST} <<EOF
cd ${WORK_DIR}
echo "Working dir`pwd`"
echo "Result save to ${SUBMISSION_DIR}"
echo "#!/bin/bash" > ${file}
printf "$1" >> ${file}
${CMD}
EOF