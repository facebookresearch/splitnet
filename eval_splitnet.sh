#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

DATASET="suncg"
TASK="pointnav"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4



LOG_LOCATION="output_files/"${TASK}"/"${DATASET}"/splitnet_pretrain_supervised_rl"

python eval_splitnet.py \
    --log-prefix ${LOG_LOCATION} \
    --dataset ${DATASET} \
    --task ${TASK} \
    --encoder-network-type ShallowVisualEncoder \
    --num-processes 4 \
    --data-subset val \
    --no-save-checkpoints \
    --no-weight-update \
    --no-tensorboard \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \

