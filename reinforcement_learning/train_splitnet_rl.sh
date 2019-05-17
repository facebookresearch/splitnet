#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="gibson"
TASK="pointnav"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Copy this file to log location for tracking the flags used.
LOG_LOCATION="output_files/"${TASK}"/"${DATASET}"/imagenet_pretrain_supervised_rl"
mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python train_splitnet.py \
    --algo ppo \
    --encoder-network-type ShallowVisualEncoder \
    --use-gae \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --entropy-coef 0.01 \
    --use-linear-clip-decay \
    --task pointnav \
    --log-prefix ${LOG_LOCATION} \
    --eval-interval 250000 \
    --num-processes 9 \
    --num-forward-rollout-steps 128 \
    --num-mini-batch 1 \
    --dataset ${DATASET} \
    --data-subset train \
    --pytorch-gpu-ids 0,1,2,3 \
    --render-gpu-ids 1,2,3 \
    --freeze-encoder-features \
    --freeze-visual-decoder-features \
    --no-visual-loss \
    --freeze-motion-decoder-features \
    --no-motion-loss \
