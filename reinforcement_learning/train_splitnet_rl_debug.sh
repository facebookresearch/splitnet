#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="mp3d"
TASK="pointnav"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python train_splitnet.py \
    --algo ppo \
    --encoder-network-type ShallowVisualEncoder \
    --log-prefix output_files/${TASK}/${DATASET}/splitnet_rl \
    --use-gae \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --num-processes 2 \
    --num-forward-rollout-steps 8 \
    --num-mini-batch 1 \
    --dataset ${DATASET} \
    --data-subset train_small \
    --task ${TASK} \
    --freeze-encoder-features \
    --freeze-visual-decoder-features \
    --no-visual-loss \
    --freeze-motion-decoder-features \
    --no-motion-loss \
    --no-tensorboard \
    --no-save-checkpoints \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \
    --use-multithreading \
    --debug \
