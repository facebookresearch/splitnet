#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

python supervised_learning/imagenet_pretrain.py \
    --log-prefix output_files/imagenet/pretrain \
    --workers 40 \
    --pytorch-gpu-ids 3,4,5,6 \
    --batch-size 256 \
    --data /raid/xkcd/imagenet/raw-data/ \
