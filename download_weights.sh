#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

WEIGHT_DIRECTORY="output_files"


if ! mkdir ${WEIGHT_DIRECTORY} 2>/dev/null; then
    echo ${WEIGHT_DIRECTORY} "directory already exists. Please specify a non-existing directory just to be safe."
    exit
fi
cd ${WEIGHT_DIRECTORY}
echo "Downloading"
wget https://dl.fbaipublicfiles.com/splitnet/splitnet.tar.gz
echo "Unzipping"
tar -zxf splitnet.tar.gz
mv splitnet/* .
rm -rf splitnet
rm -rf splitnet.tar.gz
echo "Success"
cd ..

