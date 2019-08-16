#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

DATA_DIRECTORY="downloaded_data"


if ! mkdir ${DATA_DIRECTORY} 2>/dev/null; then
    echo ${DATA_DIRECTORY} "directory already exists. Please specify a non-existing directory just to be safe."
    exit
fi
cd ${DATA_DIRECTORY}
echo "Downloading"
wget https://dl.fbaipublicfiles.com/splitnet/splitnet_dataset.tar
echo "Unzipping"
tar -xf splitnet_dataset.tar
mv splitnet_dataset/* .
rmdir splitnet_dataset
rm splitnet_dataset.tar
echo "Success"
cd ..

