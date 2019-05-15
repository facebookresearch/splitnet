#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import numpy as np


def get_all_dirs(path):
    return sorted(filter(os.path.isdir, glob.glob(os.path.join(path, "*"))))


data_dir = "logs/HabitatPointNav/shallow"
high_level_dirs = get_all_dirs(data_dir)
expected_num_lines = dict(suncg=908, gibson=1003, mp3d=498)


def parse_all_results():
    for dir_on in high_level_dirs:
        dataset = dir_on.split(os.sep)[-1]
        if dataset not in expected_num_lines:
            continue
        methods = get_all_dirs(dir_on)
        for method in methods:
            files = sorted(glob.glob(os.path.join(method, "results", "train", "*.csv")))
            best_spl = 0
            best_stats = None
            for file in files:
                data = [line.strip().split(",") for line in open(file)]
                if len(data) != expected_num_lines[dataset]:
                    continue
                # episode_id, num_steps, reward, spl, success, start_geodesic_distance, end_geodesic_distance, delta_geodesic_distance
                keys = data[2]

                spl_row = keys.index("spl")
                success_row = keys.index("success")
                vals = np.array([list(map(float, data_on)) for data_on in data[3:]])
                mean_spl = np.mean(vals[:, spl_row])
                if best_spl < mean_spl:
                    best_spl = mean_spl
                    best_stats = dict(
                        spl="%.3f" % mean_spl,
                        spl_std="%.3f" % np.std(vals[:, spl_row]),
                        success="%.3f" % np.mean(vals[:, success_row]),
                        success_std="%.3f" % np.std(vals[:, success_row]),
                        data=vals,
                    )
            if best_stats is not None:
                print(method[len(data_dir) + 1 :], best_stats)


if __name__ == "__main__":
    parse_all_results()
