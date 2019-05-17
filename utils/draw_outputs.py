#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import string

import cv2
import numpy as np
from dg_util.python_utils.drawing import draw_probability_hist
from habitat.utils.visualizations import maps


def obs_to_images(obs):
    img = obs["rgb"].copy()
    images = [img.transpose(0, 2, 3, 1)]

    # Draw top down view
    if "visited_grid" in obs:
        top_down_map = obs["visited_grid"][0, ...]
    elif "top_down_map" in obs:
        top_down_map = maps.colorize_topdown_map(obs["top_down_map"]["map"])
        map_size = 1024
        original_map_size = top_down_map.shape[:2]
        if original_map_size[0] > original_map_size[1]:
            map_scale = np.array((1, original_map_size[1] * 1.0 / original_map_size[0]))
        else:
            map_scale = np.array((original_map_size[0] * 1.0 / original_map_size[1], 1))

        new_map_size = np.round(map_size * map_scale).astype(np.int32)
        # OpenCV expects w, h but map size is in h, w
        top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

        map_agent_pos = obs["top_down_map"]["agent_map_coord"]
        map_agent_pos = np.round(map_agent_pos * new_map_size / original_map_size).astype(np.int32)
        top_down_map = maps.draw_agent(
            top_down_map, map_agent_pos, obs["heading"] - np.pi / 2, agent_radius_px=top_down_map.shape[0] / 40
        )
    else:
        top_down_map = None

    normalize = [True]
    titles = [
        (
            ("Method: %s" % obs["method"].replace("_", " ")),
            ("Step: %03d Reward: %.3f" % (obs["step"][0], obs.get("reward", [0])[0])),
            ("Action: %s" % string.capwords(obs["action_taken_name"].replace("_", " "))),
        )
    ]
    images.append(top_down_map)
    if "visited" in obs:
        titles.append((("Visited Cube Count:  %d" % obs["visited"][0]),))
    elif "distance_from_start" in obs:
        titles.append("Geo Dist From Origin: %.3f" % obs["distance_from_start"][0])
    elif "pointgoal" in obs:
        titles.append(
            (("Euc Dist:  %.3f" % obs["pointgoal"][0, 0]), ("Geo Dist: %.3f" % obs["goal_geodesic_distance"][0]))
        )

    normalize.append(False)

    for key, val in obs.items():
        if key == "depth" or key == "output_depth":
            normalize.append(False)
            val = val[:, 0, ...]
            depth = np.clip(val, -0.5, 0.5)
            depth += 0.5
            depth *= 255
            titles.append(key)
            depth = depth.astype(np.uint8)
            depth = np.reshape(depth, (-1, depth.shape[-1]))
            images.append(depth)
        elif key == "surface_normals" or key == "output_surface_normals":
            titles.append(key)
            normalize.append(False)
            val = val.copy()
            if key == "output_surface_normals":
                # Still need to be normalized
                val /= np.sqrt(np.sum(val ** 2, axis=1, keepdims=True))
            surfnorm = (np.clip((val + 1), 0, 2) * 127).astype(np.uint8).transpose((0, 2, 3, 1))
            images.append(surfnorm)
        elif key == "semantic":
            titles.append(key)
            normalize.append(False)
            seg = (val * 314.159 % 255).astype(np.uint8)
            seg = np.reshape(seg, (-1, seg.shape[-1]))
            images.append(seg)
        elif key == "output_reconstruction":
            titles.append(key)
            normalize.append(False)
            val = np.clip(val, -0.5, 0.5)
            val += 0.5
            val *= 255
            val = val.astype(np.uint8).transpose((0, 2, 3, 1))
            images.append(val)
        elif key in {"action_prob", "action_taken", "egomotion_pred", "best_next_action"}:
            if key == "action_prob":
                titles.append(("Output Distribution", "p(Forward)     p(Left)     p(Right)"))
            else:
                titles.append(key)
            if val is not None:
                normalize.append(True)
                prob_hists = np.concatenate([draw_probability_hist(pi) for pi in val.copy()], axis=0)
                images.append(prob_hists)

            else:
                images.append(None)
                normalize.append(False)
    images.append(top_down_map)
    normalize.append(True)
    titles = [string.capwords(title.replace("_", " ")) if isinstance(title, str) else title for title in titles]
    return images, titles, normalize
