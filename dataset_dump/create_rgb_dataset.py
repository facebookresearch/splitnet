#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import os
import random
from typing import List, Dict, Optional

import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
from habitat.config.default import get_config
from habitat.datasets import make_dataset

CFG = "configs/habitat_nav_task_config.yaml"


def make_config(gpu_id, split, data_path, sensors, resolution):
    config = get_config(CFG)
    config.defrost()
    config.TASK.NAME = "Nav-v0"
    config.TASK.MEASUREMENTS = []
    config.DATASET.SPLIT = split
    config.DATASET.POINTNAVV1.DATA_PATH = data_path
    config.HEIGHT = resolution
    config.WIDTH = resolution
    for sensor in sensors:
        config.SIMULATOR[sensor]["HEIGHT"] = resolution
        config.SIMULATOR[sensor]["WIDTH"] = resolution
        config.SIMULATOR[sensor]["POSITION"] = [0, 1.09, 0]
        config.SIMULATOR[sensor]["HFOV"] = 45

    config.TASK.HEIGHT = resolution
    config.TASK.WIDTH = resolution
    config.SIMULATOR.AGENT_0.SENSORS = sensors
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    return config


class RandomImageGenerator(object):
    def __init__(
        self,
        gpu_id: int,
        unique_dataset_name: str,
        split: str,
        data_path: str,
        images_before_reset: int,
        sensors: Optional[List[str]] = None,
        resolution: Optional[int] = 256,
    ) -> None:
        if sensors is None:
            sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
        self.images_before_reset = images_before_reset
        config = make_config(gpu_id, split, data_path, sensors, resolution)
        data_dir = os.path.join("data/scene_episodes/", unique_dataset_name + "_" + split)
        self.dataset_name = config.DATASET.TYPE
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")
        # Creates a dataset where each episode is a random spawn point in each scene.
        if not os.path.exists(data_path):
            dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
            # Get one episode per scene in dataset
            scene_episodes = {}
            for episode in dataset.episodes:
                if episode.scene_id not in scene_episodes:
                    scene_episodes[episode.scene_id] = episode
            scene_episodes = list(scene_episodes.values())
            print("scene episodes", len(scene_episodes))
            dataset.episodes = scene_episodes
            if not os.path.exists(data_path):
                # Multiproc do check again before write.
                json = dataset.to_json().encode("utf-8")
                with gzip.GzipFile(data_path, "w") as fout:
                    fout.write(json)
        dataset = mp3d_dataset.PointNavDatasetV1()
        with gzip.open(data_path, "rt") as f:
            dataset.from_json(f.read())

        config.TASK.SENSORS = ["POINTGOAL_SENSOR"]
        if "SEMANTIC_SESNOR" in sensors:
            config.TASK.SENSOR.append("CLASS_SEGMENTATION_SENSOR")
            config.TASK.CLASS_SEGMENTATION_SENSOR.HEIGHT = config.TASK.HEIGHT
            config.TASK.CLASS_SEGMENTATION_SENSOR.WIDTH = config.TASK.WIDTH

        config.freeze()
        self.env = habitat.Env(config=config, dataset=dataset)
        random.shuffle(self.env.episodes)
        self.num_samples = 0

    def get_sample(self) -> Dict[str, np.ndarray]:
        if self.num_samples % self.images_before_reset == 0:
            self.env.reset()

        rand_location = self.env.sim.sample_navigable_point()
        num_tries = 0
        while rand_location[1] > 1:
            rand_location = self.env.sim.sample_navigable_point()
            num_tries += 1
            if num_tries > 1000:
                self.env.reset()

        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
        self.env.sim.set_agent_state(rand_location, rand_rotation)
        obs = self.env.sim._sensor_suite.get_observations(self.env.sim._sim.get_sensor_observations())

        class_semantic = None
        if "semantic" in obs:
            # Currently unused
            semantic = obs["semantic"]
            class_semantic = self.env.sim.scene_obj_id_to_semantic_class[semantic].astype(np.int32)

        img = obs["rgb"][:, :, :3]
        depth = obs["depth"].squeeze()

        self.num_samples += 1
        return {"rgb": img, "depth": depth, "class_semantic": class_semantic}
