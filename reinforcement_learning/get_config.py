#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

from habitat.config.default import get_config


def get_dataset_config(
    dataset_path, data_subset, max_episode_length, render_gpu_id, extra_task_sensors, extra_agent_sensors
):
    config = get_config("configs/habitat_nav_task_config.yaml")
    config.defrost()

    config.DATASET.DATA_PATH = dataset_path
    config.DATASET.SPLIT = data_subset

    config.ENVIRONMENT.MAX_EPISODE_STEPS = max_episode_length

    if len(extra_task_sensors) > 0:
        config.TASK.SENSORS.extend(extra_task_sensors)

    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = render_gpu_id

    if len(extra_agent_sensors) > 0:
        config.SIMULATOR.AGENT_0.SENSORS.extend(extra_agent_sensors)

    for sensor in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR[sensor].HEIGHT = config.SIMULATOR.RGB_SENSOR.HEIGHT
        config.SIMULATOR[sensor].WIDTH = config.SIMULATOR.RGB_SENSOR.WIDTH
        config.SIMULATOR[sensor].POSITION = config.SIMULATOR.RGB_SENSOR.POSITION

    config.TASK.SLACK_REWARD = -0.01
    config.TASK.GOAL_TYPE = "dense"
    return config
