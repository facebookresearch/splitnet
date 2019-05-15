#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import glob
import gzip
import os

import habitat
import tqdm
from habitat.config.default import get_config
from habitat.core.dataset import Dataset
from habitat.datasets.pointnav import generator
from habitat.tasks.nav.nav_task import NavigationEpisode, NavigationGoal

DEBUG = False

DATA_SPLIT = "train"
DATASET = "suncg"
CFG = "configs/habitat_nav_task_config.yaml"

if DATA_SPLIT == "train":
    SPLIT = "train" if not DEBUG else "train_small"
    NUM_EPS_PER_LOCATION = 10 if DEBUG else 100
elif DATA_SPLIT == "val":
    NUM_EPS_PER_LOCATION = 10
    SPLIT = "val"
elif DATA_SPLIT == "test":
    NUM_EPS_PER_LOCATION = 10
    SPLIT = "test"
else:
    raise NotImplementedError("Not implemented for this split type")

OUTPUT_PATH = os.path.join("data", "datasets", "pointnav", DATASET, "v1", SPLIT)
if not os.path.exists(os.path.join(OUTPUT_PATH)):
    os.makedirs(os.path.join(OUTPUT_PATH))


config = get_config(CFG)
config.TASK.success_distance = 0.2
config.TASK.success_reward = 10.0
config.TASK.slack_reward = -0.01
config.TASK.goal_type = "dense"


# get list of all scenes
if DATASET == "mp3d":
    scenes = sorted(glob.glob("data/scene_datasets/mp3d/*"))
elif DATASET == "suncg":
    scenes = sorted(glob.glob("data/scene_datasets/suncg/house/*"))
elif DATASET == "gibson":
    scenes = sorted(glob.glob("data/scene_datasets/mp3d/*"))
else:
    raise NotImplementedError("Not implemented for this dataset")

num_scenes = len(scenes)
if DATA_SPLIT == "train":
    scenes = scenes[: int(num_scenes * 0.6)]
elif DATA_SPLIT == "train_small":
    scenes = scenes[: int(num_scenes * 0.1)]
elif DATA_SPLIT == "val":
    scenes = scenes[int(num_scenes * 0.6) : int(num_scenes * 0.8)]
elif DATA_SPLIT == "test":
    scenes = scenes[int(num_scenes * 0.8) :]

scenes = scenes[: min(1000, len(scenes))]

print("Num scenes", len(scenes))
print("Total num scenes", len(scenes) * NUM_EPS_PER_LOCATION)

episodes = []

for ii, house in enumerate(scenes):
    if DATASET == "mp3d":
        scene_id = house + os.sep + house + ".glb"
    elif DATASET == "suncg":
        scene_id = (house + "/house.json",)
    elif DATASET == "gibson":
        scene_id = house + ".glb"
    else:
        raise NotImplementedError("Not implemented for this dataset")

    dummy_episode = NavigationEpisode(
        episode_id=str(ii),
        goals=[NavigationGoal([0, 0, 0])],
        scene_id=scene_id,
        start_position=[0, 0, 0],
        start_rotation=[0, 0, 0, 1],
    )
    episodes.append(dummy_episode)
dataset = Dataset()
dataset.episodes = episodes
json = dataset.to_json().encode("utf-8")
with gzip.GzipFile(os.path.join(OUTPUT_PATH, "dataset_one_ep_per_scene.json.gz"), "w") as fout:
    fout.write(json)


env = habitat.Env(config=config, dataset=dataset)

episodes = []
episode_id = 0
for ii in tqdm.tqdm(range(len(scenes))):
    try:
        env.reset()
        for jj in range(NUM_EPS_PER_LOCATION):
            episode = generator.generate_pointnav_episode(env, episode_id)
            episode_id += 1
            episodes.append(episode)
    except:
        continue
new_dataset = Dataset()
new_dataset.episodes = episodes
json = new_dataset.to_json().encode("utf-8")
with gzip.GzipFile(os.path.join(OUTPUT_PATH, "dataset.json.gz"), "w") as fout:
    fout.write(json)
