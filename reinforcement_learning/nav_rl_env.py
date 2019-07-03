#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import random
from abc import ABC
from typing import Any

import habitat
import numpy as np
from dg_util.python_utils import pytorch_util as pt_util
from habitat.core.simulator import Observations
from habitat import SimulatorActions

from utils import one_hot_shortest_path_follower

#SIM_ACTION_IND_TO_SIM_ACTION = {SimulatorActions[action]: action for action in SimulatorActions}


class MultiDatasetEnv(habitat.RLEnv, ABC):
    def __init__(self, config, datasets):
        self._datasets = datasets
        if "train" in datasets:
            dataset_key = "train"
        else:
            dataset_key = random.sample(datasets.keys(), 1)[0]
        initial_dataset = self._datasets[dataset_key]

        self._config = config
        super(MultiDatasetEnv, self).__init__(config, initial_dataset)
        self._dataset_type = dataset_key

    def switch_dataset(self, data_type):
        self._dataset_type = data_type
        self._env.dataset = self._dataset
        self._env._episodes = self._dataset.episodes if self._dataset else []
        if data_type == "train":
            print("jumping to random spot")
            self.habitat_env._current_episode_index = random.randint(0, len(self.episodes) - 1)
            print("ind", self.habitat_env._current_episode_index)

    @property
    def _dataset(self):
        return self._datasets.get(self._dataset_type)

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


class FormattedRLEnv(MultiDatasetEnv, ABC):
    def __init__(self, config, datasets):
        self._previous_action = SimulatorActions.STOP
        self._slack_reward = config.TASK.SLACK_REWARD
        self._collision_reward = config.TASK.COLLISION_REWARD
        self._step_size = config.SIMULATOR.FORWARD_STEP_SIZE
        self._previous_pose = None
        self._current_pose = None
        self._total_episode_count = 0
        self._num_episodes_before_jump = config.TASK.NUM_EPISODES_BEFORE_JUMP
        self._config = config
        super(FormattedRLEnv, self).__init__(config, datasets)
        self.num_episodes_in_dataset = len(self.habitat_env.episodes)

    def switch_dataset(self, data_type):
        super(FormattedRLEnv, self).switch_dataset(data_type)
        if data_type == "train":
            self._num_episodes_before_jump = self._config.TASK.NUM_EPISODES_BEFORE_JUMP
        else:
            self._num_episodes_before_jump = -1

    def reset(self):
        if self._num_episodes_before_jump > 0 and (self._total_episode_count % self._num_episodes_before_jump) == 0:
            print("jumping to random spot")
            self.habitat_env._current_episode_index = random.randint(0, len(self.episodes) - 1)
            print("ind", self.habitat_env._current_episode_index)
        self._total_episode_count += 1

        self._previous_action = SimulatorActions.STOP
        observations = super(FormattedRLEnv, self).reset()
        self._previous_pose = (
            self.habitat_env.sim.get_agent_state().position,
            self.habitat_env.sim.get_agent_state().rotation,
        )
        self._current_pose = (
            self.habitat_env.sim.get_agent_state().position,
            self.habitat_env.sim.get_agent_state().rotation,
        )

        formatted_obs = self.format_observations(observations)
        return formatted_obs

    def step(self, action):
        self._previous_pose = self._current_pose
        self._previous_action = action #SIM_ACTION_IND_TO_SIM_ACTION[action]
        obs = self._env.step(action)
        done = self.get_done(obs)
        obs = self.format_observations(obs, done)
        reward = self.get_reward(obs)
        info = self.get_info(obs)
        return obs, reward, done, info

    def get_reward(self, observations: Observations):
        reward = self._slack_reward

        self._current_pose = (
            self.habitat_env.sim.get_agent_state().position,
            self.habitat_env.sim.get_agent_state().rotation,
        )

        if self._previous_action == SimulatorActions.MOVE_FORWARD:
            if np.linalg.norm(self._current_pose[0] - self._previous_pose[0]) < self._step_size * 0.25:
                # Collided with something
                reward += self._collision_reward
        return reward

    def format_observations(self, obs, done=False):
        if "rgb" in obs:
            obs["rgb"] = obs["rgb"].transpose(2, 0, 1)[:3, :, :]
        if "depth" in obs:
            depth = obs["depth"]
            depth[depth == 0] = 1
            depth -= 0.5
            depth = depth[np.newaxis, :, :, 0]  # Move channel axis to start
            obs["depth"] = depth
        # Action you took to get this image.
        obs["prev_action_one_hot"] = pt_util.get_one_hot_numpy(self._previous_action, len(SimulatorActions))
        return obs

    def get_info(self, observations):
        info = super(FormattedRLEnv, self).get_info(observations)
        info["episode_id"] = self.habitat_env.episodes[self.habitat_env._current_episode_index].episode_id
        return info


class PointnavRLEnv(FormattedRLEnv):
    def __init__(self, config, datasets):
        self._success_distance = config.TASK.SUCCESS_DISTANCE
        self._success_reward = config.TASK.SUCCESS_REWARD
        self._return_best_next_action = config.TASK.OBSERVE_BEST_NEXT_ACTION
        self._enable_stop_action = config.TASK.ENABLE_STOP_ACTION
        self._previous_target_distance = -1
        self._delta_target_distance = 0
        self._follower = None
        super(PointnavRLEnv, self).__init__(config, datasets)
        if self._return_best_next_action:
            sim: habitat.sims.habitat_simulator.HabitatSim = self.habitat_env.sim
            self._follower = one_hot_shortest_path_follower.OneHotShortestPathFollower(sim, self._success_distance)

    def reset(self):
        formatted_obs = super(PointnavRLEnv, self).reset()
        self._previous_target_distance = self._distance_target()
        self._delta_target_distance = 0

        return formatted_obs

    def format_observations(self, obs, done=False):
        obs = super(PointnavRLEnv, self).format_observations(obs, done)
        if self._return_best_next_action and not done:
            obs["best_next_action"] = self._follower.get_next_action(
                # np.array(self.habitat_env.current_episode.goals[0].position), SIM_ACTION_TO_NAME[self._previous_action]
                np.array(self.habitat_env.current_episode.goals[0].position),
                self._previous_action,
            )
        obs["goal_geodesic_distance"] = self._distance_target()
        return obs

    def step(self, action):
        if not self._enable_stop_action:
            if self._distance_target() < self._success_distance:
                action = SimulatorActions.STOP

        obs, reward, done, info = super(PointnavRLEnv, self).step(action)
        return obs, reward, done, info

    def get_reward(self, observations):
        reward = super(PointnavRLEnv, self).get_reward(observations)

        current_target_distance = self._distance_target()
        self._delta_target_distance = self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        reward += self._delta_target_distance

        if self._episode_success():
            reward += self._success_reward

        return reward

    def get_reward_range(self):
        return (
            self._slack_reward + self._collision_reward - self._step_size,
            self._slack_reward + self._step_size + self._success_reward,
        )

    def _distance_target(self):
        current_position = self.habitat_env.sim.get_agent_state().position
        target_position = self.habitat_env.current_episode.goals[0].position
        distance = self.habitat_env.sim.geodesic_distance(current_position, target_position)
        return distance

    def _episode_success(self):
        return self._stop_called() and self._distance_target() < self._success_distance

    def _stop_called(self):
        return self._previous_action == SimulatorActions.STOP

    def get_done(self, observations):
        done = self.habitat_env.episode_over or self._stop_called()
        return done

    def get_info(self, observations):
        info = super(PointnavRLEnv, self).get_info(observations)

        if self.get_done(observations):
            info["final_distance"] = self._distance_target()
        else:
            info.pop("spl")

        return info


class ExplorationRLEnv(FormattedRLEnv):
    def __init__(self, config, datasets):
        self._grid_size = config.TASK.GRID_SIZE
        print("config", config)
        self._new_grid_cell_reward = config.TASK.NEW_GRID_CELL_REWARD
        self._return_visited_grid = config.TASK.RETURN_VISITED_GRID
        self._camera_height = config.SIMULATOR.RGB_SENSOR.POSITION[1]
        self._visited = set()
        self._regions = None
        super(ExplorationRLEnv, self).__init__(config, datasets)

    def _check_grid_cell(self):
        location = self._current_pose[0].copy()
        if self._grid_size > 0:
            quantized_location = (location / self._grid_size).astype(np.int32)
            quantized_loc_tuple = tuple(quantized_location.tolist())
            if quantized_loc_tuple not in self._visited:
                self._visited.add(quantized_loc_tuple)
                return True
        else:
            # Use room regions instead
            location[1] += self._camera_height
            in_region = np.logical_and(
                np.all(location > self._regions[0, ...], axis=1), np.all(location < self._regions[1, ...], axis=1)
            )
            new_regions = np.where(in_region)[0]
            if len(new_regions) > 0:
                self._visited.update(set(new_regions))
                return True

        return False

    def format_observations(self, obs, done=False):
        obs = super(ExplorationRLEnv, self).format_observations(obs, done)
        if self._return_visited_grid:
            obs["visited"] = len(self._visited)
            obs["visited_grid"] = self._to_grid()
        return obs

    def reset(self):
        self._visited = set()
        obs = super(ExplorationRLEnv, self).reset()
        if self._grid_size <= 0:
            regions = self.habitat_env.sim.semantic_annotations().regions
            region_size_padding = self._step_size
            region_sizes = np.array([region.aabb.sizes for region in regions]) / 2
            region_sizes = np.maximum(region_sizes, region_size_padding)
            region_centers = np.array([region.aabb.center for region in regions])
            self._regions = np.array([region_centers - region_sizes, region_centers + region_sizes])
        self._check_grid_cell()
        return obs

    def step(self, action):
        obs, reward, done, info = super(ExplorationRLEnv, self).step(action)
        if done:
            info["visited_states"] = len(self._visited)
        return obs, reward, done, info

    def get_reward_range(self):
        return self._slack_reward + self._collision_reward, self._slack_reward + self._new_grid_cell_reward

    def get_reward(self, observations: Observations) -> Any:
        reward = super(ExplorationRLEnv, self).get_reward(observations)

        if "world_coords" in observations:
            reward += self._check_grid_cell() * 0.01
        else:
            in_new_grid_cell = self._check_grid_cell()
            if in_new_grid_cell:
                reward += self._new_grid_cell_reward
        return reward

    def get_done(self, observations: Observations) -> bool:
        return self.habitat_env.episode_over

    def _to_grid(self, padding=5):
        if len(self._visited) == 0:
            return np.zeros((3, 3), dtype=np.uint8)
        visited_cells = np.array(list(self._visited))
        visited_cells = visited_cells[:, [0, 2]]  # Ignore y axis
        mins = np.min(visited_cells, axis=0)
        maxes = np.max(visited_cells, axis=0)
        grid = np.zeros((maxes - mins + 2 * padding + 1), dtype=np.uint8)
        # Mark the visited ones
        visited_cells = visited_cells + padding - mins
        np.ravel(grid)[np.ravel_multi_index(visited_cells.T, grid.shape, mode="clip")] = 1
        grid = np.rot90(grid, 2)
        return grid


class RunAwayRLEnv(FormattedRLEnv):
    def __init__(self, config, datasets):
        print("config", config)
        self._start_position = None
        self._prev_distance = None
        super(RunAwayRLEnv, self).__init__(config, datasets)

    def reset(self):
        self._start_position = None
        obs = super(RunAwayRLEnv, self).reset()
        self._start_position = self.habitat_env.sim.get_agent_state().position
        self._prev_distance = 0
        return obs

    def _get_distance(self):
        current_position = self.habitat_env.sim.get_agent_state().position
        distance = self.habitat_env.sim.geodesic_distance(current_position, self._start_position)
        return distance

    def format_observations(self, obs, done=False):
        obs = super(RunAwayRLEnv, self).format_observations(obs, done)
        if self._start_position is None:
            obs["distance_from_start"] = 0
        else:
            obs["distance_from_start"] = self._get_distance()
        return obs

    def step(self, action):
        obs, reward, done, info = super(RunAwayRLEnv, self).step(action)
        self._prev_distance = self._get_distance()
        if not np.isfinite(reward):
            # Happens in very rare cases.
            done = True
            reward = 0
        if done:
            info["distance_from_start"] = self._prev_distance
        return obs, reward, done, info

    def get_reward_range(self):
        return self._slack_reward + self._collision_reward - self._step_size, self._slack_reward + self._step_size

    def get_reward(self, observations: Observations) -> Any:
        reward = super(RunAwayRLEnv, self).get_reward(observations)
        reward += self._get_distance() - self._prev_distance
        return reward

    def get_done(self, observations: Observations) -> bool:
        return self.habitat_env.episode_over


def make_env_fn(env_type, config, dataset, seed):
    config.freeze()
    env = env_type(config, dataset)
    env.seed(seed)
    return env
