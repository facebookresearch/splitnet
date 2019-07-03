#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat import SimulatorActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower, action_to_one_hot, EPSILON
from habitat.utils.geometry_utils import angle_between_quaternions


class OneHotShortestPathFollower(ShortestPathFollower):
    def __init__(self, sim: HabitatSim, goal_radius: float):
        super(OneHotShortestPathFollower, self).__init__(sim, goal_radius, True)
        self._prev_state = None

    def get_next_action(
        self, goal_pos: np.array, previous_action: int
    ) -> Union[int, np.array]:
        """Returns the next action along the shortest path."""
        if np.linalg.norm(goal_pos - self._sim.get_agent_state().position) <= self._goal_radius:
            return action_to_one_hot(SimulatorActions.STOP)

        max_grad_dir = self._est_max_grad_dir(goal_pos)
        if max_grad_dir is None:
            return action_to_one_hot(SimulatorActions.MOVE_FORWARD)
        return self._step_along_grad(max_grad_dir, goal_pos, previous_action)

    def _step_along_grad(
        self, grad_dir: np.quaternion, goal_pos: np.array, previous_action: int
    ) -> np.array:
        current_state = self._sim.get_agent_state()
        alpha = angle_between_quaternions(grad_dir, current_state.rotation)
        if alpha <= np.deg2rad(self._sim.config.TURN_ANGLE) + EPSILON:
            return action_to_one_hot(SimulatorActions.MOVE_FORWARD)
        else:
            if previous_action == SimulatorActions.TURN_LEFT or previous_action == SimulatorActions.TURN_RIGHT:
                # Previous action was a turn, make current suggestion match previous action.
                best_turn = previous_action
            else:
                sim_action = SimulatorActions.TURN_LEFT
                self._sim.step(sim_action)
                best_turn = (
                    SimulatorActions.TURN_LEFT
                    if (angle_between_quaternions(grad_dir, self._sim.get_agent_state().rotation) < alpha)
                    else SimulatorActions.TURN_RIGHT
                )
                self._reset_agent_state(current_state)

            # Check if forward reduces geodesic distance
            curr_dist = self._geo_dist(goal_pos)
            self._sim.step(SimulatorActions.MOVE_FORWARD)
            new_dist = self._geo_dist(goal_pos)
            new_pos = self._sim.get_agent_state().position
            movement_size = np.linalg.norm(new_pos - current_state.position)
            self._reset_agent_state(current_state)
            if new_dist < curr_dist and movement_size / self._step_size > 0.95:
                # Make probability proportional to benefit of doing forward action
                forward_ind = SimulatorActions.MOVE_FORWARD
                one_hot = np.zeros(len(SimulatorActions), dtype=np.float32)
                one_hot[forward_ind] = (curr_dist - new_dist) / self._step_size
                if one_hot[forward_ind] > 0.8:
                    one_hot[forward_ind] = 1
                elif one_hot[forward_ind] < 0.2:
                    one_hot[forward_ind] = 0
                one_hot[best_turn] = 1 - one_hot[forward_ind]
                return one_hot
            else:
                return action_to_one_hot(best_turn)
