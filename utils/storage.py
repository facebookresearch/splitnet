#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import torch
from a2c_ppo_acktr.storage import RolloutStorage
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils import pytorch_util as pt_util


class RolloutStorageWithMultipleObservations(RolloutStorage):
    def __init__(
        self,
        num_forward_rollout_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
        additional_observations_info_dict,
        primary_obs_name,
    ):
        super(RolloutStorageWithMultipleObservations, self).__init__(
            num_forward_rollout_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size
        )
        self.num_forward_rollout_steps = num_forward_rollout_steps
        self.additional_observations_dict = {}
        self.primary_obs_name = primary_obs_name
        for name, (shape, dtype) in additional_observations_info_dict.items():
            if name == self.primary_obs_name:
                continue
            dtype = pt_util.numpy_dtype_to_pytorch_dtype(dtype)
            self.additional_observations_dict[name] = torch.zeros(
                num_forward_rollout_steps + 1, num_processes, *shape, dtype=dtype
            )

        self._warn_once = {}

    def to(self, device):
        super(RolloutStorageWithMultipleObservations, self).to(device)
        for key in self.additional_observations_dict:
            self.additional_observations_dict[key] = self.additional_observations_dict[key].to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks, bad_masks):
        # Must copy additional obs before super call because super call increments step
        for key in self.additional_observations_dict:
            if key not in obs:
                if "insert" not in self._warn_once:
                    self._warn_once["insert"] = set()
                if key not in self._warn_once["insert"]:
                    print("Warning! Key", key, "not in observation in function insert")
                self._warn_once["insert"].add(key)
            else:
                self.additional_observations_dict[key][self.step + 1].copy_(obs[key])

        super(RolloutStorageWithMultipleObservations, self).insert(
            obs[self.primary_obs_name],
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks,
        )

    def copy_obs(self, obs, index):
        self.obs[index].copy_(obs[self.primary_obs_name])
        for key in self.additional_observations_dict:
            if key not in obs:
                if "copy_obs" not in self._warn_once:
                    self._warn_once["copy_obs"] = set()
                if key not in self._warn_once["copy_obs"]:
                    print("Warning! Key", key, "not in observation in function copy_obs")
                self._warn_once["copy_obs"].add(key)
            else:
                self.additional_observations_dict[key][index].copy_(obs[key])

    def copy_seq(self, seq_dict):
        obs = seq_dict[self.primary_obs_name]
        self.obs.copy_(obs)

        used_keys = set()
        for key, val in seq_dict.items():
            if hasattr(self, key):
                getattr(self, key).copy_(val)
                used_keys.add(key)

        for key in used_keys:
            del seq_dict[key]

        for key in self.additional_observations_dict:
            if key not in seq_dict:
                if "copy_seq" not in self._warn_once:
                    self._warn_once["copy_seq"] = set()
                if key not in self._warn_once["copy_seq"]:
                    print("Warning! Key", key, "not in observation in function copy_seq")
                self._warn_once["copy_seq"].add(key)
            else:
                self.additional_observations_dict[key].copy_(seq_dict[key])

    def after_update(self):
        for key in self.additional_observations_dict:
            self.additional_observations_dict[key][0].copy_(self.additional_observations_dict[key][-1])
        super(RolloutStorageWithMultipleObservations, self).after_update()

    def remove_worker_storage(self, index):
        with torch.no_grad():
            for key in self.additional_observations_dict:
                self.additional_observations_dict[key] = torch.cat(
                    (
                        self.additional_observations_dict[key][:, :index],
                        self.additional_observations_dict[key][:, index + 1 :],
                    ),
                    dim=1,
                )
            self.obs = torch.cat((self.obs[:, :index], self.obs[:, index + 1 :]), dim=1)
            self.recurrent_hidden_states = torch.cat(
                (self.recurrent_hidden_states[:, :index], self.recurrent_hidden_states[:, index + 1 :]), dim=1
            )
            self.rewards = torch.cat((self.rewards[:, :index], self.rewards[:, index + 1 :]), dim=1)
            self.value_preds = torch.cat((self.value_preds[:, :index], self.value_preds[:, index + 1 :]), dim=1)
            self.returns = torch.cat((self.returns[:, :index], self.returns[:, index + 1 :]), dim=1)
            self.action_log_probs = torch.cat(
                (self.action_log_probs[:, :index], self.action_log_probs[:, index + 1 :]), dim=1
            )
            self.actions = torch.cat((self.actions[:, :index], self.actions[:, index + 1 :]), dim=1)
            self.masks = torch.cat((self.masks[:, :index], self.masks[:, index + 1 :]), dim=1)

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_forward_rollout_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_forward_rollout_steps
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_forward_rollout_steps, num_processes * num_forward_rollout_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            additional_obs_batch = {
                key: val[:-1].view(-1, val.size()[2:])[indices]
                for key, val in self.additional_observations_dict.items()
            }
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, additional_obs_batch

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            additional_obs_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                additional_obs_batch.append(
                    {key: val[:-1, ind] for key, val in self.additional_observations_dict.items()}
                )
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_forward_rollout_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            additional_obs_batch = {
                key: torch.stack([additional_obs_batch[ii][key] for ii in range(num_envs_per_batch)], 1)
                for key in additional_obs_batch[0]
            }

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = pt_util.remove_dim(obs_batch, 1)
            additional_obs_batch = {key: pt_util.remove_dim(val, 1) for key, val in additional_obs_batch.items()}
            actions_batch = pt_util.remove_dim(actions_batch, 1)
            value_preds_batch = pt_util.remove_dim(value_preds_batch, 1)
            return_batch = pt_util.remove_dim(return_batch, 1)
            masks_batch = pt_util.remove_dim(masks_batch, 1)
            old_action_log_probs_batch = pt_util.remove_dim(old_action_log_probs_batch, 1)
            adv_targ = pt_util.remove_dim(adv_targ, 1)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, additional_obs_batch
