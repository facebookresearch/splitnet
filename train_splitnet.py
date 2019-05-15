#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import random
import time
from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from a2c_ppo_acktr.utils import update_linear_schedule
from habitat.datasets import make_dataset
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.utils.visualizations.utils import images_to_video

from base_habitat_rl_runner import ACTION_SPACE, SIM_ACTION_TO_NAME, ACTION_SPACE_TO_SIM_ACTION
from eval_splitnet import HabitatRLEvalRunner, REWARD_SCALAR
from utils import drawing
from utils import pytorch_util as pt_util
from utils import tensorboard_logger
from utils.storage import RolloutStorageWithMultipleObservations


class HabitatRLTrainAndEvalRunner(HabitatRLEvalRunner):
    def __init__(self, create_decoder=True):
        self.rollouts = None
        self.logger = None
        self.train_stats = None
        super(HabitatRLTrainAndEvalRunner, self).__init__(create_decoder)

    def setup(self, create_decoder=True):
        assert self.shell_args.update_policy_decoder_features or self.shell_args.update_encoder_features
        super(HabitatRLTrainAndEvalRunner, self).setup(create_decoder)
        self.shell_args.cuda = not self.shell_args.no_cuda and torch.cuda.is_available()

        print("Starting make dataset")
        start_t = time.time()
        config = self.configs[0]
        dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
        observation_shape_chw = (3, config.SIMULATOR.RGB_SENSOR.HEIGHT, config.SIMULATOR.RGB_SENSOR.WIDTH)
        print("made dataset")

        assert len(dataset.episodes) > 0, "empty datasets"
        if self.shell_args.num_train_scenes > 0:
            scene_ids = sorted(dataset.scene_ids)
            random.seed(0)
            random.shuffle(scene_ids)
            used_scene_ids = set(scene_ids[: self.shell_args.num_train_scenes])
            dataset.filter_episodes(lambda x: x.scene_id in used_scene_ids)

        if self.shell_args.record_video:
            random.shuffle(dataset.episodes)

        datasets = dataset.get_splits(
            self.shell_args.num_processes, remove_unused_episodes=True, collate_scene_ids=True
        )

        print("Dataset creation time %.3f" % (time.time() - start_t))

        self.rollouts = RolloutStorageWithMultipleObservations(
            self.shell_args.num_forward_rollout_steps,
            self.shell_args.num_processes,
            observation_shape_chw,
            self.gym_action_space,
            self.agent.recurrent_hidden_state_size,
            self.observation_space,
            "rgb",
        )
        self.rollouts.to(self.device)

        print("Feeding dummy batch")
        dummy_start = time.time()
        self.optimizer.update(self.rollouts, self.shell_args)
        print("Done feeding dummy batch %.3f" % (time.time() - dummy_start))

        self.logger = None
        if self.shell_args.tensorboard:
            self.logger = tensorboard_logger.Logger(
                os.path.join(self.shell_args.log_prefix, self.shell_args.tensorboard_dirname, self.time_str + "_train")
            )

        self.datasets = {"train": datasets, "val": self.eval_datasets}

        self.train_stats = dict(
            num_episodes=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            num_steps=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            reward=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            spl=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            visited_states=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            success=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            end_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            start_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            delta_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            distance_from_start=np.zeros(self.shell_args.num_processes, dtype=np.float32),
        )

    def train_model(self):
        episode_rewards = deque(maxlen=10)
        current_episode_rewards = np.zeros(self.shell_args.num_processes)
        episode_lengths = deque(maxlen=10)
        current_episode_lengths = np.zeros(self.shell_args.num_processes)
        current_rewards = np.zeros(self.shell_args.num_processes)

        total_num_steps = self.start_iter
        fps_timer = [time.time(), total_num_steps]
        timers = np.zeros(3)
        egomotion_loss = 0

        video_frames = []
        num_episodes = 0
        # self.evaluate_model()

        obs = self.envs.reset()
        if self.compute_surface_normals:
            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        if self.shell_args.algo == "supervised":
            obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE])
        self.rollouts.copy_obs(obs, 0)
        distances = pt_util.to_numpy_array(obs["goal_geodesic_distance"])
        self.train_stats["start_geodesic_distance"][:] = distances
        previous_visual_features = None
        egomotion_pred = None
        prev_action = None
        prev_action_probs = None
        num_updates = (
            int(self.shell_args.num_env_steps) // self.shell_args.num_forward_rollout_steps
        ) // self.shell_args.num_processes

        try:
            for iter_count in range(num_updates):
                if self.shell_args.tensorboard:
                    if iter_count % 500 == 0:
                        print("Logging conv summaries")
                        self.logger.network_conv_summary(self.agent, total_num_steps)
                    elif iter_count % 100 == 0:
                        print("Logging variable summaries")
                        self.logger.network_variable_summary(self.agent, total_num_steps)

                if self.shell_args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    update_linear_schedule(self.optimizer.optimizer, iter_count, num_updates, self.shell_args.lr)

                if self.shell_args.algo == "ppo" and self.shell_args.use_linear_clip_decay:
                    self.optimizer.clip_param = self.shell_args.clip_param * (1 - iter_count / float(num_updates))

                if hasattr(self.agent.base, "enable_decoder"):
                    if self.shell_args.record_video:
                        self.agent.base.enable_decoder()
                    else:
                        self.agent.base.disable_decoder()

                for step in range(self.shell_args.num_forward_rollout_steps):
                    with torch.no_grad():
                        start_t = time.time()
                        value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                            {
                                "images": self.rollouts.obs[step],
                                "target_vector": self.rollouts.additional_observations_dict["pointgoal"][step],
                                "prev_action_one_hot": self.rollouts.additional_observations_dict[
                                    "prev_action_one_hot"
                                ][step],
                            },
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step],
                        )
                        action_cpu = pt_util.to_numpy_array(action.squeeze(1))
                        translated_action_space = ACTION_SPACE[action_cpu]
                        if not self.shell_args.end_to_end:
                            self.rollouts.additional_observations_dict["visual_encoder_features"][
                                self.rollouts.step
                            ].copy_(self.agent.base.visual_encoder_features)

                        if self.shell_args.use_motion_loss:
                            if self.shell_args.record_video:
                                if previous_visual_features is not None:
                                    egomotion_pred = self.agent.base.predict_egomotion(
                                        self.agent.base.visual_features, previous_visual_features
                                    )
                            previous_visual_features = self.agent.base.visual_features.detach()

                        timers[1] += time.time() - start_t

                        if self.shell_args.record_video:
                            # Copy so we don't mess with obs itself
                            draw_obs = OrderedDict()
                            for key, val in obs.items():
                                draw_obs[key] = pt_util.to_numpy_array(val).copy()
                            best_next_action = draw_obs.pop("best_next_action", None)

                            if prev_action is not None:
                                draw_obs["action_taken"] = pt_util.to_numpy_array(self.agent.last_dist.probs).copy()
                                draw_obs["action_taken"][:] = 0
                                draw_obs["action_taken"][np.arange(self.shell_args.num_processes), prev_action] = 1
                                draw_obs["action_taken_name"] = SIM_ACTION_TO_NAME[
                                    ACTION_SPACE_TO_SIM_ACTION[ACTION_SPACE[prev_action.squeeze()]]
                                ]
                                draw_obs["action_prob"] = pt_util.to_numpy_array(prev_action_probs).copy()
                            else:
                                draw_obs["action_taken"] = None
                                draw_obs["action_taken_name"] = SIM_ACTION_TO_NAME[SimulatorActions.STOP]
                                draw_obs["action_prob"] = None
                            prev_action = action_cpu
                            prev_action_probs = self.agent.last_dist.probs.detach()
                            if (
                                hasattr(self.agent.base, "decoder_outputs")
                                and self.agent.base.decoder_outputs is not None
                            ):
                                min_channel = 0
                                for key, num_channels in self.agent.base.decoder_output_info:
                                    outputs = self.agent.base.decoder_outputs[
                                        :, min_channel : min_channel + num_channels, ...
                                    ]
                                    draw_obs["output_" + key] = pt_util.to_numpy_array(outputs).copy()
                                    min_channel += num_channels
                            draw_obs["rewards"] = current_rewards.copy()
                            draw_obs["step"] = current_episode_lengths.copy()
                            draw_obs["method"] = self.shell_args.method_name
                            if best_next_action is not None:
                                draw_obs["best_next_action"] = best_next_action
                            if self.shell_args.use_motion_loss:
                                if egomotion_pred is not None:
                                    draw_obs["egomotion_pred"] = pt_util.to_numpy_array(
                                        F.softmax(egomotion_pred, dim=1)
                                    ).copy()
                                else:
                                    draw_obs["egomotion_pred"] = None
                            images, titles, normalize = drawing.obs_to_images(draw_obs)
                            if self.shell_args.algo == "supervised":
                                im_inds = [0, 2, 3, 1, 9, 6, 7, 8, 5, 4]
                            else:
                                im_inds = [0, 2, 3, 1, 6, 7, 8, 5]
                            height, width = images[0].shape[:2]
                            subplot_image = drawing.subplot(
                                images,
                                2,
                                5,
                                titles=titles,
                                normalize=normalize,
                                order=im_inds,
                                output_width=max(width, 320),
                                output_height=max(height, 320),
                            )
                            video_frames.append(subplot_image)

                        # save dists from previous step or else on reset they will be overwritten
                        distances = pt_util.to_numpy_array(obs["goal_geodesic_distance"])

                        start_t = time.time()
                        obs, rewards, dones, infos = self.envs.step(translated_action_space)
                        timers[0] += time.time() - start_t
                        obs["reward"] = rewards
                        if self.shell_args.algo == "supervised":
                            obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE]).to(
                                torch.float32
                            )
                        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                        rewards *= REWARD_SCALAR
                        rewards = np.clip(rewards, -10, 10)

                        if self.shell_args.record_video and not dones[0]:
                            obs["top_down_map"] = infos[0]["top_down_map"]

                        if self.compute_surface_normals:
                            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))

                        current_rewards = pt_util.to_numpy_array(rewards)
                        current_episode_rewards += pt_util.to_numpy_array(rewards).squeeze()
                        current_episode_lengths += 1
                        for ii, done_e in enumerate(dones):
                            if done_e:
                                num_episodes += 1
                                if self.shell_args.record_video:
                                    final_rgb = draw_obs["rgb"].transpose(0, 2, 3, 1).squeeze(0)
                                    if self.shell_args.task == "pointnav":
                                        if infos[ii]["spl"] > 0:
                                            draw_obs["action_taken_name"] = "Stop. Success"
                                            draw_obs["reward"] = [self.configs[0].TASK.SUCCESS_REWARD]
                                            final_rgb[:] = final_rgb * np.float32(0.5) + np.tile(
                                                np.array([0, 128, 0], dtype=np.uint8),
                                                (final_rgb.shape[0], final_rgb.shape[1], 1),
                                            )
                                        else:
                                            draw_obs["action_taken_name"] = "Timeout. Failed"
                                            final_rgb[:] = final_rgb * np.float32(0.5) + np.tile(
                                                np.array([128, 0, 0], dtype=np.uint8),
                                                (final_rgb.shape[0], final_rgb.shape[1], 1),
                                            )
                                    elif self.shell_args.task == "exploration" or self.shell_args.task == "flee":
                                        draw_obs["action_taken_name"] = "End of episode."
                                    final_rgb = final_rgb[np.newaxis, ...].transpose(0, 3, 1, 2)
                                    draw_obs["rgb"] = final_rgb

                                    images, titles, normalize = drawing.obs_to_images(draw_obs)
                                    im_inds = [0, 2, 3, 1, 6, 7, 8, 5]
                                    height, width = images[0].shape[:2]
                                    subplot_image = drawing.subplot(
                                        images,
                                        2,
                                        5,
                                        titles=titles,
                                        normalize=normalize,
                                        order=im_inds,
                                        output_width=max(width, 320),
                                        output_height=max(height, 320),
                                    )
                                    video_frames.extend(
                                        [subplot_image]
                                        * (self.configs[0].ENVIRONMENT.MAX_EPISODE_STEPS + 30 - len(video_frames))
                                    )

                                    if "top_down_map" in infos[0]:
                                        video_dir = os.path.join(self.shell_args.log_prefix, "videos")
                                        if not os.path.exists(video_dir):
                                            os.makedirs(video_dir)
                                        im_path = os.path.join(
                                            self.shell_args.log_prefix, "videos", "total_steps_%d.png" % total_num_steps
                                        )
                                        from habitat.utils.visualizations import maps
                                        import imageio

                                        top_down_map = maps.colorize_topdown_map(infos[0]["top_down_map"]["map"])
                                        imageio.imsave(im_path, top_down_map)

                                    images_to_video(
                                        video_frames,
                                        os.path.join(self.shell_args.log_prefix, "videos"),
                                        "total_steps_%d" % total_num_steps,
                                    )
                                    video_frames = []

                                if self.shell_args.task == "pointnav":
                                    print(
                                        "FINISHED EPISODE %d Length %d Reward %.3f SPL %.4f"
                                        % (
                                            num_episodes,
                                            current_episode_lengths[ii],
                                            current_episode_rewards[ii],
                                            infos[ii]["spl"],
                                        )
                                    )
                                    self.train_stats["spl"][ii] = infos[ii]["spl"]
                                    self.train_stats["success"][ii] = self.train_stats["spl"][ii] > 0
                                    self.train_stats["end_geodesic_distance"][ii] = (
                                        distances[ii] - self.configs[0].SIMULATOR.FORWARD_STEP_SIZE
                                    )
                                    self.train_stats["delta_geodesic_distance"][ii] = (
                                        self.train_stats["start_geodesic_distance"][ii]
                                        - self.train_stats["end_geodesic_distance"][ii]
                                    )
                                    self.train_stats["num_steps"][ii] = current_episode_lengths[ii]
                                elif self.shell_args.task == "exploration":
                                    print(
                                        "FINISHED EPISODE %d Reward %.3f States Visited %d"
                                        % (num_episodes, current_episode_rewards[ii], infos[ii]["visited_states"])
                                    )
                                    self.train_stats["visited_states"][ii] = infos[ii]["visited_states"]
                                elif self.shell_args.task == "flee":
                                    print(
                                        "FINISHED EPISODE %d Reward %.3f Distance from start %.4f"
                                        % (num_episodes, current_episode_rewards[ii], infos[ii]["distance_from_start"])
                                    )
                                    self.train_stats["distance_from_start"][ii] = infos[ii]["distance_from_start"]

                                self.train_stats["num_episodes"][ii] += 1
                                self.train_stats["reward"][ii] = current_episode_rewards[ii]

                                if self.shell_args.tensorboard:
                                    log_dict = {"single_episode/reward": self.train_stats["reward"][ii]}
                                    if self.shell_args.task == "pointnav":
                                        log_dict.update(
                                            {
                                                "single_episode/num_steps": self.train_stats["num_steps"][ii],
                                                "single_episode/spl": self.train_stats["spl"][ii],
                                                "single_episode/success": self.train_stats["success"][ii],
                                                "single_episode/start_geodesic_distance": self.train_stats[
                                                    "start_geodesic_distance"
                                                ][ii],
                                                "single_episode/end_geodesic_distance": self.train_stats[
                                                    "end_geodesic_distance"
                                                ][ii],
                                                "single_episode/delta_geodesic_distance": self.train_stats[
                                                    "delta_geodesic_distance"
                                                ][ii],
                                            }
                                        )
                                    elif self.shell_args.task == "exploration":
                                        log_dict["single_episode/visited_states"] = self.train_stats["visited_states"][
                                            ii
                                        ]
                                    elif self.shell_args.task == "flee":
                                        log_dict["single_episode/distance_from_start"] = self.train_stats[
                                            "distance_from_start"
                                        ][ii]
                                    self.logger.dict_log(
                                        log_dict, step=(total_num_steps + self.shell_args.num_processes * step + ii)
                                    )

                                episode_rewards.append(current_episode_rewards[ii])
                                current_episode_rewards[ii] = 0
                                episode_lengths.append(current_episode_lengths[ii])
                                current_episode_lengths[ii] = 0
                                self.train_stats["start_geodesic_distance"][ii] = obs["goal_geodesic_distance"][ii]

                        # If done then clean the history of observations.
                        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
                        bad_masks = torch.FloatTensor(
                            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
                        )

                        self.rollouts.insert(
                            obs, recurrent_hidden_states, action, action_log_prob, value, rewards, masks, bad_masks
                        )

                with torch.no_grad():
                    start_t = time.time()
                    next_value = self.agent.get_value(
                        {
                            "images": self.rollouts.obs[-1],
                            "target_vector": self.rollouts.additional_observations_dict["pointgoal"][-1],
                            "prev_action_one_hot": self.rollouts.additional_observations_dict["prev_action_one_hot"][
                                -1
                            ],
                        },
                        self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1],
                    ).detach()
                    timers[1] += time.time() - start_t

                self.rollouts.compute_returns(
                    next_value, self.shell_args.use_gae, self.shell_args.gamma, self.shell_args.tau
                )

                if not self.shell_args.no_weight_update:
                    start_t = time.time()
                    if self.shell_args.algo == "supervised":
                        (
                            total_loss,
                            action_loss,
                            visual_loss_total,
                            visual_loss_dict,
                            egomotion_loss,
                            forward_model_loss,
                        ) = self.optimizer.update(self.rollouts, self.shell_args)
                    else:
                        (
                            total_loss,
                            value_loss,
                            action_loss,
                            dist_entropy,
                            visual_loss_total,
                            visual_loss_dict,
                            egomotion_loss,
                            forward_model_loss,
                        ) = self.optimizer.update(self.rollouts, self.shell_args)

                    timers[2] += time.time() - start_t

                self.rollouts.after_update()

                # save for every interval-th episode or for the last epoch
                if iter_count % self.shell_args.save_interval == 0 or iter_count == num_updates - 1:
                    self.save_checkpoint(5, total_num_steps)

                total_num_steps += self.shell_args.num_processes * self.shell_args.num_forward_rollout_steps

                if not self.shell_args.no_weight_update and iter_count % self.shell_args.log_interval == 0:
                    log_dict = {}
                    if len(episode_rewards) > 1:
                        end = time.time()
                        nsteps = total_num_steps - fps_timer[1]
                        fps = int((total_num_steps - fps_timer[1]) / (end - fps_timer[0]))
                        timers /= nsteps
                        env_spf = timers[0]
                        forward_spf = timers[1]
                        backward_spf = timers[2]
                        print(
                            (
                                "{} Updates {}, num timesteps {}, FPS {}, Env FPS "
                                "{}, \n Last {} training episodes: mean/median reward "
                                "{:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n"
                            ).format(
                                datetime.datetime.now(),
                                iter_count,
                                total_num_steps,
                                fps,
                                int(1.0 / env_spf),
                                len(episode_rewards),
                                np.mean(episode_rewards),
                                np.median(episode_rewards),
                                np.min(episode_rewards),
                                np.max(episode_rewards),
                            )
                        )

                        if self.shell_args.tensorboard:
                            log_dict.update(
                                {
                                    "stats/full_spf": 1.0 / (fps + 1e-10),
                                    "stats/env_spf": env_spf,
                                    "stats/forward_spf": forward_spf,
                                    "stats/backward_spf": backward_spf,
                                    "stats/full_fps": fps,
                                    "stats/env_fps": 1.0 / (env_spf + 1e-10),
                                    "stats/forward_fps": 1.0 / (forward_spf + 1e-10),
                                    "stats/backward_fps": 1.0 / (backward_spf + 1e-10),
                                    "episode/mean_rewards": np.mean(episode_rewards),
                                    "episode/median_rewards": np.median(episode_rewards),
                                    "episode/min_rewards": np.min(episode_rewards),
                                    "episode/max_rewards": np.max(episode_rewards),
                                    "episode/mean_lengths": np.mean(episode_lengths),
                                    "episode/median_lengths": np.median(episode_lengths),
                                    "episode/min_lengths": np.min(episode_lengths),
                                    "episode/max_lengths": np.max(episode_lengths),
                                }
                            )
                        fps_timer[0] = time.time()
                        fps_timer[1] = total_num_steps
                        timers[:] = 0
                    if self.shell_args.tensorboard:
                        log_dict.update(
                            {
                                "loss/action": action_loss,
                                "loss/0_total": total_loss,
                                "loss/visual/0_total": visual_loss_total,
                                "loss/exploration/egomotion": egomotion_loss,
                                "loss/exploration/forward_model": forward_model_loss,
                            }
                        )
                        if self.shell_args.algo != "supervised":
                            log_dict.update({"loss/entropy": dist_entropy, "loss/value": value_loss})
                        for key, val in visual_loss_dict.items():
                            log_dict["loss/visual/" + key] = val
                        self.logger.dict_log(log_dict, step=total_num_steps)

                if self.shell_args.eval_interval is not None and total_num_steps % self.shell_args.eval_interval < (
                    self.shell_args.num_processes * self.shell_args.num_forward_rollout_steps
                ):
                    self.save_checkpoint(-1, total_num_steps)
                    self.set_log_iter(total_num_steps)
                    self.evaluate_model()
                    # reset the env datasets
                    self.envs.unwrapped.call(
                        ["switch_dataset"] * self.shell_args.num_processes, [("train",)] * self.shell_args.num_processes
                    )
                    obs = self.envs.reset()
                    if self.compute_surface_normals:
                        obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
                    obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                    if self.shell_args.algo == "supervised":
                        obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE])
                    self.rollouts.copy_obs(obs, 0)
                    distances = pt_util.to_numpy_array(obs["goal_geodesic_distance"])
                    self.train_stats["start_geodesic_distance"][:] = distances
                    previous_visual_features = None
                    egomotion_pred = None
                    prev_action = None
                    prev_action_probs = None
        except:
            # Catch all exceptions so a final save can be performed
            import traceback

            traceback.print_exc()
        finally:
            self.save_checkpoint(-1, total_num_steps)


def main():
    runner = HabitatRLTrainAndEvalRunner()
    runner.train_model()


if __name__ == "__main__":
    main()
