#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import glob
import os
import random
import time
from collections import OrderedDict
from collections import deque

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from habitat.datasets import make_dataset
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from base_habitat_rl_runner import ACTION_SPACE, ACTION_SPACE_TO_SIM_ACTION, SIM_ACTION_TO_NAME
from base_habitat_rl_runner import BaseHabitatRLRunner
from reinforcement_learning.get_config import get_dataset_config
from utils import drawing
from utils import pytorch_util as pt_util
from utils import tensorboard_logger

REWARD_SCALAR = 1.0


def get_eval_dataset(shell_args, data_subset="val"):
    if shell_args.dataset == "suncg":
        data_path = "data/datasets/pointnav/suncg/v1/{split}/{split}.json.gz"
    elif shell_args.dataset == "mp3d":
        data_path = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
    elif shell_args.dataset == "gibson":
        data_path = "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
    else:
        raise NotImplementedError("No rule for this dataset.")

    config = get_dataset_config(data_path, data_subset, shell_args.max_episode_length, 0, [], [])
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    assert len(dataset.episodes) > 0, "empty datasets"
    return dataset


class HabitatRLEvalRunner(BaseHabitatRLRunner):
    def __init__(self, create_decoder=True):
        self.eval_datasets = None
        self.eval_dataset_sizes = None
        self.eval_logger = None
        self.num_eval_episodes_total = None
        self.eval_dir = None
        self.log_iter = None

        super(HabitatRLEvalRunner, self).__init__(create_decoder)

    def setup(self, create_decoder):
        super(HabitatRLEvalRunner, self).setup(create_decoder)
        eval_dataset = get_eval_dataset(self.shell_args)
        if self.shell_args.record_video:
            random.shuffle(eval_dataset.episodes)
        self.num_eval_episodes_total = len(eval_dataset.episodes)
        self.eval_datasets = eval_dataset.get_splits(self.shell_args.num_processes, allow_uneven_splits=True)
        self.eval_logger = None
        if self.shell_args.tensorboard and self.shell_args.eval_interval is not None:
            self.eval_logger = tensorboard_logger.Logger(
                os.path.join(self.shell_args.log_prefix, self.shell_args.tensorboard_dirname, self.time_str + "_test")
            )
        self.datasets = {"val": self.eval_datasets}

        self.eval_dir = os.path.join(
            self.shell_args.log_prefix, self.shell_args.results_dirname, self.shell_args.data_subset
        )
        self.set_log_iter(self.start_iter)

    def set_log_iter(self, iteration):
        self.log_iter = iteration
        if self.eval_logger is not None:
            self.eval_logger.count = iteration

    def evaluate_model(self):
        self.envs.unwrapped.call(
            ["switch_dataset"] * self.shell_args.num_processes, [("val",)] * self.shell_args.num_processes
        )

        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        try:
            eval_net_file_name = sorted(
                glob.glob(os.path.join(self.shell_args.log_prefix, self.shell_args.checkpoint_dirname, "*") + "/*.pt"),
                key=os.path.getmtime,
            )[-1]
            eval_net_file_name = (
                self.shell_args.log_prefix.replace(os.sep, "_")
                + "_"
                + "_".join(eval_net_file_name.split(os.sep)[-2:])[:-3]
            )
        except IndexError:
            print("Warning, no weights found")
            eval_net_file_name = "random_weights"
        eval_output_file = open(os.path.join(self.eval_dir, eval_net_file_name + ".csv"), "w")
        print("Writing results to", eval_output_file.name)

        # Save the evaled net for posterity
        if self.shell_args.save_checkpoints:
            save_model = self.agent
            pt_util.save(
                save_model,
                os.path.join(self.shell_args.log_prefix, self.shell_args.checkpoint_dirname, "eval_weights"),
                num_to_keep=-1,
                iteration=self.log_iter,
            )
            print("Wrote model to file for safe keeping")

        obs = self.envs.reset()
        if self.compute_surface_normals:
            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        recurrent_hidden_states = torch.zeros(
            self.shell_args.num_processes,
            self.agent.recurrent_hidden_state_size,
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.ones(self.shell_args.num_processes, 1, dtype=torch.float32, device=self.device)

        episode_rewards = deque(maxlen=10)
        current_episode_rewards = np.zeros(self.shell_args.num_processes)
        episode_lengths = deque(maxlen=10)
        current_episode_lengths = np.zeros(self.shell_args.num_processes)

        total_num_steps = self.log_iter
        fps_timer = [time.time(), total_num_steps]
        timers = np.zeros(3)

        num_episodes = 0

        print("Config\n", self.configs[0])

        # Initialize every time eval is run rather than just at the start
        dataset_sizes = np.array([len(dataset.episodes) for dataset in self.eval_datasets])

        eval_stats = dict(
            episode_ids=[None for _ in range(self.shell_args.num_processes)],
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
        eval_stats_means = dict(
            num_episodes=0,
            num_steps=0,
            reward=0,
            spl=0,
            visited_states=0,
            success=0,
            end_geodesic_distance=0,
            start_geodesic_distance=0,
            delta_geodesic_distance=0,
            distance_from_start=0,
        )
        eval_output_file.write("name,%s,iter,%d\n\n" % (eval_net_file_name, self.log_iter))
        if self.shell_args.task == "pointnav":
            eval_output_file.write(
                (
                    "episode_id,num_steps,reward,spl,success,start_geodesic_distance,"
                    "end_geodesic_distance,delta_geodesic_distance\n"
                )
            )
        elif self.shell_args.task == "exploration":
            eval_output_file.write("episode_id,reward,visited_states\n")
        elif self.shell_args.task == "flee":
            eval_output_file.write("episode_id,reward,distance_from_start\n")
        distances = pt_util.to_numpy_array(obs["goal_geodesic_distance"])
        eval_stats["start_geodesic_distance"][:] = distances
        progress_bar = tqdm.tqdm(total=self.num_eval_episodes_total)
        all_done = False
        iter_count = 0
        video_frames = []
        previous_visual_features = None
        egomotion_pred = None
        prev_action = None
        prev_action_probs = None
        if hasattr(self.agent.base, "enable_decoder"):
            if self.shell_args.record_video:
                self.agent.base.enable_decoder()
            else:
                self.agent.base.disable_decoder()
        while not all_done:
            with torch.no_grad():
                start_t = time.time()
                value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                    {
                        "images": obs["rgb"].to(self.device),
                        "target_vector": obs["pointgoal"].to(self.device),
                        "prev_action_one_hot": obs["prev_action_one_hot"].to(self.device),
                    },
                    recurrent_hidden_states,
                    masks,
                )
                action_cpu = pt_util.to_numpy_array(action.squeeze(1))
                translated_action_space = ACTION_SPACE[action_cpu]

                timers[1] += time.time() - start_t

                if self.shell_args.record_video:
                    if self.shell_args.use_motion_loss:
                        if previous_visual_features is not None:
                            egomotion_pred = self.agent.base.predict_egomotion(
                                self.agent.base.visual_features, previous_visual_features
                            )
                        previous_visual_features = self.agent.base.visual_features.detach()

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
                    if hasattr(self.agent.base, "decoder_outputs") and self.agent.base.decoder_outputs is not None:
                        min_channel = 0
                        for key, num_channels in self.agent.base.decoder_output_info:
                            outputs = self.agent.base.decoder_outputs[:, min_channel : min_channel + num_channels, ...]
                            draw_obs["output_" + key] = pt_util.to_numpy_array(outputs).copy()
                            min_channel += num_channels
                    draw_obs["rewards"] = eval_stats["reward"]
                    draw_obs["step"] = current_episode_lengths.copy()
                    draw_obs["method"] = self.shell_args.method_name
                    if best_next_action is not None:
                        draw_obs["best_next_action"] = best_next_action
                    if self.shell_args.use_motion_loss:
                        if egomotion_pred is not None:
                            draw_obs["egomotion_pred"] = pt_util.to_numpy_array(F.softmax(egomotion_pred, dim=1)).copy()
                        else:
                            draw_obs["egomotion_pred"] = None
                    images, titles, normalize = drawing.obs_to_images(draw_obs)
                    im_inds = [0, 2, 3, 1, 6, 7, 8, 5]
                    height, width = images[0].shape[:2]
                    subplot_image = drawing.subplot(
                        images,
                        2,
                        4,
                        titles=titles,
                        normalize=normalize,
                        output_width=max(width, 320),
                        output_height=max(height, 320),
                        order=im_inds,
                        fancy_text=True,
                    )
                    video_frames.append(subplot_image)

                # save dists from previous step or else on reset they will be overwritten
                distances = pt_util.to_numpy_array(obs["goal_geodesic_distance"])

                start_t = time.time()
                obs, rewards, dones, infos = self.envs.step(translated_action_space)
                timers[0] += time.time() - start_t
                obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                rewards *= REWARD_SCALAR
                rewards = np.clip(rewards, -10, 10)

                if self.shell_args.record_video and not dones[0]:
                    obs["top_down_map"] = infos[0]["top_down_map"]

                if self.compute_surface_normals:
                    obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))

                current_episode_rewards += pt_util.to_numpy_array(rewards).squeeze()
                current_episode_lengths += 1
                to_pause = []
                for ii, done_e in enumerate(dones):
                    if done_e:
                        num_episodes += 1

                        if self.shell_args.record_video:
                            if "top_down_map" in infos[ii]:
                                video_dir = os.path.join(self.shell_args.log_prefix, "videos")
                                if not os.path.exists(video_dir):
                                    os.makedirs(video_dir)
                                im_path = os.path.join(
                                    self.shell_args.log_prefix, "videos", "total_steps_%d.png" % total_num_steps
                                )
                                top_down_map = maps.colorize_topdown_map(infos[ii]["top_down_map"]["map"])
                                imageio.imsave(im_path, top_down_map)

                            images_to_video(
                                video_frames,
                                os.path.join(self.shell_args.log_prefix, "videos"),
                                "total_steps_%d" % total_num_steps,
                            )
                            video_frames = []

                        eval_stats["episode_ids"][ii] = infos[ii]["episode_id"]

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
                            eval_stats["spl"][ii] = infos[ii]["spl"]
                            eval_stats["success"][ii] = eval_stats["spl"][ii] > 0
                            eval_stats["num_steps"][ii] = current_episode_lengths[ii]
                            eval_stats["end_geodesic_distance"][ii] = (
                                infos[ii]["final_distance"] if eval_stats["success"][ii] else distances[ii]
                            )
                            eval_stats["delta_geodesic_distance"][ii] = (
                                eval_stats["start_geodesic_distance"][ii] - eval_stats["end_geodesic_distance"][ii]
                            )
                        elif self.shell_args.task == "exploration":
                            print(
                                "FINISHED EPISODE %d Reward %.3f States Visited %d"
                                % (num_episodes, current_episode_rewards[ii], infos[ii]["visited_states"])
                            )
                            eval_stats["visited_states"][ii] = infos[ii]["visited_states"]
                        elif self.shell_args.task == "flee":
                            print(
                                "FINISHED EPISODE %d Reward %.3f Distance from start %.4f"
                                % (num_episodes, current_episode_rewards[ii], infos[ii]["distance_from_start"])
                            )
                            eval_stats["distance_from_start"][ii] = infos[ii]["distance_from_start"]

                        eval_stats["num_episodes"][ii] += 1
                        eval_stats["reward"][ii] = current_episode_rewards[ii]

                        if eval_stats["num_episodes"][ii] <= dataset_sizes[ii]:
                            progress_bar.update(1)
                            eval_stats_means["num_episodes"] += 1
                            eval_stats_means["reward"] += eval_stats["reward"][ii]
                            if self.shell_args.task == "pointnav":
                                eval_output_file.write(
                                    "%s,%d,%f,%f,%d,%f,%f,%f\n"
                                    % (
                                        eval_stats["episode_ids"][ii],
                                        eval_stats["num_steps"][ii],
                                        eval_stats["reward"][ii],
                                        eval_stats["spl"][ii],
                                        eval_stats["success"][ii],
                                        eval_stats["start_geodesic_distance"][ii],
                                        eval_stats["end_geodesic_distance"][ii],
                                        eval_stats["delta_geodesic_distance"][ii],
                                    )
                                )
                                eval_stats_means["num_steps"] += eval_stats["num_steps"][ii]
                                eval_stats_means["spl"] += eval_stats["spl"][ii]
                                eval_stats_means["success"] += eval_stats["success"][ii]
                                eval_stats_means["start_geodesic_distance"] += eval_stats["start_geodesic_distance"][ii]
                                eval_stats_means["end_geodesic_distance"] += eval_stats["end_geodesic_distance"][ii]
                                eval_stats_means["delta_geodesic_distance"] += eval_stats["delta_geodesic_distance"][ii]
                            elif self.shell_args.task == "exploration":
                                eval_output_file.write(
                                    "%s,%f,%d\n"
                                    % (
                                        eval_stats["episode_ids"][ii],
                                        eval_stats["reward"][ii],
                                        eval_stats["visited_states"][ii],
                                    )
                                )
                                eval_stats_means["visited_states"] += eval_stats["visited_states"][ii]
                            elif self.shell_args.task == "flee":
                                eval_output_file.write(
                                    "%s,%f,%f\n"
                                    % (
                                        eval_stats["episode_ids"][ii],
                                        eval_stats["reward"][ii],
                                        eval_stats["distance_from_start"][ii],
                                    )
                                )
                                eval_stats_means["distance_from_start"] += eval_stats["distance_from_start"][ii]
                            eval_output_file.flush()
                            if eval_stats["num_episodes"][ii] == dataset_sizes[ii]:
                                to_pause.append(ii)

                        episode_rewards.append(current_episode_rewards[ii])
                        current_episode_rewards[ii] = 0
                        episode_lengths.append(current_episode_lengths[ii])
                        current_episode_lengths[ii] = 0
                        eval_stats["start_geodesic_distance"][ii] = obs["goal_geodesic_distance"][ii]

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones]).to(self.device)

                # Reverse in order to maintain order in case of multiple.
                to_pause.reverse()
                for ii in to_pause:
                    # Pause the environments that are done from the vectorenv.
                    print("Pausing env", ii)
                    self.envs.unwrapped.pause_at(ii)
                    current_episode_rewards = np.concatenate(
                        (current_episode_rewards[:ii], current_episode_rewards[ii + 1 :])
                    )
                    current_episode_lengths = np.concatenate(
                        (current_episode_lengths[:ii], current_episode_lengths[ii + 1 :])
                    )
                    for key in eval_stats:
                        eval_stats[key] = np.concatenate((eval_stats[key][:ii], eval_stats[key][ii + 1 :]))
                    dataset_sizes = np.concatenate((dataset_sizes[:ii], dataset_sizes[ii + 1 :]))

                    for key in obs:
                        if type(obs[key]) == torch.Tensor:
                            obs[key] = torch.cat((obs[key][:ii], obs[key][ii + 1 :]), dim=0)
                        else:
                            obs[key] = np.concatenate((obs[key][:ii], obs[key][ii + 1 :]), axis=0)

                    recurrent_hidden_states = torch.cat(
                        (recurrent_hidden_states[:ii], recurrent_hidden_states[ii + 1 :]), dim=0
                    )
                    masks = torch.cat((masks[:ii], masks[ii + 1 :]), dim=0)

                if len(dataset_sizes) == 0:
                    progress_bar.close()
                    all_done = True

            total_num_steps += self.shell_args.num_processes

            if iter_count % (self.shell_args.log_interval * 100) == 0:
                log_dict = {}
                if len(episode_rewards) > 1:
                    end = time.time()
                    nsteps = total_num_steps - fps_timer[1]
                    fps = int((total_num_steps - fps_timer[1]) / (end - fps_timer[0]))
                    timers /= nsteps
                    env_spf = timers[0]
                    forward_spf = timers[1]
                    print(
                        (
                            "{} Updates {}, num timesteps {}, FPS {}, Env FPS {}, "
                            "\n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, "
                            "min/max reward {:.3f}/{:.3f}\n"
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
                                "stats/full_fps": fps,
                                "stats/env_fps": 1.0 / (env_spf + 1e-10),
                                "stats/forward_fps": 1.0 / (forward_spf + 1e-10),
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
                        self.eval_logger.dict_log(log_dict, step=self.log_iter)
                    fps_timer[0] = time.time()
                    fps_timer[1] = total_num_steps
                    timers[:] = 0
            iter_count += 1
        print("Finished testing")
        print("Wrote results to", eval_output_file.name)

        eval_stats_means = {key: val / eval_stats_means["num_episodes"] for key, val in eval_stats_means.items()}
        if self.shell_args.tensorboard:
            log_dict = {"single_episode/reward": eval_stats_means["reward"]}
            if self.shell_args.task == "pointnav":
                log_dict.update(
                    {
                        "single_episode/num_steps": eval_stats_means["num_steps"],
                        "single_episode/spl": eval_stats_means["spl"],
                        "single_episode/success": eval_stats_means["success"],
                        "single_episode/start_geodesic_distance": eval_stats_means["start_geodesic_distance"],
                        "single_episode/end_geodesic_distance": eval_stats_means["end_geodesic_distance"],
                        "single_episode/delta_geodesic_distance": eval_stats_means["delta_geodesic_distance"],
                    }
                )
            elif self.shell_args.task == "exploration":
                log_dict["single_episode/visited_states"] = eval_stats_means["visited_states"]
            elif self.shell_args.task == "flee":
                log_dict["single_episode/distance_from_start"] = eval_stats_means["distance_from_start"]
            self.eval_logger.dict_log(log_dict, step=self.log_iter)
        self.envs.unwrapped.resume_all()


def main():
    runner = HabitatRLEvalRunner()
    runner.evaluate_model()


if __name__ == "__main__":
    main()
