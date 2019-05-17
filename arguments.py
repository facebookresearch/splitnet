#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument("--algo", default="ppo", help="algorithm to use: ppo | supervised")

    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)")

    parser.add_argument("--eps", type=float, default=1e-5, help="Adam optimizer epsilon (default: 1e-5)")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards (default: 0.99)")

    parser.add_argument("--use-gae", action="store_true", default=False, help="use generalized advantage estimation")

    parser.add_argument("--tau", type=float, default=0.95, help="gae parameter (default: 0.95)")

    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")

    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)")

    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max norm of gradients (default: 0.5)")

    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    parser.add_argument(
        "--num-processes", type=int, default=16, help="how many training CPU processes to use (default: 16)"
    )

    parser.add_argument(
        "--num-forward-rollout-steps", type=int, default=8, help="number of forward steps in rollouts (default: 8)"
    )

    parser.add_argument(
        "--max-episode-length", type=int, default=500, help="maximum number of steps in a single episode (default: 500)"
    )

    parser.add_argument("--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)")

    parser.add_argument("--num-mini-batch", type=int, default=1, help="number of batches for ppo (default: 1)")

    parser.add_argument("--clip-param", type=float, default=0.2, help="ppo clip parameter (default: 0.2)")

    parser.add_argument(
        "--log-interval", type=int, default=10, help="log interval, one log per n updates (default: 10)"
    )

    parser.add_argument(
        "--save-interval", type=int, default=100, help="save interval, one save per n updates (default: 100)"
    )

    parser.add_argument(
        "--num-env-steps", type=int, default=int(1e8), help="number of environment steps to train (default: 1e8)"
    )

    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")

    parser.add_argument(
        "--use-linear-lr-decay", action="store_true", default=False, help="use a linear schedule on the learning rate"
    )

    parser.add_argument(
        "--use-linear-clip-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the ppo clipping parameter",
    )

    parser.add_argument("--no-tensorboard", action="store_true", default=False, help="disable tensorboard logging")

    parser.add_argument(
        "--clear-weights", action="store_true", default=False, help="do not load previous model weights if they exist"
    )

    parser.add_argument(
        "--dataset-dir", type=str, default="data/datasets/pointnav/habitat_test_scenes/v1/train", help="path to dataset"
    )

    parser.add_argument(
        "--use-multithreading",
        action="store_true",
        default=False,
        help="use multithreading instead of multiprocessing (can be useful for debugging)",
    )

    parser.add_argument(
        "--record-video", action="store_true", default=False, help="record the results of forward passes in a video"
    )

    parser.add_argument(
        "--no-weight-update",
        action="store_true",
        default=False,
        help="Do not do weight updates. Useful for looking at checkpoints/eval.",
    )

    parser.add_argument(
        "--data-subset",
        type=str,
        default="train_small",
        help="Picks which data subset to load: train | val | test | train_small",
        required=True,
    )

    parser.add_argument(
        "--dataset", type=str, default="mp3d", help="Picks which dataset to load: mp3d | gibson | suncg", required=True
    )

    parser.add_argument(
        "--freeze-encoder-features", action="store_true", default=False, help="Does not update the encoder features."
    )

    parser.add_argument(
        "--freeze-visual-decoder-features",
        action="store_true",
        default=False,
        help="Does not update the visual decoder features. Gradients still flow through if the loss is enabled.",
    )

    parser.add_argument(
        "--freeze-motion-decoder-features",
        action="store_true",
        default=False,
        help="Does not update the motion decoder features. Gradients still flow through if the loss is enabled.",
    )

    parser.add_argument(
        "--freeze-policy-decoder-features",
        action="store_true",
        default=False,
        help="Does not update the policy decoder features. Gradients still flow through if the loss is enabled.",
    )

    parser.add_argument("--no-visual-loss", action="store_true", default=False, help="disable visual loss")

    parser.add_argument("--no-motion-loss", action="store_true", default=False, help="disable motion loss")

    parser.add_argument("--no-policy-loss", action="store_true", default=False, help="disable policy loss")

    parser.add_argument("--log-prefix", type=str, default="output_files", required=True, help="path to logs, checkpoints, etc.")

    parser.add_argument(
        "--tensorboard-dirname", type=str, default="tensorboard", help="path under log-prefix for tensorboard logs."
    )

    parser.add_argument(
        "--checkpoint-dirname", type=str, default="checkpoints", help="path under log-prefix for checkpoints."
    )

    parser.add_argument(
        "--results-dirname", type=str, default="results", help="path under log-prefix for results logs."
    )

    parser.add_argument("--no-save-checkpoints", action="store_true", default=False, help="disable saving checkpoints")

    parser.add_argument("--debug", action="store_true", default=False, help="Sets the debug flag.")

    parser.add_argument(
        "--eval-interval", type=int, default=None, help="eval interval, one eval per n updates (default: No eval)"
    )

    parser.add_argument(
        "--end-to-end",
        action="store_true",
        default=False,
        help="If true, gradients flow from RL back through the visual features.",
    )

    parser.add_argument("--encoder-network-type", type=str, required=True, help="Name of the encoder network type.")

    parser.add_argument("--render-gpu-ids", type=str, required=True, help="ID(s) of the gpu(s) to use for rendering.")

    parser.add_argument(
        "--pytorch-gpu-ids",
        type=str,
        required=True,
        help="ID(s) of the gpu(s) to use for pytorch. " "The first will be where the data unrolls are located.",
    )

    parser.add_argument(
        "--task", type=str, required=True, help="Name of the task that is being used: pointnav | exploration | flee"
    )

    parser.add_argument(
        "--num-train-scenes", type=int, default=-1, help="Number of scenes to use during training (if >0)."
    )

    parser.add_argument(
        "--blind", action="store_true", default=False, help="Blind the network (no images fed in as input)."
    )

    parser.add_argument("--method-name", type=str, default="SplitNet", help="Name of the method that is being used.")

    parser.add_argument(
        "--saved-variable-prefix",
        type=str,
        default="",
        help="Prefix on variable names that should be renamed. This should usually be left empty.",
    )

    parser.add_argument(
        "--new-variable-prefix",
        type=str,
        default="",
        help="New prefix on variables that will be renamed. This should usually be left empty.",
    )

    args = parser.parse_args()

    args.tensorboard = not args.no_tensorboard
    args.load_model = not args.clear_weights
    args.num_envs = args.num_processes
    args.save_checkpoints = not args.no_save_checkpoints

    args.update_encoder_features = not args.freeze_encoder_features
    args.update_visual_decoder_features = not args.freeze_visual_decoder_features
    args.update_motion_decoder_features = not args.freeze_motion_decoder_features
    args.update_policy_decoder_features = not args.freeze_policy_decoder_features

    args.use_visual_loss = not args.no_visual_loss
    args.use_motion_loss = not args.no_motion_loss
    args.use_policy_loss = not args.no_policy_loss

    assert not args.record_video or (
        args.num_processes == 1
    ), "Video only supports one process. Otherwise the videos get too busy."

    if not args.no_weight_update:
        assert (
            args.no_visual_loss or args.update_encoder_features
        ), "Must update encoder features if using visual loss. Otherwise what's the point?"
        assert args.use_policy_loss or args.use_motion_loss or args.use_visual_loss, "Must enable at least one loss"
        if args.use_visual_loss:
            assert (
                args.update_encoder_features or args.update_visual_decoder_features
            ), "Must update something affected by visual loss."
        if args.use_motion_loss:
            assert (
                args.update_encoder_features and args.end_to_end
            ) or args.update_motion_decoder_features, "Must update something affected by motion loss."
        if args.use_policy_loss:
            assert (
                args.update_encoder_features and args.end_to_end
            ) or args.update_policy_decoder_features, "Must update something affected by policy loss."
        assert (
            args.update_encoder_features
            or args.update_motion_decoder_features
            or args.update_policy_decoder_features
            or args.update_visual_decoder_features
        ), "Must enable at least one set of weights to update"
        if args.no_visual_loss:
            assert args.freeze_visual_decoder_features
        if args.no_motion_loss:
            assert args.freeze_motion_decoder_features
        if args.no_policy_loss:
            assert args.freeze_policy_decoder_features
        if args.update_encoder_features and args.use_policy_loss:
            if not args.end_to_end:
                print(
                    "Warning, no gradients will propagate to the encoder from the policy without the end-to-end flag."
                )
    return args
