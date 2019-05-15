#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.optim as optim
from dg_util.python_utils import pytorch_util as pt_util
from torch import nn


def get_visual_loss(outputs, label_dict, output_info):
    min_channel = 0
    visual_losses = {}
    for key, num_channels in output_info:
        outputs_on = outputs[:, min_channel : min_channel + num_channels, ...]
        if key == "reconstruction":
            labels = label_dict["rgb"].to(torch.float32) / 128.0 - 1
            visual_loss = F.l1_loss(outputs_on, labels, reduction="none")
        else:
            labels = label_dict[key]
            if key == "depth":
                visual_loss = F.l1_loss(outputs_on, labels, reduction="none")
            elif key == "semantic":
                assert labels.max() < outputs_on.shape[1]
                visual_loss = 0.25 * F.cross_entropy(outputs_on, labels, reduction="none")
            elif key == "surface_normals":
                # Normalize
                outputs_on = outputs_on / outputs_on.norm(dim=1, keepdim=True)
                # Cosine similarity
                visual_loss = -torch.sum(outputs_on * labels, dim=1, keepdim=True)
            else:
                raise NotImplementedError("Loss not implemented")
        visual_loss = torch.mean(visual_loss)
        if key == "surface_normals":
            visual_loss = visual_loss + 1  # just a shift so it's not negative
        visual_losses[key] = visual_loss
        min_channel += num_channels

    visual_loss_total = sum(visual_losses.values())
    visual_losses = {key: val.item() for key, val in visual_losses.items()}
    visual_loss_value = visual_loss_total.item()
    return visual_loss_total, visual_loss_value, visual_losses


def get_object_existence_loss(outputs, labels):
    loss = F.binary_cross_entropy_with_logits(outputs, labels)
    return loss


def get_visual_loss_with_rollout(rollouts, decoder_output_info, decoder_outputs):
    labels = rollouts.additional_observations_dict.copy()
    labels["rgb"] = rollouts.obs
    labels = {key: pt_util.remove_dim(val[:-1], 1) for key, val in labels.items()}
    return get_visual_loss(decoder_outputs, labels, decoder_output_info)


def get_egomotion_loss(actions, egomotion_pred):
    loss = F.cross_entropy(egomotion_pred, actions, reduction="none")
    return loss


def get_feature_prediction_loss(features, features_pred):
    loss = 1 - F.cosine_similarity(features, features_pred, dim=2)
    return loss


class Optimizer(object):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=lr, eps=eps)

        self.verified_gradient_propagation = [False] * 2


class VisualPPO(Optimizer):
    def update(self, rollouts, shell_args):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_total_epoch = 0
        visual_loss_value = 0
        egomotion_loss_value = 0
        feature_prediction_loss_value = 0
        visual_losses = {}

        if hasattr(self.actor_critic.base, "decoder_enabled") and (
            shell_args.use_visual_loss or shell_args.use_motion_loss
        ):

            if shell_args.freeze_encoder_features:
                visual_features = pt_util.remove_dim(
                    rollouts.additional_observations_dict["visual_encoder_features"][:-1], 1
                )
            else:
                visual_features, decoder_outputs, class_pred = self.actor_critic.base.visual_encoder(
                    pt_util.remove_dim(rollouts.obs[:-1], 1), shell_args.use_visual_loss
                )

            visual_loss_total = 0
            egomotion_loss_total = 0
            feature_loss_total = 0

            assert (
                shell_args.no_visual_loss or shell_args.update_encoder_features
            ), "Must update encoder features if using visual loss. Otherwise what's the point?"
            if shell_args.use_visual_loss:
                visual_loss_total, visual_loss_value, visual_losses = get_visual_loss_with_rollout(
                    rollouts, self.actor_critic.base.decoder_output_info, decoder_outputs
                )

            if shell_args.use_motion_loss:
                visual_features = self.actor_critic.base.visual_projection(visual_features)

                visual_features = visual_features.view(rollouts.obs.shape[0] - 1, rollouts.obs.shape[1], -1)
                actions = rollouts.actions[:-1].view(-1)
                egomotion_pred = self.actor_critic.base.predict_egomotion(visual_features[1:], visual_features[:-1])

                egomotion_loss = get_egomotion_loss(actions, egomotion_pred)
                egomotion_loss = egomotion_loss * rollouts.masks[1:-1].view(-1)
                egomotion_loss_total = 0.25 * torch.mean(egomotion_loss)
                egomotion_loss_value = egomotion_loss_total.item()

                action_one_hot = pt_util.get_one_hot(actions, self.actor_critic.num_actions)
                next_feature_pred = self.actor_critic.base.predict_next_features(visual_features[:-1], action_one_hot)
                feature_loss = get_feature_prediction_loss(
                    visual_features[1:].detach(), next_feature_pred.view(visual_features[1:].shape)
                )
                feature_loss = feature_loss.view(-1) * rollouts.masks[1:-1].view(-1)
                feature_loss_total = torch.mean(feature_loss)

                feature_prediction_loss_value = feature_loss_total.item()

            self.optimizer.zero_grad()
            (visual_loss_total + egomotion_loss_total + feature_loss_total).backward()

            if not self.verified_gradient_propagation[0]:
                # Check that appropriate gradients are propagated.
                if shell_args.update_encoder_features:
                    for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                        assert param.grad is not None and param.grad.abs().sum().item() > 1e-6
                else:
                    for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                        assert param.grad is None

                if shell_args.use_visual_loss:
                    assert shell_args.update_visual_decoder_features or shell_args.update_encoder_features
                    if shell_args.update_visual_decoder_features:
                        for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                            assert param.grad is not None and param.grad.abs().sum().item() > 1e-10
                    else:
                        for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                            assert param.grad is None
                else:
                    assert shell_args.freeze_visual_decoder_features
                    for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                        assert param.grad is None

                if shell_args.use_motion_loss:
                    assert shell_args.update_motion_decoder_features or shell_args.update_encoder_features
                    if shell_args.update_motion_decoder_features:
                        for param in self.actor_critic.base.egomotion_layer.parameters():
                            assert param.grad is not None and param.grad.abs().sum().item() > 1e-10
                    else:
                        for param in self.actor_critic.base.egomotion_layer.parameters():
                            assert param.grad is None
                else:
                    assert shell_args.freeze_motion_decoder_features
                    for param in self.actor_critic.base.egomotion_layer.parameters():
                        assert param.grad is None
                self.verified_gradient_propagation[0] = True

            self.optimizer.step()
        else:
            assert shell_args.freeze_visual_decoder_features
            assert shell_args.freeze_motion_decoder_features
            assert shell_args.no_visual_loss
            assert shell_args.no_motion_loss

        decoder_enabled = hasattr(self.actor_critic.base, "decoder_enabled") and self.actor_critic.base.decoder_enabled
        if decoder_enabled:
            self.actor_critic.base.disable_decoder()
        if shell_args.use_policy_loss:
            for _ in range(self.ppo_epoch):
                if self.actor_critic.is_recurrent:
                    data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

                for sample in data_generator:
                    (
                        obs_batch,
                        recurrent_hidden_states_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                        additional_obs_batch,
                    ) = sample

                    try:
                        with torch.autograd.detect_anomaly():
                            inputs = {
                                "target_vector": additional_obs_batch["pointgoal"],
                                "prev_action_one_hot": additional_obs_batch["prev_action_one_hot"],
                            }
                            if (
                                "visual_encoder_features" in additional_obs_batch
                                and not self.actor_critic.base.end_to_end
                            ):
                                inputs["visual_encoder_features"] = additional_obs_batch["visual_encoder_features"]
                            else:
                                inputs["images"] = obs_batch
                            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                                inputs, recurrent_hidden_states_batch, masks_batch, actions_batch
                            )

                            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                            surr1 = ratio * adv_targ
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                            action_loss = -torch.min(surr1, surr2).mean()

                            if self.use_clipped_value_loss:
                                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                                    -self.clip_param, self.clip_param
                                )
                                value_losses = 2 * F.smooth_l1_loss(values, return_batch, reduction="none")
                                value_losses_clipped = 2 * F.smooth_l1_loss(
                                    value_pred_clipped, return_batch, reduction="none"
                                )
                                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                            else:
                                value_loss = F.smooth_l1_loss(return_batch, values, reduction="none")

                            action_loss = 10 * action_loss
                            value_loss = 10 * value_loss
                            value_loss = value_loss * self.value_loss_coef
                            dist_entropy = dist_entropy * self.entropy_coef
                            loss_total = value_loss + action_loss - dist_entropy
                            self.optimizer.zero_grad()
                            loss_total.backward()

                            if not shell_args.end_to_end:
                                for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                                    param.grad = None

                            if not self.verified_gradient_propagation[1]:
                                # Check that appropriate gradients are propagated.
                                if shell_args.update_encoder_features and shell_args.end_to_end:
                                    for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                                        assert param.grad is not None and param.grad.abs().sum().item() > 1e-10
                                else:
                                    for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                                        assert param.grad is None

                                if shell_args.use_policy_loss:
                                    assert (
                                        shell_args.update_policy_decoder_features or shell_args.update_encoder_features
                                    )
                                    if shell_args.update_policy_decoder_features:
                                        for param in self.actor_critic.base.rl_layers.parameters():
                                            assert param.grad is not None and param.grad.abs().sum().item() > 1e-10
                                    else:
                                        for param in self.actor_critic.base.rl_layers.parameters():
                                            assert param.grad is None
                                else:
                                    assert shell_args.freeze_policy_decoder_features
                                    for param in self.actor_critic.base.rl_layers.parameters():
                                        assert param.grad is None

                                self.verified_gradient_propagation[1] = True

                            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                    except RuntimeError as re:
                        import traceback

                        traceback.print_exc()
                        import pdb

                        pdb.set_trace()
                        print("anomoly", re)
                        raise re

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()
                    loss_total_epoch += loss_total.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        if decoder_enabled:
            self.actor_critic.base.enable_decoder()

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_total_epoch /= num_updates

        return (
            loss_total_epoch,
            value_loss_epoch,
            action_loss_epoch,
            dist_entropy_epoch,
            visual_loss_value,
            visual_losses,
            egomotion_loss_value,
            feature_prediction_loss_value,
        )


class BehavioralCloningOptimizer(Optimizer):
    def update(self, rollouts, shell_args):

        decoder_enabled = hasattr(self.actor_critic.base, "decoder_enabled") and self.actor_critic.base.decoder_enabled

        if shell_args.use_visual_loss:
            self.actor_critic.base.enable_decoder()

        visual_loss_value = 0
        egomotion_loss_value = 0
        feature_prediction_loss_value = 0
        visual_loss_total = 0
        visual_losses = {}

        T, N = rollouts.obs.shape[:2]

        if not self.actor_critic.base.aah_im_blind:
            if shell_args.update_encoder_features or self.actor_critic.base.end_to_end:
                visual_features, decoder_outputs, class_pred = self.actor_critic.base.visual_encoder(
                    pt_util.remove_dim(rollouts.obs[:-1], 1), shell_args.use_visual_loss
                )
                if shell_args.use_visual_loss:
                    visual_loss_total, visual_loss_value, visual_losses = get_visual_loss_with_rollout(
                        rollouts, self.actor_critic.base.decoder_output_info, decoder_outputs
                    )
            else:
                visual_features = pt_util.remove_dim(
                    rollouts.additional_observations_dict["visual_encoder_features"][:-1], 1
                )
        else:
            visual_features = None

        rl_features = visual_features
        if not self.actor_critic.base.aah_im_blind and not self.actor_critic.base.end_to_end:
            rl_features = rl_features.detach()
        value, actor_features, _ = self.actor_critic.base(
            {
                "visual_encoder_features": rl_features,
                "target_vector": pt_util.remove_dim(rollouts.additional_observations_dict["pointgoal"][:-1], 1),
                "prev_action_one_hot": pt_util.remove_dim(
                    rollouts.additional_observations_dict["prev_action_one_hot"][:-1], 1
                ),
            },
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
        )
        action_logits = self.actor_critic.dist(actor_features).logits

        label_probs = rollouts.additional_observations_dict["best_next_action"][:-1]
        label_probs /= torch.sum(label_probs, dim=2, keepdim=True) + 1e-10
        action_loss = pt_util.weighted_loss(
            torch.sum(
                pt_util.multi_class_cross_entropy_loss(
                    action_logits, pt_util.remove_dim(label_probs, 1), reduction="none"
                ),
                dim=1,
            ),
            pt_util.remove_dim(rollouts.masks[:-1], dim=(1, 2)),
        )
        action_loss_total = action_loss.item()

        total_loss = action_loss + visual_loss_total

        if shell_args.use_motion_loss:
            visual_features = self.actor_critic.base.visual_projection(visual_features).view(T - 1, N, -1)

            actions = rollouts.actions[:-1].view(-1)
            egomotion_pred = self.actor_critic.base.predict_egomotion(visual_features[1:], visual_features[:-1])

            egomotion_loss = get_egomotion_loss(actions, egomotion_pred)
            egomotion_loss = egomotion_loss * rollouts.masks[1:-1].view(-1)
            egomotion_loss_total = 0.25 * torch.mean(egomotion_loss)
            egomotion_loss_value = egomotion_loss_total.item()

            action_one_hot = pt_util.get_one_hot(actions, self.actor_critic.num_actions)
            next_feature_pred = self.actor_critic.base.predict_next_features(visual_features[:-1], action_one_hot)
            feature_loss = get_feature_prediction_loss(
                visual_features[1:].detach(), next_feature_pred.view(visual_features[1:].shape)
            )
            feature_loss = feature_loss.view(-1)
            feature_loss = feature_loss * rollouts.masks[1:-1].view(-1)
            feature_loss_total = torch.mean(feature_loss)

            feature_prediction_loss_value = feature_loss_total.item()

            total_loss = total_loss + (feature_loss_total + egomotion_loss_total)

        self.optimizer.zero_grad()
        total_loss.backward()

        if not self.verified_gradient_propagation[0]:
            self.verified_gradient_propagation[0] = True
            # Check that appropriate gradients are propagated.
            if shell_args.update_encoder_features:
                for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                    assert param.grad is not None
                    if param.grad.abs().sum().item() < 1e-10:
                        print("Warning, gradients are 0. If this message continues to appear, there is a problem.")
                        self.verified_gradient_propagation[0] = False
                        break
            else:
                for param in self.actor_critic.base.visual_encoder.module.encoder.parameters():
                    assert param.grad is None

            if shell_args.use_visual_loss:
                assert shell_args.update_visual_decoder_features or shell_args.update_encoder_features
                if shell_args.update_visual_decoder_features:
                    for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                        assert param.grad is not None
                else:
                    for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                        assert param.grad is None
            else:
                assert shell_args.freeze_visual_decoder_features
                for param in self.actor_critic.base.visual_encoder.module.decoder.parameters():
                    assert param.grad is None

            if shell_args.use_motion_loss:
                assert shell_args.update_motion_decoder_features or shell_args.update_encoder_features
                if shell_args.update_motion_decoder_features:
                    for param in self.actor_critic.base.egomotion_layer.parameters():
                        assert param.grad is not None
                    for param in self.actor_critic.base.motion_model_layer.parameters():
                        assert param.grad is not None
                else:
                    for param in self.actor_critic.base.egomotion_layer.parameters():
                        assert param.grad is None
                    for param in self.actor_critic.base.motion_model_layer.parameters():
                        assert param.grad is None
            else:
                assert shell_args.freeze_motion_decoder_features
                for param in self.actor_critic.base.egomotion_layer.parameters():
                    assert param.grad is None
                for param in self.actor_critic.base.motion_model_layer.parameters():
                    assert param.grad is None

            if shell_args.use_policy_loss:
                assert shell_args.update_policy_decoder_features or shell_args.update_encoder_features
                if shell_args.update_policy_decoder_features:
                    for param in self.actor_critic.base.rl_layers.parameters():
                        assert param.grad is not None
                else:
                    for param in self.actor_critic.base.rl_layers.parameters():
                        assert param.grad is None
            else:
                assert shell_args.freeze_policy_decoder_features
                for param in self.actor_critic.base.rl_layers.parameters():
                    assert param.grad is None

        self.optimizer.step()

        if not decoder_enabled:
            self.actor_critic.base.disable_decoder()

        return (
            total_loss.item(),
            action_loss_total,
            visual_loss_value,
            visual_losses,
            egomotion_loss_value,
            feature_prediction_loss_value,
        )
