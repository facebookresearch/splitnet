#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

# Much of the basic code taken from https://github.com/kevinlu1211/pytorch-decoder-resnet-50-encoder
from abc import ABC

import numpy as np
import torch
import torchvision.models as models
from a2c_ppo_acktr import model
from a2c_ppo_acktr.utils import init
from dg_util.python_utils import pytorch_util as pt_util
from torch import nn

from networks.building_blocks import ConvBlock, Bridge, ShallowUpBlockForHourglassNet


class EncoderDecoderInterface(nn.Module, ABC):
    def __init__(self, decoder_output_info, create_decoder=True):
        super(EncoderDecoderInterface, self).__init__()
        self.decoder_output_info = decoder_output_info
        self.num_outputs = sum([x[1] for x in self.decoder_output_info])
        self.create_decoder = create_decoder

        self.encoder = None
        self.decoder = None
        self.construct_encoder()
        if self.create_decoder:
            self.construct_decoder()

    def construct_encoder(self):
        raise NotImplementedError

    def construct_decoder(self):
        raise NotImplementedError

    def input_transform(self, x):
        raise NotImplementedError

    @property
    def num_output_channels(self):
        raise NotImplementedError

    def forward(self, x, decoder_enabled):
        x = self.input_transform(x)
        deepest_visual_features = self.encoder(x)
        decoder_outputs = None
        if decoder_enabled:
            decoder_outputs = self.decoder(x)

        return deepest_visual_features, decoder_outputs


class BaseEncoderDecoder(EncoderDecoderInterface, ABC):
    def __init__(self, decoder_output_info, create_decoder=True):
        self.bridge = None
        self.out = None
        super(BaseEncoderDecoder, self).__init__(decoder_output_info, create_decoder)
        self.class_pred_layer = None
        if "semantic" in {info[0] for info in self.decoder_output_info}:
            num_classes = [info[1] for info in self.decoder_output_info if info[0] == "semantic"][0]
            self.class_pred_layer = nn.Sequential(
                nn.Linear(128, 128), nn.ELU(inplace=True), nn.Linear(128, num_classes)
            )

    def construct_decoder(self):
        self.bridge = Bridge(128, 128)
        up_blocks = [
            ShallowUpBlockForHourglassNet(128, 128, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(128, 64, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(64, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
        ]
        self.decoder = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(32, self.num_outputs, kernel_size=1, stride=1)

    def input_transform(self, x):
        x = x.type(torch.float32)
        x = x / 128.0 - 1
        return x

    def forward(self, x, decoder_enabled):
        x = self.input_transform(x)
        deepest_visual_features = self.encoder(x)

        # decoder Part
        decoder_outputs = None
        class_pred = None
        if decoder_enabled:
            x = self.bridge(deepest_visual_features)

            for i, block in enumerate(self.decoder, 1):
                x = block(x)
            decoder_outputs = self.out(x)

            if self.class_pred_layer is not None:
                class_pred_input = torch.mean(deepest_visual_features, dim=(2, 3))
                class_pred = self.class_pred_layer(class_pred_input)

        return deepest_visual_features, decoder_outputs, class_pred


class ShallowVisualEncoder(BaseEncoderDecoder):
    def __init__(self, decoder_output_info, create_decoder=True):
        super(ShallowVisualEncoder, self).__init__(decoder_output_info, create_decoder)

    def construct_encoder(self):
        self.encoder = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, padding=3, kernel_size=7, stride=4),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def num_output_channels(self):
        return 128


class ImagenetModel(nn.Module):
    """Model for pretraining on ImageNet data."""

    def __init__(self):
        super(ImagenetModel, self).__init__()
        self.visual_encoder = ShallowVisualEncoder({}, False)
        self.class_pred_layer = nn.Sequential(nn.Linear(7 * 7 * 128, 1024), nn.ELU(inplace=True), nn.Linear(1024, 1000))

    def forward(self, x):
        x = self.visual_encoder(x, False)[0]
        x = x.view(x.shape[0], -1)
        x = self.class_pred_layer(x)
        return x


class ResNetEncoder(BaseEncoderDecoder):
    def __init__(self, decoder_output_info, create_decoder=True):
        super(ResNetEncoder, self).__init__(decoder_output_info, create_decoder)

    def construct_encoder(self):
        # Remove the avg-pool and fc layers from ResNet
        self.encoder = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

    @property
    def num_output_channels(self):
        return 512

    def input_transform(self, x):
        x = x.type(torch.float32)
        x = x / 255
        x = pt_util.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x

    def forward(self, x, decoder_enabled):
        x = self.input_transform(x)
        x = self.encoder(x)
        return x, None, None


class RLBaseWithVisualEncoder(model.NNBase):
    def __init__(
        self,
        encoder_type,
        decoder_output_info,
        recurrent=False,
        end_to_end=False,
        hidden_size=512,
        target_vector_size=None,
        action_size=None,
        gpu_ids=None,
        create_decoder=True,
        blind=False,
    ):
        assert action_size is not None
        self.aah_im_blind = blind
        self.end_to_end = end_to_end
        self.action_size = action_size
        self.target_vector_size = target_vector_size
        self.decoder_enabled = False
        self.decoder_outputs = None
        self.class_pred = None
        self.visual_encoder_features = None
        self.visual_features = None

        super(RLBaseWithVisualEncoder, self).__init__(
            recurrent,
            recurrent_input_size=hidden_size + self.target_vector_size + self.action_size,
            hidden_size=hidden_size,
        )

        if self.aah_im_blind:
            self.blind_projection = nn.Sequential(
                nn.Linear(
                    self.target_vector_size + self.action_size, hidden_size + self.target_vector_size + self.action_size
                )
            )
        else:
            self.visual_encoder = encoder_type(decoder_output_info, create_decoder)
            self.num_output_channels = self.visual_encoder.num_output_channels

            self.visual_encoder = pt_util.get_data_parallel(self.visual_encoder, gpu_ids)

            self.decoder_output_info = decoder_output_info

            self.visual_projection = nn.Sequential(
                ConvBlock(self.num_output_channels, hidden_size),
                ConvBlock(hidden_size, hidden_size),
                nn.AvgPool2d(2, 2),
                pt_util.RemoveDim((2, 3)),
                nn.Linear(hidden_size * 4 * 4, hidden_size),
            )

        self.rl_layers = nn.Sequential(
            nn.Linear(hidden_size + self.target_vector_size + self.action_size, hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(inplace=True),
        )

        self.egomotion_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ELU(inplace=True), nn.Linear(hidden_size, action_size)
        )

        self.motion_model_layer = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size), nn.ELU(inplace=True), nn.Linear(hidden_size, hidden_size)
        )

        self.critic_linear = init(
            nn.Linear(hidden_size, 1), nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

    @property
    def output_size(self):
        return self._hidden_size

    def enable_decoder(self):
        self.decoder_enabled = True
        self.decoder_outputs = None

    def disable_decoder(self):
        self.decoder_enabled = False
        self.decoder_outputs = None
        pass

    def forward(self, inputs, rnn_hxs, masks):
        images = inputs.get("images")
        visual_encoder_features = inputs.get("visual_encoder_features")
        target_vector = None
        if self.target_vector_size > 0:
            target_vector = inputs.get("target_vector")
        prev_action_one_hot = inputs["prev_action_one_hot"]

        if self.aah_im_blind:
            if target_vector is None:
                rl_features = self.blind_projection(prev_action_one_hot)
            else:
                rl_features = self.blind_projection(torch.cat((target_vector, prev_action_one_hot), dim=1))
        else:
            if visual_encoder_features is not None:
                self.visual_encoder_features = visual_encoder_features
            else:
                # Pool and reshape the features
                self.visual_encoder_features, self.decoder_outputs, self.class_pred = self.visual_encoder(
                    images, self.decoder_enabled
                )
                if not self.end_to_end:
                    self.visual_encoder_features = self.visual_encoder_features.detach()

            self.visual_features = self.visual_projection(self.visual_encoder_features)

            if target_vector is not None:
                rl_features = torch.cat((self.visual_features, target_vector, prev_action_one_hot), dim=1)
            else:
                rl_features = torch.cat((self.visual_features, prev_action_one_hot), dim=1)

        # RL Part
        if self.is_recurrent:
            rl_features, rnn_hxs = self._forward_gru(rl_features, rnn_hxs, masks)
        if target_vector is not None:
            x = self.rl_layers(torch.cat((rl_features, target_vector, prev_action_one_hot), dim=1))
        else:
            x = self.rl_layers(torch.cat((rl_features, prev_action_one_hot), dim=1))

        return self.critic_linear(x), x, rnn_hxs

    def predict_egomotion(self, visual_features_curr, visual_features_prev):
        feature_t_concat = torch.cat((visual_features_curr, visual_features_prev), dim=-1)
        if len(feature_t_concat.shape) > 2:
            feature_t_concat = feature_t_concat.view(-1, self.egomotion_layer[0].weight.shape[1])
        egomotion_pred = self.egomotion_layer(feature_t_concat)
        return egomotion_pred

    def predict_next_features(self, visual_features_curr, action):
        feature_shape = visual_features_curr.shape
        if len(visual_features_curr.shape) > 2:
            visual_features_curr = visual_features_curr.view(
                -1, self.motion_model_layer[0].weight.shape[1] - self.action_size
            )
        if len(action.shape) > 2:
            action = action.view(-1, self.action_size)
        next_features_delta = self.motion_model_layer(torch.cat((visual_features_curr, action), dim=1))
        next_features = visual_features_curr + next_features_delta
        next_features = next_features.view(feature_shape)
        return next_features


class VisualPolicy(model.Policy):
    def __init__(self, action_space, base, base_kwargs=None):
        def base_fn(x, **kwargs):
            return base(**kwargs)

        super(VisualPolicy, self).__init__([None], action_space, base_fn, base_kwargs)
        self.num_actions = action_space.n
        self.last_dist = None

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        self.last_dist = dist

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        self.last_dist = dist

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
