# Much of the basic code taken from https://github.com/kevinlu1211/pytorch-decoder-resnet-50-encoder

import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.GroupNorm(32, out_channels)
        self.nonlinearity = nn.ELU(inplace=True)
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.nonlinearity(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the decoder which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels))

    def forward(self, x):
        return self.bridge(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


class UpBlockForHourglassNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
    ):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = Interpolate(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x):
        """
        :param up_x: this is the output from the previous up block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ShallowUpBlockForHourglassNet(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
    ):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = Interpolate(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, up_x):
        """
        :param up_x: this is the output from the previous up block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = self.conv_block(x)
        return x
