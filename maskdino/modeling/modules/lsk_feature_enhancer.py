from collections import OrderedDict

import torch
from torch import nn


class LSKBlock(nn.Module):
    """Large Selective Kernel attention for one feature map.

    The learnable residual scale starts at zero, so enabling this module keeps
    pretrained MaskDINO behavior stable at the beginning of fine-tuning.
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden_channels = max(16, channels // reduction)
        hidden_channels = min(hidden_channels, channels)

        self.dwconv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dwconv7 = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            groups=channels,
        )
        self.proj5 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj7 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.spatial_select = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        branch5 = self.proj5(self.dwconv5(x))
        branch7 = self.proj7(self.dwconv7(self.dwconv5(x)))
        branches = torch.cat([branch5, branch7], dim=1)

        avg_attn = torch.mean(branches, dim=1, keepdim=True)
        max_attn = torch.max(branches, dim=1, keepdim=True)[0]
        select = self.spatial_select(torch.cat([avg_attn, max_attn], dim=1)).sigmoid()

        enhanced = branch5 * select[:, 0:1] + branch7 * select[:, 1:2]
        enhanced = self.proj_out(enhanced)
        return x + self.gamma * enhanced


class LSKFeatureEnhancer(nn.Module):
    """Apply LSKBlock to selected Detectron2 backbone feature maps."""

    def __init__(self, input_shape, in_features, reduction=2):
        super().__init__()
        self.in_features = tuple(in_features)
        blocks = OrderedDict()
        for name in self.in_features:
            if name not in input_shape:
                raise ValueError(f"Feature '{name}' is not in backbone output_shape: {list(input_shape)}")
            blocks[name] = LSKBlock(input_shape[name].channels, reduction=reduction)
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, features):
        if not self.blocks:
            return features

        enhanced = dict(features)
        for name, block in self.blocks.items():
            enhanced[name] = block(features[name])
        return enhanced


def build_feature_enhancer(cfg, input_shape):
    if not cfg.MODEL.FEATURE_ENHANCER.ENABLED:
        return nn.Identity()

    name = cfg.MODEL.FEATURE_ENHANCER.NAME.lower()
    if name != "lsk":
        raise ValueError(f"Unsupported feature enhancer: {cfg.MODEL.FEATURE_ENHANCER.NAME}")

    in_features = cfg.MODEL.FEATURE_ENHANCER.IN_FEATURES
    if len(in_features) == 0:
        in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES

    return LSKFeatureEnhancer(
        input_shape=input_shape,
        in_features=in_features,
        reduction=cfg.MODEL.FEATURE_ENHANCER.REDUCTION,
    )
