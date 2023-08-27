import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

from yolo_pose.model.config import Config


class Pointnet(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._pre_upsample_layers = nn.Sequential(*[
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_pointnet_layers_pre_upsample)
        ])

        self._post_upsample_layers = nn.Sequential(*[
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_pointnet_layers_post_upsample)
        ])

        self._point_output_layer = nn.Conv2d(config.feature_depth, config.n_prototype_points, kernel_size=1, stride=1)
        self._direction_output_layer = nn.Conv2d(config.feature_depth, config.n_prototype_points, kernel_size=1, stride=1)

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        x = self._pre_upsample_layers(fpn_output)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self._post_upsample_layers(x)

        point_output = self._point_output_layer(x)
        point_output = F.sigmoid(point_output)
        direction_output = self._direction_output_layer(x)
        direction_output = torch.clamp(direction_output, -pi, pi)

        return point_output, direction_output
