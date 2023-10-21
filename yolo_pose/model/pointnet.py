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

        self._point_map_layer = nn.Conv2d(config.feature_depth, config.n_point_maps, kernel_size=1, stride=1)

    def forward(self, fpn_output: torch.Tensor) -> torch.Tensor:
        x = self._pre_upsample_layers(fpn_output)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self._post_upsample_layers(x)

        point_map = self._point_map_layer(x)
        point_map = F.sigmoid(point_map)

        return point_map
