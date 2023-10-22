import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

from yolo_pose.model.config import Config


class Pointnet(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        belief_stages = []
        affinity_stages = []

        for i, layer_config in enumerate(config.pointnet_layers):
            in_depth = config.feature_depth if i == 0 else \
                config.feature_depth + config.prototype_belief_depth + config.prototype_affinity_depth

            belief_stage = self._create_stage(
                in_depth,
                config.prototype_belief_depth,
                layer_config,
            )
            affinity_stage = self._create_stage(
                in_depth,
                config.prototype_affinity_depth,
                layer_config,
            )

            belief_stages.append(belief_stage)
            affinity_stages.append(affinity_stage)

        self._belief_stages = nn.ModuleList(belief_stages)
        self._affinity_stages = nn.ModuleList(affinity_stages)

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        belief = self._belief_stages[0].forward(fpn_output)
        affinity = self._affinity_stages[0].forward(fpn_output)

        for (belief_stage, affinity_stage) in zip(self._belief_stages[1:], self._affinity_stages[1:]):
            belief = belief_stage.forward(torch.cat((belief, affinity, fpn_output), dim=1))
            affinity = affinity_stage.forward(torch.cat((belief, affinity, fpn_output), dim=1))

        return belief, affinity

    def _create_stage(self, in_depth, out_depth, layer_config) -> nn.Module:
        padding, kernel_size, layer_count, final_depth = layer_config
        layers = []

        layers.append(nn.Conv2d(
            in_depth,
            self._config.pointnet_feature_depth,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ))

        for i in range(layer_count - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(
                self._config.pointnet_feature_depth,
                self._config.pointnet_feature_depth,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            ))

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(
            self._config.pointnet_feature_depth,
            final_depth,
            kernel_size=1,
            stride=1
        ))

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(
            final_depth,
            out_depth,
            kernel_size=1,
            stride=1
        ))

        return nn.Sequential(*layers)
