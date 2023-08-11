import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_pose.model.config import Config


class PredictionHead(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self._config = config

        self._initial_layers = nn.Sequential(*[
            nn.Conv2d(config.feature_depth, config.feature_depth, kernel_size=3, stride=1, padding=1)
            for _ in range(self._config.n_prediction_head_layers)
        ])

        self._classification_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * config.n_classes, kernel_size=1, stride=1)
        self._box_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * 4, kernel_size=1, stride=1)
        self._mask_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * config.n_prototype_masks, kernel_size=1, stride=1)
        self._point_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * config.n_prototype_points, kernel_size=1, stride=1)

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        x = self._initial_layers(fpn_output)

        classification_output = self._classification_layer(x)
        classification_output = classification_output.reshape(classification_output.size(0), -1, self._config.n_classes).permute(0, 2, 1)
        box_output = self._box_layer(x)
        box_output = box_output.reshape(box_output.size(0), -1, 4).permute(0, 2, 1)
        mask_output = self._mask_layer(x)
        mask_output = mask_output.reshape(mask_output.size(0), -1, self._config.n_prototype_masks).permute(0, 2, 1)
        point_output = self._point_layer(x)
        point_output = point_output.reshape(point_output.size(0), -1, self._config.n_prototype_points).permute(0, 2, 1)

        return (classification_output, box_output, mask_output, point_output)
