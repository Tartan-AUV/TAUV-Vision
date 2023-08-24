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
        self._box_encoding_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * 4, kernel_size=1, stride=1)
        self._mask_coeff_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * config.n_prototype_masks, kernel_size=1, stride=1)
        self._point_coeff_layer = nn.Conv2d(config.feature_depth, len(config.anchor_aspect_ratios) * config.n_prototype_points, kernel_size=1, stride=1)

    def forward(self, fpn_output: torch.Tensor) -> (torch.Tensor, ...):
        x = self._initial_layers(fpn_output)

        classification = self._classification_layer(x)
        classification = classification.reshape(classification.size(0), -1, self._config.n_classes)
        box_encoding = self._box_encoding_layer(x)
        box_encoding = box_encoding.reshape(box_encoding.size(0), -1, 4)
        box_encoding[:, :, 2:4] = 2 * (F.sigmoid(box_encoding[:, :, 2:4]) - 0.5)
        box_encoding[:, :, 0] /= x.size(2)
        box_encoding[:, :, 1] /= x.size(3)
        mask_coeff = self._mask_coeff_layer(x)
        mask_coeff = mask_coeff.reshape(mask_coeff.size(0), -1, self._config.n_prototype_masks)
        point_coeff = self._point_coeff_layer(x)
        point_coeff = point_coeff.reshape(point_coeff.size(0), -1, self._config.n_prototype_points)

        return classification, box_encoding, mask_coeff, point_coeff
